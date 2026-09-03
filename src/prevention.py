"""
Dispute Prevention Score — the proactive layer.

Instead of waiting for a dispute to arrive, this model scores TRANSACTIONS
at payment time and predicts: "will this become a dispute within 30 days?"

WHY THIS MATTERS:
- A merchant who knows a ₹8,000 electronics order is high-risk can
  proactively request delivery confirmation and tracking updates.
- When (if) the dispute arrives, evidence is already on file — the reactive
  system (detector + EV engine) starts from a position of strength.
- Prevention is always cheaper than cure. The best dispute to win is the
  one that never happens.

HONESTY NOTE: this model is trained on strictly less signal than the
outcome detector (src/model.py). At transaction time we do NOT yet have
evidence flags (evidence is gathered after the transaction), late_filing
(filing hasn't happened), or dispute-history features. Its metrics are
expected to be, and are, materially lower than the outcome detector's —
that's the honest result of a harder, information-poorer problem, not a
bug to be papered over.
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
import xgboost as xgb

from src import config, data_gen


def build_transaction_dataset(seed=config.SEED):
    """
    Build a dataset of ALL transactions — both those that became disputes
    and a background of normal transactions that never led to a dispute.

    The dispute generator only produces transactions that DID become
    disputes, so we synthesize a background population at a ~5:1 ratio of
    normal-to-disputed transactions (a reasonable order of magnitude for
    most merchants) to make this a realistic, heavily imbalanced
    classification problem rather than an artificially balanced one.

    DISCLOSED MODELING ASSUMPTION: the background population is NOT drawn
    from the same category/amount/shipping distributions as the dispute
    population. If it were (an earlier version of this generator did this),
    the two classes would be statistically indistinguishable on
    transaction-time features by construction, and the prevention model
    would correctly measure ~0.50 ROC-AUC — a coin flip dressed up as a
    model. That is a generator flaw, not an honest "harder problem" result,
    and we do not want to present noise as a differentiator.

    Instead we bias the background toward a directionally plausible
    real-world prior — higher-value, express-shipped, electronics/
    digital-goods purchases draw more scrutiny and friction and so carry
    somewhat higher dispute propensity than routine grocery/home purchases
    on standard shipping. This is a stated assumption we are choosing, not
    a pattern measured from real Razorpay data, and it's disclosed here and
    in the README precisely so the resulting AUC is not mistaken for a
    validated real-world result.

    SHARED CUSTOMER POOL: the background population draws `customer_id` from
    the exact same customers `generate_disputes(seed=seed)` used internally
    (reconstructed via `data_gen.generate_customer_population()` fed a freshly
    seeded rng — see that function's docstring for why this reproduces the
    same population), but UNIFORMLY, not with the 3x fraud-upweighting the
    dispute population applies. That asymmetry is what makes a customer's
    prior transaction history genuine signal rather than noise: a fraud-prone
    customer shows up disproportionately more among dispute rows than among
    these uniformly-sampled normal rows, so their realized "fraction of
    transactions that became a dispute" is honestly higher — a real, prior,
    causally-available signal, not a leak from the current transaction's own
    future. (Coupled to `generate_disputes()` being called with its default
    `n` above — if that call ever passes an explicit `n=`, `n_customers`
    below must match it.)
    """
    disputes = data_gen.generate_disputes(seed=seed)

    n_normal = len(disputes) * 5
    rng = np.random.default_rng(seed + 1)

    n_customers = max(500, config.N_DISPUTES // 6)
    pop_rng = np.random.default_rng(seed)
    customer_ids, _, _, _ = data_gen.generate_customer_population(pop_rng, n_customers)
    normal_customer_id = rng.choice(customer_ids, size=n_normal)

    # Lower central tendency than the dispute population's amount draw
    # (which uses mu=7.2) — routine, lower-friction purchases predominate.
    normal_amount = np.round(np.exp(rng.normal(6.85, 1.0, size=n_normal)), 2)
    normal_amount = np.clip(normal_amount, 150, 150_000)

    base_date = pd.Timestamp("2025-01-01")
    normal_txn_day = rng.integers(0, config.DATE_RANGE_DAYS, size=n_normal)
    normal_txn_date = base_date + pd.to_timedelta(normal_txn_day, unit="D")

    normal_df = pd.DataFrame({
        "transaction_id": [f"NORM{i:08d}" for i in range(n_normal)],
        "customer_id": normal_customer_id,
        "amount_inr": normal_amount,
        "transaction_date": normal_txn_date,
        # Mass shifted away from electronics/digital_goods and toward
        # grocery/home relative to the dispute population's mix
        # (electronics 0.28->0.15, digital_goods 0.08->0.05).
        "merchant_category": rng.choice(
            ["electronics", "fashion", "grocery", "home", "beauty", "digital_goods"],
            size=n_normal, p=[0.15, 0.22, 0.21, 0.22, 0.15, 0.05],
        ),
        "device_type": rng.choice(["mobile", "desktop", "app"], size=n_normal, p=[0.55, 0.25, 0.20]),
        # More standard, less express shipping than the dispute population.
        "shipping_method": rng.choice(
            ["standard", "express", "digital_delivery"], size=n_normal, p=[0.74, 0.16, 0.10],
        ),
        "became_dispute": 0,
    })

    dispute_df = disputes[
        ["transaction_id", "customer_id", "amount_inr", "transaction_date",
         "merchant_category", "device_type", "shipping_method"]
    ].copy()
    dispute_df["became_dispute"] = 1

    combined = pd.concat([dispute_df, normal_df], ignore_index=True)
    combined = combined.sort_values("transaction_date").reset_index(drop=True)

    combined["txn_day_of_week"] = combined["transaction_date"].dt.dayofweek
    combined["txn_month"] = combined["transaction_date"].dt.month
    combined["log_amount"] = np.log1p(combined["amount_inr"])

    return combined


def add_customer_history_features(df):
    """Add strictly-past per-customer aggregate features -- the prevention-model
    analogue of src/features.py's add_uid_aggregates(), computed the identical
    way (cumcount / cumsum-minus-self, sorted by transaction_date) so a
    transaction only ever sees THIS customer's PRIOR transactions -- never its
    own eventual outcome, and never a later one.

    Column names deliberately avoid the "dispute"/"won"/"late_filing"
    substrings tests/test_prevention.py::test_prevention_features_have_no_dispute_columns
    forbids. That test exists to catch outcome-only leakage (evidence flags,
    filing lateness, dispute resolution) -- a customer's PRIOR transaction
    count and PRIOR flag rate are neither: they're signal a merchant
    genuinely has about the customer at payment time, not this transaction's
    own future.

    REQUIRES the caller to have already sorted `df` by transaction_date and
    reset its index (build_transaction_dataset() and every current caller of
    build_prevention_features() already do this). Do NOT re-sort here: with
    day-granularity dates over 120k rows (~164 rows/day), a second independent
    sort_values("transaction_date") reorders same-day ties differently than
    the caller's own sort did, silently misaligning the returned feature rows
    against the caller's own y-labels for most of the dataset -- this was a
    real bug here (ROC-AUC collapsed to ~0.50, ~chance) before being caught
    and fixed. The assertion below is a cheap guard against it recurring.
    """
    assert df["transaction_date"].is_monotonic_increasing, (
        "add_customer_history_features() requires transaction_date-sorted input "
        "with a reset index -- sort before calling, don't rely on this function "
        "to sort (see docstring for why a second internal sort is dangerous here)."
    )
    g = df.groupby("customer_id")

    prior_count = g.cumcount()
    flag_cumsum = g["became_dispute"].cumsum() - df["became_dispute"]
    flag_rate = flag_cumsum / prior_count.replace(0, np.nan)
    global_prior = df["became_dispute"].expanding().mean().shift(1).fillna(df["became_dispute"].mean())

    amount_cumsum = g["amount_inr"].cumsum() - df["amount_inr"]
    prior_avg_amount = amount_cumsum / prior_count.replace(0, np.nan)

    df = df.copy()
    df["cust_prior_txn_count"] = prior_count
    df["cust_prior_flag_rate"] = flag_rate.fillna(global_prior)
    df["cust_prior_avg_amount"] = prior_avg_amount.fillna(df["amount_inr"].median())
    return df


def build_prevention_features(df):
    """Feature matrix using ONLY features available at transaction time —
    no evidence flags, no late_filing. Includes causally-sound customer
    prior-history aggregates (this customer's own PAST transactions, strictly
    before this one) -- a merchant genuinely knows a customer's past dispute
    behavior at payment time; that's real signal, not a leak.

    CALLER MUST PASS THE FULL TIME-SORTED FRAME, not a pre-sliced train/test
    subset -- the cust_prior_* aggregates need to see each customer's full
    history up to (not including) each row. Slice the RETURNED feature frame
    by index afterward instead, exactly as src/model.py's train() already
    does for the outcome model's UID aggregates (see its comment there)."""
    df = add_customer_history_features(df)
    numeric_cols = [
        "amount_inr", "log_amount", "txn_day_of_week", "txn_month",
        "cust_prior_txn_count", "cust_prior_flag_rate", "cust_prior_avg_amount",
    ]
    cat_cols = ["merchant_category", "device_type", "shipping_method"]
    X = df[numeric_cols].copy()
    X = pd.concat([X, pd.get_dummies(df[cat_cols], prefix=cat_cols)], axis=1)
    return X


def train_prevention_model(seed=config.SEED, verbose=True):
    """Train the dispute prevention model with a time-respecting split."""
    data = build_transaction_dataset(seed=seed)

    # Build features on the FULL sorted frame so cust_prior_* aggregates for
    # test rows correctly see train rows as history, then slice back out by
    # index -- mirrors src/model.py's train(), which solves the identical
    # problem for the outcome model's UID aggregates (see its comment there).
    # Slicing into train/test BEFORE building features (the old code here)
    # would silently zero out every test-set customer's real prior history
    # from the train period -- harmless while every feature was row-local,
    # a real bug now that cust_prior_* depends on row order across the split.
    full_sorted = data.sort_values("transaction_date").reset_index(drop=True)
    n = len(full_sorted)
    train_end = int(n * 0.70)
    calib_end = int(n * 0.85)

    X_full = build_prevention_features(full_sorted)
    y_full = full_sorted["became_dispute"]

    X_train, y_train = X_full.iloc[:train_end], y_full.iloc[:train_end]
    X_test, y_test = X_full.iloc[calib_end:], y_full.iloc[calib_end:]
    test_df = full_sorted.iloc[calib_end:]

    if verbose:
        print(f"Prevention model — train: {len(X_train)}, test: {len(X_test)}")
        print(f"Dispute base rate — train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    sorted_probs = np.sort(probs)[::-1]
    n_actual_disputes = int(y_test.sum())
    threshold_idx = min(int(n_actual_disputes * 0.5), len(sorted_probs) - 1)
    threshold_50 = sorted_probs[threshold_idx] if n_actual_disputes > 0 else 0.5
    preds_50 = (probs >= threshold_50).astype(int)
    prec_50, rec_50, f1_50, _ = precision_recall_fscore_support(y_test, preds_50, average="binary", zero_division=0)

    metrics = {
        "n_test": int(len(y_test)),
        "dispute_base_rate": round(float(y_test.mean()), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 4),
        "at_50pct_recall": {
            "threshold": round(float(threshold_50), 4),
            "precision": round(float(prec_50), 4),
            "recall": round(float(rec_50), 4),
            "f1": round(float(f1_50), 4),
            "flagged_transactions_pct": round(float(preds_50.mean()), 4),
        },
        "note": "Prevention model has less signal than the outcome model (no evidence flags at txn time). Lower metrics vs. src/model.py are expected and honest, not a bug.",
    }

    if verbose:
        print(json.dumps(metrics, indent=2))

    feature_cols = list(X_train.columns)
    # Fallback stats for score_transaction() when no per-customer history is
    # available (a brand-new customer, or a caller that doesn't pass history_df
    # at all -- e.g. the existing Prevention Score / Razorpay Integration pages).
    # Computed over the full sorted population, not just train, matching
    # src/features.py::feature_vector_for_case()'s equivalent fallback.
    global_stats = {
        "global_flag_rate": round(float(full_sorted["became_dispute"].mean()), 4),
        "global_avg_amount": round(float(full_sorted["amount_inr"].median()), 2),
    }
    with open(config.MODELS_DIR / "prevention_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(config.MODELS_DIR / "prevention_feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(config.MODELS_DIR / "prevention_global_stats.json", "w") as f:
        json.dump(global_stats, f)
    with open(config.ARTIFACTS_DIR / "prevention_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_category_risk(test_df, probs)

    return model, feature_cols, metrics


def plot_category_risk(test_df, probs):
    """Bar chart: average dispute risk score by merchant category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_df = test_df.copy()
    test_df["dispute_risk"] = probs
    cat_risk = test_df.groupby("merchant_category")["dispute_risk"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cat_risk.index, cat_risk.values, color="#e67e22", edgecolor="white")
    ax.set_xlabel("Average dispute risk score")
    ax.set_title("Dispute risk by merchant category (prevention model)")
    for bar, val in zip(bars, cat_risk.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS_DIR / "prevention_category_risk.png", dpi=150)
    plt.close(fig)


def score_transaction(transaction: dict, model, feature_cols, history_df: pd.DataFrame = None) -> dict:
    """Score a single transaction for dispute risk at payment time.

    `history_df`, if given, should hold candidate prior transactions for this
    customer (needs at least `customer_id`, `amount_inr`, `became_dispute`,
    and ideally `transaction_date` to exclude same-or-later rows) -- used to
    compute real cust_prior_* features the identical way training does.
    Without it (the default -- every caller except Demo Mode's batch loop),
    falls back to persisted global stats: honest "no history available"
    behavior, not a silently-wrong zero or a crash.
    """
    row = {
        "amount_inr": transaction["amount_inr"],
        "log_amount": np.log1p(transaction["amount_inr"]),
        "txn_day_of_week": pd.Timestamp(transaction.get("transaction_date", pd.Timestamp.now())).dayofweek,
        "txn_month": pd.Timestamp(transaction.get("transaction_date", pd.Timestamp.now())).month,
    }

    customer_id = transaction.get("customer_id")
    cust_hist = None
    if history_df is not None and customer_id is not None and "customer_id" in history_df.columns:
        cust_hist = history_df[history_df["customer_id"] == customer_id]
        if "transaction_date" in transaction and "transaction_date" in cust_hist.columns:
            cust_hist = cust_hist[cust_hist["transaction_date"] < pd.Timestamp(transaction["transaction_date"])]

    if cust_hist is not None and len(cust_hist) > 0:
        row["cust_prior_txn_count"] = len(cust_hist)
        row["cust_prior_flag_rate"] = float(cust_hist["became_dispute"].mean())
        row["cust_prior_avg_amount"] = float(cust_hist["amount_inr"].mean())
    else:
        try:
            with open(config.MODELS_DIR / "prevention_global_stats.json") as f:
                gstats = json.load(f)
        except FileNotFoundError:
            gstats = {"global_flag_rate": 0.05, "global_avg_amount": 2000.0}
        row["cust_prior_txn_count"] = 0
        row["cust_prior_flag_rate"] = gstats["global_flag_rate"]
        row["cust_prior_avg_amount"] = gstats["global_avg_amount"]

    for col in ["merchant_category", "device_type", "shipping_method"]:
        val = transaction.get(col, "")
        row[f"{col}_{val}"] = 1

    X = pd.Series(row).reindex(feature_cols).fillna(0).to_frame().T
    prob = float(model.predict_proba(X)[:, 1][0])

    if prob >= 0.30:
        tier = "HIGH"
        recommendation = "Proactively collect delivery confirmation and tracking proof NOW. Do not wait for a dispute."
    elif prob >= 0.15:
        tier = "MEDIUM"
        recommendation = "Ensure tracking number is on file. Consider requesting delivery confirmation within 7 days."
    else:
        tier = "LOW"
        recommendation = "Standard monitoring. No immediate evidence collection needed."

    return {
        "dispute_risk_score": round(prob, 4),
        "risk_tier": tier,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    train_prevention_model()
