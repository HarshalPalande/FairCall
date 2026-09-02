"""
Synthetic dataset generator — Visa reason code 13.1 ("Merchandise / Services
Not Received") disputes only.

HONESTY NOTE (see README "Dataset" section): this data is fully synthetic.
We generate it because it lets us (a) control and disclose the exact
data-generating process, (b) guarantee reproducibility with a fixed seed,
and (c) avoid presenting labels derived from a repurposed dataset (e.g.
IEEE-CIS, which is a fraud-detection dataset, not a chargeback-outcome
dataset) as if they were real chargeback win/loss outcomes. Every reported
metric in this project is a metric against synthetic ground truth, not
real-world performance, and is described that way everywhere it's reported.

The generator encodes, by construction, the root causes of dispute loss
named in the project brief: missing evidence, weak/inconsistent evidence,
late submission, and genuine unwinnable fraud. This gives the downstream
model and rule-based components real, recoverable structure to learn from
and check against, instead of pure noise.
"""
import numpy as np
import pandas as pd

from src import config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_disputes(n=config.N_DISPUTES, seed=config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- customer population -------------------------------------------
    n_customers = max(500, n // 6)
    customer_ids = np.array([f"CUST{i:06d}" for i in range(n_customers)])
    # latent, never exposed as a feature: genuine long-term trustworthiness
    customer_trust = rng.beta(3, 2, size=n_customers)
    # ~7% of customers are chronic abusers / account-takeover victims whose
    # disputes are effectively unwinnable regardless of evidence quality.
    is_fraud_customer = rng.random(n_customers) < 0.07
    customer_first_seen = rng.integers(0, config.DATE_RANGE_DAYS // 2, size=n_customers)

    # Fraud customers dispute more often (fits real-world skew).
    customer_weights = np.where(is_fraud_customer, 3.0, 1.0)
    customer_weights = customer_weights / customer_weights.sum()
    cust_idx = rng.choice(n_customers, size=n, p=customer_weights)

    customer_id = customer_ids[cust_idx]
    trust = customer_trust[cust_idx]
    is_fraud = is_fraud_customer[cust_idx]
    first_seen = customer_first_seen[cust_idx]

    # --- transaction / dispute timing -----------------------------------
    txn_day_offset = np.array([
        rng.integers(fs, config.DATE_RANGE_DAYS) for fs in first_seen
    ])
    base_date = pd.Timestamp("2025-01-01")
    transaction_date = base_date + pd.to_timedelta(txn_day_offset, unit="D")
    dispute_delay_days = rng.integers(2, 45, size=n)  # includes "late" filers
    dispute_date = transaction_date + pd.to_timedelta(dispute_delay_days, unit="D")

    # --- transaction attributes ------------------------------------------
    amount = np.round(np.exp(rng.normal(7.2, 1.0, size=n)), 2)  # skewed, INR
    amount = np.clip(amount, 150, 150_000)

    merchant_category = rng.choice(
        ["electronics", "fashion", "grocery", "home", "beauty", "digital_goods"],
        size=n, p=[0.28, 0.24, 0.12, 0.16, 0.12, 0.08],
    )
    device_type = rng.choice(["mobile", "desktop", "app"], size=n, p=[0.55, 0.25, 0.20])
    shipping_method = rng.choice(
        ["standard", "express", "digital_delivery"], size=n, p=[0.62, 0.28, 0.10]
    )

    # --- evidence availability --------------------------------------------
    # Genuine merchants with good ops tend to have more complete evidence;
    # fraud/ATO cases often have *some* evidence present but it's weaker,
    # which is what makes them deceptively hard, not obviously empty.
    evidence_base_rate = 0.35 + 0.5 * trust
    evidence = {}
    for etype in config.EVIDENCE_TYPES:
        noise = rng.normal(0, 0.08, size=n)
        p = np.clip(evidence_base_rate + noise, 0.03, 0.97)
        evidence[etype] = (rng.random(n) < p).astype(int)
    # digital_delivery orders can't have a signed POD or tracking number —
    # structurally absent, not "missing evidence" in the failure sense.
    digital_mask = shipping_method == "digital_delivery"
    evidence["signed_pod"][digital_mask] = 0
    evidence["tracking_number"][digital_mask] = 0

    late_filing = (dispute_delay_days > 30).astype(int)

    # --- per-customer history proxies (raw, still causal-safe fields) ----
    account_age_days_at_dispute = txn_day_offset - first_seen

    # --- latent win-probability model --------------------------------
    completeness = np.mean([evidence[e] for e in config.REQUIRED_EVIDENCE_FOR_13_1], axis=0)
    all_evidence_mean = np.mean([evidence[e] for e in config.EVIDENCE_TYPES], axis=0)

    amount_z = (np.log(amount) - np.log(amount).mean()) / np.log(amount).std()

    win_logit = (
        -0.4
        + 3.4 * completeness
        + 1.1 * all_evidence_mean
        + 1.8 * (trust - 0.5)
        - 0.9 * late_filing
        - 0.15 * amount_z
        + rng.normal(0, 0.6, size=n)
    )
    # Genuine fraud disputes are capped near-unwinnable regardless of
    # evidence — this is the "genuine fraud" root cause from the brief.
    win_logit = np.where(is_fraud, np.minimum(win_logit, -1.6) + rng.normal(0, 0.3, size=n), win_logit)

    win_prob_true = _sigmoid(win_logit)
    won = (rng.random(n) < win_prob_true).astype(int)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(n)],
        "customer_id": customer_id,
        "reason_code": config.REASON_CODE,
        "transaction_date": transaction_date,
        "dispute_date": dispute_date,
        "amount_inr": amount,
        "merchant_category": merchant_category,
        "device_type": device_type,
        "shipping_method": shipping_method,
        "late_filing": late_filing,
        "account_age_days_at_dispute": account_age_days_at_dispute,
        **{f"has_{e}": evidence[e] for e in config.EVIDENCE_TYPES},
        "won": won,
    })
    df = df.sort_values("dispute_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_disputes()
    out_path = config.DATA_DIR / "disputes.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} synthetic reason-code-{config.REASON_CODE} disputes to {out_path}")
    print(f"Base win rate: {df['won'].mean():.3f}")
