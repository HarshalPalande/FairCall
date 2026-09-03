"""
Root-cause intelligence — WHY disputes are lost, and what to change.

Every other component here is predictive ("will this dispute be won?") or
decisive ("what should we do about it?"). This one is PRESCRIPTIVE: it looks
across the whole dispute population and answers "what should this merchant
change about how they operate?"

Two things it deliberately does differently from a naive analytics dashboard:

1. IT RANKS BY MONEY, NOT BY RATE. The segment with the worst win rate is
   usually not the segment costing the most, because volume dominates. A
   merchant fixing things in win-rate order fixes the wrong thing first.
   Expected loss (amount x probability of losing) is the number to sort on.

2. IT SEPARATES CORRELATION FROM CAUSATION, LOUDLY. The observational gap
   between "disputes with a tracking number" and "disputes without one" is
   NOT the value of adding tracking numbers. In this dataset both evidence
   availability and win probability are driven by the same latent customer
   trust (`src/data_gen.py`: evidence_base_rate = 0.35 + 0.5*trust, and
   win_logit carries 1.8*(trust-0.5)) -- so the observed gap is confounded
   by construction, and quoting it as "recoverable revenue" would be exactly
   the kind of number this project refuses to publish elsewhere.

   The defensible estimate instead re-scores each dispute with the evidence
   flag flipped, using the SAME calibrated model (`src/counterfactual.py`),
   holding everything else fixed. That's a model-based counterfactual, not a
   population difference. Both are reported side by side so the gap between
   them is visible rather than hidden.
"""
import numpy as np
import pandas as pd

from src import config
from src.counterfactual import rank_evidence_impact
from src.features import build_feature_frame


def segment_loss_analysis(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Loss profile per segment, ranked by expected rupees lost.

    expected_loss_inr = sum over the segment of amount x (1 - won), i.e. the
    money actually lost in that segment. Ranked on this rather than win rate
    because a merchant's remediation budget is finite and should go where the
    money is, not where the percentage looks worst.
    """
    g = df.groupby(by)
    out = pd.DataFrame({
        "disputes": g.size(),
        "win_rate": g["won"].mean(),
        "avg_amount_inr": g["amount_inr"].mean(),
        "expected_loss_inr": g.apply(
            lambda x: float((x["amount_inr"] * (1 - x["won"])).sum()), include_groups=False
        ),
    })
    out["pct_of_total_loss"] = out["expected_loss_inr"] / out["expected_loss_inr"].sum()
    return out.sort_values("expected_loss_inr", ascending=False)


def structurally_impossible_mask(df: pd.DataFrame, evidence_type: str) -> pd.Series:
    """Which rows CANNOT have this evidence, by the nature of the order.

    Digital delivery has no parcel, so there is no tracking number and no
    signed proof of delivery to collect -- `src/data_gen.py` enforces this
    when generating. Counting those as "missing evidence the merchant should
    go collect" would be recommending an impossible action, so they're
    excluded from the addressable gap rather than silently inflating it.
    """
    if evidence_type in ("tracking_number", "signed_pod"):
        return df["shipping_method"] == "digital_delivery"
    return pd.Series(False, index=df.index)


def evidence_gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Observational win-rate gap per evidence type. CONFOUNDED BY DESIGN --
    see the module docstring. Reported for transparency next to the
    model-based counterfactual, never as a standalone "recoverable" figure."""
    rows = []
    for etype in config.EVIDENCE_TYPES:
        col = f"has_{etype}"
        if col not in df.columns:
            continue
        impossible = structurally_impossible_mask(df, etype)
        present = df[df[col] == 1]
        absent = df[(df[col] == 0) & (~impossible)]
        rows.append({
            "evidence_type": etype,
            "missing_addressable": len(absent),
            "missing_structurally_impossible": int(impossible.sum()),
            "win_rate_with": float(present["won"].mean()) if len(present) else np.nan,
            "win_rate_without": float(absent["won"].mean()) if len(absent) else np.nan,
            "observational_gap_pp": (
                float((present["won"].mean() - absent["won"].mean()) * 100)
                if len(present) and len(absent) else np.nan
            ),
            "amount_at_stake_inr": float(absent["amount_inr"].sum()),
        })
    return pd.DataFrame(rows).sort_values("amount_at_stake_inr", ascending=False)


def counterfactual_evidence_value(
    df: pd.DataFrame, model, calibrator, feature_cols: list, sample_size: int = 300, seed: int = config.SEED
) -> pd.DataFrame:
    """Model-based value of adding each evidence type, the defensible estimate.

    For a sample of disputes, flips each MISSING evidence flag 0->1 and
    re-scores with the same calibrated model, holding every other feature
    fixed. Averages the predicted win-probability lift, then scales it by the
    money actually at stake in those disputes.

    This is an estimate under the model, not a promise: it inherits whatever
    the model gets wrong, and it assumes the evidence could have been
    collected at all (structurally impossible cases are excluded upstream).
    """
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    X, df_feat = build_feature_frame(sample)

    lifts = {e: [] for e in config.EVIDENCE_TYPES}
    amounts = {e: [] for e in config.EVIDENCE_TYPES}

    for pos in range(len(df_feat)):
        row = df_feat.iloc[pos]
        # astype(float): a row slice out of build_feature_frame's mixed-dtype
        # frame (bool one-hots + float numerics) comes back as an object-dtype
        # Series, which XGBoost rejects. pipeline.py doesn't hit this because it
        # builds its Series from an all-numeric dict; cast here rather than
        # changing rank_evidence_impact's contract for its existing callers.
        results = rank_evidence_impact(model, calibrator, X.iloc[pos].astype(float), feature_cols)
        for r in results:
            impossible = (
                r.evidence_type in ("tracking_number", "signed_pod")
                and row["shipping_method"] == "digital_delivery"
            )
            if r.current_value == 0 and not impossible:
                lifts[r.evidence_type].append(r.delta)
                amounts[r.evidence_type].append(float(row["amount_inr"]))

    rows = []
    for etype in config.EVIDENCE_TYPES:
        n = len(lifts[etype])
        if n == 0:
            continue
        mean_lift = float(np.mean(lifts[etype]))
        rows.append({
            "evidence_type": etype,
            "sampled_missing": n,
            "mean_predicted_lift_pp": mean_lift * 100,
            # Value per rupee at stake: a win recovers the amount instead of losing
            # it, so a lift of p on an amount A is worth A*p in expectation.
            "modelled_value_per_1k_at_stake_inr": mean_lift * 1000,
            "sampled_amount_at_stake_inr": float(np.sum(amounts[etype])),
        })
    return pd.DataFrame(rows).sort_values("mean_predicted_lift_pp", ascending=False)


def late_filing_analysis(df: pd.DataFrame) -> dict:
    """Operational finding: late-filed disputes and what they cost.

    Unlike evidence availability, filing promptly is entirely within the
    merchant's control and has no structural blockers -- which makes it the
    most directly actionable line in this whole module.
    """
    late = df[df["late_filing"] == 1]
    on_time = df[df["late_filing"] == 0]
    return {
        "late_count": len(late),
        "late_share": len(late) / len(df) if len(df) else 0.0,
        "win_rate_late": float(late["won"].mean()) if len(late) else np.nan,
        "win_rate_on_time": float(on_time["won"].mean()) if len(on_time) else np.nan,
        "gap_pp": (
            float((on_time["won"].mean() - late["won"].mean()) * 100)
            if len(late) and len(on_time) else np.nan
        ),
        "expected_loss_late_inr": float((late["amount_inr"] * (1 - late["won"])).sum()),
    }
