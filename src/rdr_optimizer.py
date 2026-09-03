"""
RDR Rule Optimizer.

Visa lets a merchant configure a limited number of RDR rule scenarios (up to 10)
using attributes like transaction amount and reason code. Any incoming dispute
matching a rule is auto-refunded pre-dispute, and for non-fraud codes that keeps
the event off the VAMP ratio.

The question this answers:

    Given my historical dispute distribution and my current VAMP headroom, what
    amount threshold should I set for auto-resolution to minimise TOTAL cost —
    refund outlay plus VAMP exposure plus representment costs?

That is a genuine trade-off, not a heuristic. Auto-resolving more disputes costs
more in refunds but buys back VAMP headroom; auto-resolving fewer preserves
revenue but risks crossing the Excessive threshold, which applies penalties to
the merchant's entire dispute volume rather than just the marginal case.

The optimizer sweeps candidate thresholds and returns the FULL cost curve, so the
trade-off is inspectable rather than hidden behind a single recommended number.

TWO HONEST NOTES ON THE MODEL, because the output is a ₹ recommendation:

  1. The RDR threshold is capped at the hard ceiling (config.HARD_CEILING_INR).
     Auto-refunding is an automated money movement, so it is bound by the same
     "no large automated decision without a human" rule as every other automated
     action in src/decision_engine.py. A sweep that recommended auto-refunding
     ₹80,000 disputes would be recommending something the decision engine will
     refuse to do.

  2. Escalated disputes are priced as accepted losses. That is a deliberately
     pessimistic floor, not a prediction: a human analyst would contest some of
     them and recover value. It means the "remaining slice" cost is an upper
     bound, and the optimizer is therefore mildly biased TOWARD auto-resolving.
     Stated here rather than left implicit — the backtest in src/backtest.py
     models analyst behaviour explicitly and this module does not.
"""
import numpy as np
import pandas as pd

from src import config, vamp


def evaluate_rdr_threshold(
    scored_df: pd.DataFrame,
    amount_threshold: float,
    reason_code: str = config.REASON_CODE,
    vamp_status=None,
) -> dict:
    """Simulate: auto-resolve every dispute at or below `amount_threshold` via RDR,
    handle the rest through the existing contest/escalate logic.

    `scored_df` needs columns: amount_inr, actual_won, win_probability,
    system_action (as produced by src.backtest.score_test_set).
    """
    if vamp_status is None:
        vamp_status = vamp.compute_vamp_status()

    vamp_excluded = reason_code in config.RDR_VAMP_EXCLUDED_REASON_CODES
    marginal_vamp = vamp.marginal_vamp_cost(vamp_status)

    auto_resolved = scored_df[scored_df["amount_inr"] <= amount_threshold]
    remaining = scored_df[scored_df["amount_inr"] > amount_threshold]

    # --- Cost of the RDR-resolved slice ---
    refund_outlay = auto_resolved["amount_inr"].sum()
    rdr_fees = len(auto_resolved) * config.RDR_RESOLUTION_FEE_INR
    rdr_vamp_cost = 0.0 if vamp_excluded else len(auto_resolved) * marginal_vamp
    rdr_total_cost = refund_outlay + rdr_fees + rdr_vamp_cost

    # --- Cost of the remaining slice (existing contest/escalate behaviour) ---
    contested = remaining[remaining["system_action"] == "AUTO_CONTEST"]
    escalated = remaining[remaining["system_action"] == "ESCALATE"]

    contest_outcomes = np.where(
        contested["actual_won"] == 1,
        contested["amount_inr"] - config.CONTEST_COST_INR,
        -(contested["amount_inr"] + config.CONTEST_COST_INR),
    )
    contest_net = float(contest_outcomes.sum()) if len(contested) else 0.0

    # Pessimistic floor — see note 2 in the module docstring.
    escalate_loss = float(escalated["amount_inr"].sum()) if len(escalated) else 0.0

    # Every remaining dispute still counts toward VAMP.
    remaining_vamp_cost = len(remaining) * marginal_vamp

    # Costs are positive numbers here; contest_net is a signed outcome, so a
    # positive net recovery reduces cost.
    remaining_total = -contest_net + escalate_loss + remaining_vamp_cost

    # --- VAMP position under this policy ---
    events_removed = len(auto_resolved) if vamp_excluded else 0
    scale = (vamp_status.monthly_dispute_events / len(scored_df)) if len(scored_df) else 1.0
    monthly_events_removed = int(events_removed * scale)
    new_monthly_events = max(0, vamp_status.monthly_dispute_events - monthly_events_removed)
    new_status = vamp.compute_vamp_status(
        monthly_settled_txns=vamp_status.monthly_settled_txns,
        monthly_dispute_events=new_monthly_events,
        threshold=vamp_status.threshold,
    )

    return {
        "amount_threshold_inr": amount_threshold,
        "disputes_auto_resolved": len(auto_resolved),
        "disputes_remaining": len(remaining),
        "pct_auto_resolved": round(len(auto_resolved) / len(scored_df), 4) if len(scored_df) else 0,
        "refund_outlay_inr": round(float(refund_outlay), 2),
        "rdr_fees_inr": round(float(rdr_fees), 2),
        "rdr_total_cost_inr": round(float(rdr_total_cost), 2),
        "remaining_slice_cost_inr": round(float(remaining_total), 2),
        "total_cost_inr": round(float(rdr_total_cost + remaining_total), 2),
        "vamp_ratio_before_pct": round(vamp_status.current_ratio * 100, 4),
        "vamp_ratio_after_pct": round(new_status.current_ratio * 100, 4),
        "vamp_headroom_before": vamp_status.headroom_events,
        "vamp_headroom_after": new_status.headroom_events,
        "monthly_events_removed_from_ratio": monthly_events_removed,
        "crosses_threshold_before": vamp_status.is_excessive,
        "crosses_threshold_after": new_status.is_excessive,
    }


def optimize_rdr_threshold(
    scored_df: pd.DataFrame,
    reason_code: str = config.REASON_CODE,
    n_candidates: int = 25,
    vamp_status=None,
    max_threshold: float = None,
) -> pd.DataFrame:
    """Sweep candidate amount thresholds and return the full cost curve.

    The sweep is capped at the hard ceiling by default: RDR is an automated money
    movement and is bound by the same ceiling as every other automated action, so
    recommending a threshold above it would recommend something
    `src.decision_engine` will refuse to execute.
    """
    if len(scored_df) == 0:
        return pd.DataFrame()

    if max_threshold is None:
        max_threshold = min(
            float(scored_df["amount_inr"].quantile(0.95)), config.HARD_CEILING_INR
        )

    candidates = np.linspace(0, max_threshold, n_candidates)
    rows = [
        evaluate_rdr_threshold(scored_df, float(t), reason_code, vamp_status)
        for t in candidates
    ]
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> dict:
    """The minimum-total-cost row from an optimizer sweep."""
    if curve.empty:
        return {}
    return curve.loc[curve["total_cost_inr"].idxmin()].to_dict()
