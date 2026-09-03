"""
VAMP (Visa Acquirer Monitoring Program) risk modelling.

WHY THIS EXISTS:
The EV Decision Engine originally priced a dispute at its transaction amount.
That is incomplete. Under VAMP, every dispute also increments a portfolio-level
ratio that determines whether the merchant faces per-dispute penalties, reserve
requirements, or loss of processing entirely.

The subtle and important part: the VAMP ratio counts disputes FILED, not disputes
LOST. A merchant who contests and WINS still carries that dispute in their ratio.
Only pre-dispute deflection (CE3.0 via Order Insight, RDR) keeps an event off the
ratio.

This means:
  - Contesting a dispute protects the transaction value but NOT the VAMP ratio.
  - Preventing a dispute protects both.
  - Therefore the prevention layer (src/prevention.py) is not a nice-to-have;
    it is the only lever that addresses portfolio risk.

All threshold values are from Visa's published VAMP program rules (see README).
The merchant portfolio numbers are PLACEHOLDER simulation values, disclosed as such.
"""
from dataclasses import dataclass

from src import config


@dataclass
class VAMPStatus:
    current_ratio: float
    threshold: float
    headroom_events: int          # how many more disputes before crossing
    is_excessive: bool
    is_monitored: bool            # above the 1,500 event/month floor
    monthly_settled_txns: int
    monthly_dispute_events: int
    distance_to_threshold_pct: float


def compute_vamp_status(
    monthly_settled_txns: int = config.MERCHANT_MONTHLY_SETTLED_TXNS,
    monthly_dispute_events: int = config.MERCHANT_MONTHLY_DISPUTE_EVENTS,
    threshold: float = config.VAMP_EXCESSIVE_THRESHOLD,
) -> VAMPStatus:
    """Compute the merchant's current VAMP standing."""
    ratio = monthly_dispute_events / monthly_settled_txns if monthly_settled_txns else 0.0
    max_allowed_events = int(monthly_settled_txns * threshold)
    headroom = max_allowed_events - monthly_dispute_events

    return VAMPStatus(
        current_ratio=round(ratio, 6),
        threshold=threshold,
        headroom_events=headroom,
        is_excessive=ratio >= threshold,
        is_monitored=monthly_dispute_events >= config.VAMP_MONITORING_FLOOR_EVENTS,
        monthly_settled_txns=monthly_settled_txns,
        monthly_dispute_events=monthly_dispute_events,
        distance_to_threshold_pct=round((threshold - ratio) * 100, 4),
    )


def marginal_vamp_cost(status: VAMPStatus = None) -> float:
    """
    The expected ₹ cost of ONE additional dispute event entering the VAMP ratio.

    Model (deliberately simple and explainable, not a black box):

    - If the merchant is ALREADY Excessive: each additional dispute costs the
      flat per-dispute penalty directly.

    - If the merchant is BELOW the threshold: the marginal cost is the penalty
      amount weighted by the probability that this dispute is the one that
      pushes the portfolio over. We approximate that probability as
      1 / headroom_events — i.e. with 100 events of headroom, each new dispute
      carries 1/100th of the crossing risk. When headroom is large the marginal
      cost is negligible; as headroom shrinks it rises sharply, which is the
      behaviour we want.

    - Crossing the threshold does not just cost one penalty — it applies the
      penalty to the merchant's ENTIRE monthly dispute volume, plus triggers
      reserves. We capture the direct fee exposure here and note the reserve /
      termination risk qualitatively rather than inventing a number for it.
    """
    if status is None:
        status = compute_vamp_status()

    if not status.is_monitored:
        return 0.0  # below the monitoring floor, VAMP does not apply

    if status.is_excessive:
        return config.VAMP_PENALTY_PER_DISPUTE_INR

    if status.headroom_events <= 0:
        return config.VAMP_PENALTY_PER_DISPUTE_INR

    # Probability this dispute triggers crossing, times the portfolio-wide
    # penalty exposure if crossing occurs.
    p_crossing = 1.0 / status.headroom_events
    portfolio_exposure = config.VAMP_PENALTY_PER_DISPUTE_INR * status.monthly_dispute_events
    return round(p_crossing * portfolio_exposure, 2)


def vamp_cost_breakdown(status: VAMPStatus = None) -> dict:
    """Explainable breakdown for the UI — no black box."""
    if status is None:
        status = compute_vamp_status()
    marginal = marginal_vamp_cost(status)
    return {
        "current_vamp_ratio_pct": round(status.current_ratio * 100, 4),
        "threshold_pct": round(status.threshold * 100, 2),
        "headroom_events": status.headroom_events,
        "is_excessive": status.is_excessive,
        "is_monitored": status.is_monitored,
        "marginal_vamp_cost_per_dispute_inr": marginal,
        "penalty_per_dispute_inr": config.VAMP_PENALTY_PER_DISPUTE_INR,
        "note": (
            "VAMP ratio counts disputes FILED, not disputes LOST. Contesting does "
            "not reduce this cost — only pre-dispute deflection does."
        ),
    }
