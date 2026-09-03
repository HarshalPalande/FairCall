"""
EV Decision Engine — the core differentiator.

Not a confidence threshold. Computes the actual expected ₹ value of
contesting vs. accepting, and only auto-acts when:
  1. EV(contest) > EV(accept), AND
  2. amount <= HARD_CEILING_INR (a fixed rule, never overridden by
     confidence — see config.py), AND
  3. evidence completeness >= MIN_EVIDENCE_COMPLETENESS_FOR_AUTO.

EV math:
    EV(accept)  = -amount
        (merchant eats the full loss, no contest fees incurred)

    EV(contest) = P(win) * (amount - contest_cost)
                + (1 - P(win)) * (-amount - contest_cost)
                = amount * (2*P(win) - 1) - contest_cost

    where contest_cost = CONTEST_LABOR_COST_INR + CHARGEBACK_ADMIN_FEE_INR
    is charged regardless of outcome (network/acquirer admin fee + labor
    to prepare the packet).

This is why a naive fixed-probability-threshold system and this system
disagree on borderline, high-amount cases: EV(contest) scales with amount,
so the SAME win probability can be worth contesting at ₹1,000 and worth
escalating-for-human-judgment at ₹80,000, purely because the downside of
being wrong scales too. That divergence is demo case #3 in the brief.
"""
from dataclasses import dataclass
from enum import Enum

from src import config, vamp


class Action(str, Enum):
    AUTO_CONTEST = "AUTO_CONTEST"
    ESCALATE = "ESCALATE"


@dataclass
class EVDecision:
    action: Action
    ev_contest: float
    ev_accept: float
    win_prob: float
    amount: float
    contest_cost: float
    ceiling_blocked: bool
    evidence_blocked: bool
    reasons: list


def ev_contest(win_prob: float, amount: float, contest_cost: float = config.CONTEST_COST_INR) -> float:
    return amount * (2 * win_prob - 1) - contest_cost


def ev_accept(amount: float) -> float:
    return -amount


def hard_ceiling_check(amount: float, ceiling: float = config.HARD_CEILING_INR) -> bool:
    """Returns True if the amount is BLOCKED from auto-action (i.e. exceeds ceiling)."""
    return amount > ceiling


def decide(
    win_prob: float,
    amount: float,
    evidence_completeness: float,
    contest_cost: float = config.CONTEST_COST_INR,
    ceiling: float = config.HARD_CEILING_INR,
    min_completeness: float = config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO,
) -> EVDecision:
    if not (0.0 <= win_prob <= 1.0):
        raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}")

    evc = ev_contest(win_prob, amount, contest_cost)
    eva = ev_accept(amount)

    ceiling_blocked = hard_ceiling_check(amount, ceiling)
    evidence_blocked = evidence_completeness < min_completeness

    reasons = []
    if evc <= eva:
        reasons.append(f"EV(contest)={evc:,.2f} does not exceed EV(accept)={eva:,.2f}")
    if ceiling_blocked:
        reasons.append(f"amount ₹{amount:,.2f} exceeds hard ceiling ₹{ceiling:,.2f}")
    if evidence_blocked:
        reasons.append(
            f"evidence completeness {evidence_completeness:.2f} below minimum {min_completeness:.2f} for auto-action"
        )

    if evc > eva and not ceiling_blocked and not evidence_blocked:
        action = Action.AUTO_CONTEST
        reasons = [f"EV(contest)={evc:,.2f} > EV(accept)={eva:,.2f}; within ceiling; evidence sufficient"]
    else:
        action = Action.ESCALATE

    return EVDecision(
        action=action,
        ev_contest=round(evc, 2),
        ev_accept=round(eva, 2),
        win_prob=win_prob,
        amount=amount,
        contest_cost=contest_cost,
        ceiling_blocked=ceiling_blocked,
        evidence_blocked=evidence_blocked,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# VAMP-aware EV — see src/vamp.py and README "VAMP Portfolio Risk"
# ---------------------------------------------------------------------------
#
# The functions above price a dispute purely at its transaction amount. Under
# VAMP (Visa Acquirer Monitoring Program), every FILED dispute also increments
# a portfolio-level ratio that can trigger per-dispute penalties, reserve
# requirements, or processing termination — regardless of whether the dispute
# is won or lost. The functions below make that cost visible without changing
# `decide()` or `EVDecision`, so every existing caller and test is unaffected.


def ev_contest_vamp_aware(win_prob, amount, contest_cost=config.CONTEST_COST_INR, vamp_cost=None):
    """
    EV(contest) including VAMP portfolio cost.

    The dispute has already been FILED at this point, so it already counts in
    the VAMP ratio regardless of whether we contest or accept. That means
    vamp_cost appears in BOTH branches and does not change the contest-vs-accept
    comparison.

    This is exactly the point: contesting cannot undo VAMP exposure. The only
    decision that avoids it is PREVENTION, which happens before a dispute exists.
    We surface it here so the number is visible in the decision, not hidden.
    """
    if vamp_cost is None:
        vamp_cost = vamp.marginal_vamp_cost()
    return amount * (2 * win_prob - 1) - contest_cost - vamp_cost


def ev_accept_vamp_aware(amount, vamp_cost=None):
    """EV(accept) including VAMP cost — see note in ev_contest_vamp_aware."""
    if vamp_cost is None:
        vamp_cost = vamp.marginal_vamp_cost()
    return -amount - vamp_cost


def ev_prevented(amount, prevention_cost=0.0):
    """
    EV of a dispute that never happened because evidence was collected proactively.

    This is the branch the other two cannot reach: no transaction loss, no
    contest cost, and critically NO VAMP RATIO IMPACT, because the dispute was
    deflected pre-emptively rather than fought after filing.

    prevention_cost is the cost of proactively gathering evidence (courier
    confirmation requests, etc.) — small relative to a dispute.
    """
    return -prevention_cost


@dataclass
class VAMPAwareDecision:
    action: Action
    ev_contest: float
    ev_accept: float
    ev_contest_vamp_aware: float
    ev_accept_vamp_aware: float
    vamp_cost: float
    win_prob: float
    amount: float
    reasons: list
    vamp_note: str


def decide_vamp_aware(
    win_prob,
    amount,
    evidence_completeness,
    contest_cost=config.CONTEST_COST_INR,
    ceiling=config.HARD_CEILING_INR,
    min_completeness=config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO,
    vamp_cost=None,
):
    """
    Same decision logic as decide(), but reports VAMP-aware EV figures alongside
    the plain ones so the analyst can see the portfolio cost explicitly.

    The ACTION is unchanged by VAMP (because VAMP cost applies to both branches
    once a dispute is filed) — that is the honest finding, and we say so rather
    than pretending VAMP changes the contest/accept call.
    """
    if vamp_cost is None:
        vamp_cost = vamp.marginal_vamp_cost()

    base = decide(win_prob, amount, evidence_completeness, contest_cost, ceiling, min_completeness)

    return VAMPAwareDecision(
        action=base.action,
        ev_contest=base.ev_contest,
        ev_accept=base.ev_accept,
        ev_contest_vamp_aware=round(ev_contest_vamp_aware(win_prob, amount, contest_cost, vamp_cost), 2),
        ev_accept_vamp_aware=round(ev_accept_vamp_aware(amount, vamp_cost), 2),
        vamp_cost=round(vamp_cost, 2),
        win_prob=win_prob,
        amount=amount,
        reasons=base.reasons,
        vamp_note=(
            f"This dispute adds ₹{vamp_cost:,.2f} of VAMP portfolio risk regardless "
            f"of the contest outcome. Contesting cannot recover it — only preventing "
            f"the dispute would have."
        ),
    )
