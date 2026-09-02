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

from src import config


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
