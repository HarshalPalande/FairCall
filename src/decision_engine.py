"""
Four-way decision engine.

The original EV engine (src/ev_engine.py) chooses between AUTO_CONTEST and
ESCALATE. That models the decision as "fight or don't fight," which is an
incomplete picture of the levers a merchant actually has. Visa's pre-dispute
infrastructure gives four meaningfully different actions, and they differ in
BOTH cost and VAMP consequence:

    DEFLECT      Order Insight — push order detail at cardholder inquiry. If it
                 works, no dispute is ever filed and nothing touches the VAMP
                 ratio. Only viable when rich order data exists.

    AUTO_RESOLVE RDR — refund the cardholder pre-dispute per merchant rules.
                 Loses the transaction value, but for NON-FRAUD reason codes the
                 event is excluded from the VAMP ratio. For fraud codes (10.4) it
                 is NOT excluded, which makes RDR a much weaker lever there.

    CONTEST      Representment. Recovers the transaction value if won, but the
                 dispute was already filed and already counts toward VAMP.

    ESCALATE     Human analyst review, with full context attached.

The interesting consequence: for a non-fraud dispute, RDR-refunding a case you
would probably WIN can still be the cheaper decision once VAMP exposure is priced
in. A two-output engine cannot express that. This one can.

--- WHY ESCALATE IS NOT SCORED AGAINST THE OTHERS -------------------------------

ESCALATE is deliberately NOT given an expected value and NOT entered into the
max() comparison. Two reasons, and the second one is a safety property:

1. It isn't comparable. "A human decides next" has no ₹ value we can compute
   without modelling the analyst's own decision, and inventing one (e.g. pricing
   it as the accept-loss) would be a made-up number driving a real routing choice.

2. Scoring it breaks the hard ceiling. If ESCALATE competes on EV, it loses
   almost always — DEFLECT's expected value is a convex combination of "cheap
   success" and "the accept loss", so it dominates a flat accept-loss whenever
   deflection has any chance of working. An over-ceiling dispute would then be
   routed to DEFLECT — an automated action — instead of to a person, and the
   ceiling guarantee would be silently gone while still "passing" a test that
   only asserts the action isn't CONTEST.

So the structure here is: **ESCALATE is the fallback, not a competitor.** We
choose the best VIABLE AUTOMATED action; if no automated action is permitted,
the case goes to a human. That makes the guardrail unconditional by construction
rather than by arithmetic.

--- THE GUARDRAILS --------------------------------------------------------------

    Hard ₹ ceiling      Gates EVERY automated action (DEFLECT, AUTO_RESOLVE and
                        CONTEST alike). Its purpose is "no single large automated
                        decision without a human" — that is a property of the
                        AMOUNT, not of which lever you happen to pull, and
                        auto-refunding ₹150,000 is exactly as much a large
                        automated money movement as auto-contesting it.

    Evidence minimum    Gates CONTEST only, exactly as in the original engine.
                        It is a statement about the strength of the case you would
                        argue, not about the amount at risk. Notably it does NOT
                        gate AUTO_RESOLVE: refunding is the conservative action,
                        and thin evidence is a reason to refund rather than fight,
                        which is precisely the lever the two-output engine lacked.

Neither gate is overridable by any EV computation.
"""
from dataclasses import dataclass, field
from enum import Enum

from src import config, vamp


class DecisionAction(str, Enum):
    DEFLECT = "DEFLECT"
    AUTO_RESOLVE = "AUTO_RESOLVE"
    CONTEST = "CONTEST"
    ESCALATE = "ESCALATE"


#: The actions the system may take without a human in the loop.
AUTOMATED_ACTIONS = (
    DecisionAction.DEFLECT,
    DecisionAction.AUTO_RESOLVE,
    DecisionAction.CONTEST,
)


@dataclass
class ActionOption:
    action: DecisionAction
    expected_value_inr: float
    vamp_cost_inr: float
    viable: bool
    rationale: str
    blocked_reason: str = None
    #: ESCALATE carries no comparable EV — see the module docstring.
    ev_comparable: bool = True


@dataclass
class FourWayDecision:
    chosen: DecisionAction
    options: list = field(default_factory=list)
    win_prob: float = 0.0
    amount: float = 0.0
    reason_code: str = ""
    vamp_cost_inr: float = 0.0
    vamp_headroom: int = 0
    ceiling_blocked: bool = False
    evidence_blocked: bool = False
    reasons: list = field(default_factory=list)


def rdr_excluded_from_vamp(reason_code: str) -> bool:
    """Whether an RDR resolution keeps this reason code off the VAMP ratio.
    Non-fraud codes: yes. Fraud code 10.4: no."""
    return reason_code in config.RDR_VAMP_EXCLUDED_REASON_CODES


def evaluate_options(
    win_prob: float,
    amount: float,
    evidence_completeness: float,
    reason_code: str = config.REASON_CODE,
    has_order_data: bool = True,
    contest_cost: float = config.CONTEST_COST_INR,
    ceiling: float = config.HARD_CEILING_INR,
    min_completeness: float = config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO,
    vamp_status=None,
) -> list:
    """Price every available action. Returns a list of ActionOption.

    The ceiling gates all three automated actions; the evidence minimum gates
    CONTEST only. See the module docstring for why they differ.
    """
    if not (0.0 <= win_prob <= 1.0):
        raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}")

    if vamp_status is None:
        vamp_status = vamp.compute_vamp_status()
    marginal_vamp = vamp.marginal_vamp_cost(vamp_status)

    ceiling_blocked = amount > ceiling
    ceiling_msg = (
        f"amount ₹{amount:,.2f} exceeds the hard ceiling ₹{ceiling:,.2f} — no "
        f"automated action is permitted at this value, regardless of EV."
    )

    options = []

    # --- DEFLECT (Order Insight) ---------------------------------------
    p_deflect = config.ORDER_INSIGHT_BASE_DEFLECT_RATE * evidence_completeness
    # Success: we pay only the Order Insight cost and no dispute is ever filed.
    # Failure: the inquiry becomes a dispute, so we carry the accept-loss and the
    # VAMP cost on top of the attempt cost.
    ev_deflect = (
        p_deflect * (-config.ORDER_INSIGHT_COST_INR)
        + (1 - p_deflect) * (-amount - marginal_vamp - config.ORDER_INSIGHT_COST_INR)
    )
    deflect_blocked = None
    if ceiling_blocked:
        deflect_blocked = ceiling_msg
    elif not has_order_data:
        deflect_blocked = "Order Insight requires rich order data to push to the issuer."
    options.append(ActionOption(
        action=DecisionAction.DEFLECT,
        expected_value_inr=round(ev_deflect, 2),
        vamp_cost_inr=round((1 - p_deflect) * marginal_vamp, 2),
        viable=deflect_blocked is None,
        rationale=(
            f"Order Insight deflection at {p_deflect:.0%} estimated success (base rate "
            f"scaled by evidence completeness). Deflected inquiries never become "
            f"disputes and never touch the VAMP ratio."
        ),
        blocked_reason=deflect_blocked,
    ))

    # --- AUTO_RESOLVE (RDR) --------------------------------------------
    vamp_excluded = rdr_excluded_from_vamp(reason_code)
    rdr_vamp_cost = 0.0 if vamp_excluded else marginal_vamp
    ev_rdr = -amount - config.RDR_RESOLUTION_FEE_INR - rdr_vamp_cost
    options.append(ActionOption(
        action=DecisionAction.AUTO_RESOLVE,
        expected_value_inr=round(ev_rdr, 2),
        vamp_cost_inr=round(rdr_vamp_cost, 2),
        viable=not ceiling_blocked,
        rationale=(
            f"RDR refunds the cardholder pre-dispute. Reason code {reason_code} is "
            + ("NON-FRAUD, so the event is excluded from the VAMP ratio."
               if vamp_excluded else
               "a FRAUD code, so it still counts toward VAMP even when refunded — "
               "RDR is a much weaker lever here.")
        ),
        blocked_reason=ceiling_msg if ceiling_blocked else None,
    ))

    # --- CONTEST (representment) ---------------------------------------
    ev_contest = amount * (2 * win_prob - 1) - contest_cost - marginal_vamp
    evidence_blocked = evidence_completeness < min_completeness
    contest_blocked = None
    if ceiling_blocked:
        contest_blocked = ceiling_msg
    elif evidence_blocked:
        contest_blocked = (
            f"evidence completeness {evidence_completeness:.0%} is below the "
            f"{min_completeness:.0%} minimum for auto-action."
        )
    options.append(ActionOption(
        action=DecisionAction.CONTEST,
        expected_value_inr=round(ev_contest, 2),
        vamp_cost_inr=round(marginal_vamp, 2),
        viable=contest_blocked is None,
        rationale=(
            f"Representment at {win_prob:.0%} win probability. Recovers the transaction "
            f"if won — but the dispute was already filed, so the VAMP cost applies "
            f"either way and cannot be recovered."
        ),
        blocked_reason=contest_blocked,
    ))

    # --- ESCALATE ------------------------------------------------------
    # Always available, never scored. The figure below is the merchant's exposure
    # if the analyst simply accepts; it is context for the human, NOT an EV that
    # competes with the automated options. See the module docstring.
    options.append(ActionOption(
        action=DecisionAction.ESCALATE,
        expected_value_inr=round(-amount - marginal_vamp, 2),
        vamp_cost_inr=round(marginal_vamp, 2),
        viable=True,
        rationale=(
            "Route to a human analyst with score, EV comparison, evidence gaps and "
            "VAMP context attached. Not EV-scored: the value depends on a decision "
            "a person has not made yet."
        ),
        ev_comparable=False,
    ))

    return options


def decide_four_way(
    win_prob: float,
    amount: float,
    evidence_completeness: float,
    reason_code: str = config.REASON_CODE,
    has_order_data: bool = True,
    contest_cost: float = config.CONTEST_COST_INR,
    ceiling: float = config.HARD_CEILING_INR,
    min_completeness: float = config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO,
    vamp_status=None,
) -> FourWayDecision:
    """Choose the highest-EV viable AUTOMATED action, or escalate to a human.

    Guardrails are applied first and are absolute. If the amount is over the hard
    ceiling, no automated action is viable and the case escalates — regardless of
    how favourable any EV looks. ESCALATE is the fallback, never an EV competitor;
    that is what makes the ceiling guarantee structural rather than arithmetic.
    """
    if vamp_status is None:
        vamp_status = vamp.compute_vamp_status()

    options = evaluate_options(
        win_prob, amount, evidence_completeness, reason_code, has_order_data,
        contest_cost, ceiling, min_completeness, vamp_status,
    )

    ceiling_blocked = amount > ceiling
    evidence_blocked = evidence_completeness < min_completeness

    viable_automated = [
        o for o in options if o.action in AUTOMATED_ACTIONS and o.viable
    ]

    if viable_automated:
        best = max(viable_automated, key=lambda o: o.expected_value_inr)
        chosen = best.action
        reasons = [f"Chose {chosen.value} with EV ₹{best.expected_value_inr:,.2f}."]
    else:
        chosen = DecisionAction.ESCALATE
        reasons = ["No automated action is permitted — escalating to a human analyst."]

    for o in options:
        if not o.viable and o.blocked_reason:
            reasons.append(f"{o.action.value} unavailable: {o.blocked_reason}")

    # Surface the interesting case explicitly: RDR beating a winnable contest.
    contest_opt = next(o for o in options if o.action == DecisionAction.CONTEST)
    if (chosen == DecisionAction.AUTO_RESOLVE
            and contest_opt.viable
            and win_prob > 0.5):
        reasons.append(
            f"Note: representment would probably WIN ({win_prob:.0%}), but RDR still "
            f"scores higher once VAMP exposure is priced in — refunding keeps this "
            f"event off the ratio entirely."
        )

    return FourWayDecision(
        chosen=chosen,
        options=options,
        win_prob=win_prob,
        amount=amount,
        reason_code=reason_code,
        vamp_cost_inr=round(vamp.marginal_vamp_cost(vamp_status), 2),
        vamp_headroom=vamp_status.headroom_events,
        ceiling_blocked=ceiling_blocked,
        evidence_blocked=evidence_blocked,
        reasons=reasons,
    )
