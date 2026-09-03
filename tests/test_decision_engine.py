"""
Tests for the four-way decision engine and RDR optimizer.

The guardrail tests here are written to assert the SAFETY PROPERTY, not a proxy
for it. The distinction matters: an earlier design of this engine scored ESCALATE
against the automated actions on expected value, which meant an over-ceiling
dispute was routed to DEFLECT — an automated action — instead of to a human. A
test asserting only `chosen != CONTEST` passes happily against that bug, because
the action genuinely isn't CONTEST; it's a *different* automated action. So the
tests below assert `chosen == ESCALATE`, which is the property the hard ceiling
actually promises.
"""
import pandas as pd
import pytest

from src import config, decision_engine, rdr_optimizer, vamp
from src.decision_engine import AUTOMATED_ACTIONS, DecisionAction


def _opt(decision, action):
    return next(o for o in decision.options if o.action == action)


# --- shape ------------------------------------------------------------------


def test_all_four_actions_are_priced():
    d = decision_engine.decide_four_way(
        win_prob=0.8, amount=3_000.0, evidence_completeness=1.0)
    assert {o.action for o in d.options} == {
        DecisionAction.DEFLECT, DecisionAction.AUTO_RESOLVE,
        DecisionAction.CONTEST, DecisionAction.ESCALATE,
    }


def test_escalate_is_never_ev_comparable():
    """ESCALATE carries a context figure, not an EV that competes."""
    d = decision_engine.decide_four_way(
        win_prob=0.8, amount=3_000.0, evidence_completeness=1.0)
    assert _opt(d, DecisionAction.ESCALATE).ev_comparable is False
    for action in AUTOMATED_ACTIONS:
        assert _opt(d, action).ev_comparable is True


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        decision_engine.decide_four_way(win_prob=1.5, amount=1_000.0, evidence_completeness=1.0)
    with pytest.raises(ValueError):
        decision_engine.decide_four_way(win_prob=0.5, amount=-1.0, evidence_completeness=1.0)


# --- the hard ceiling: the property, not a proxy ----------------------------


def test_over_ceiling_escalates_to_a_human():
    """The ceiling promises a PERSON, not merely 'not CONTEST'."""
    d = decision_engine.decide_four_way(
        win_prob=0.99,
        amount=config.HARD_CEILING_INR + 1,
        evidence_completeness=1.0,
    )
    assert d.chosen == DecisionAction.ESCALATE
    assert d.ceiling_blocked is True


def test_over_ceiling_blocks_every_automated_action():
    d = decision_engine.decide_four_way(
        win_prob=0.99, amount=config.HARD_CEILING_INR + 1, evidence_completeness=1.0)
    for action in AUTOMATED_ACTIONS:
        assert _opt(d, action).viable is False, f"{action} should be ceiling-blocked"


@pytest.mark.parametrize("amount", [25_001.0, 42_000.0, 150_000.0])
@pytest.mark.parametrize("win_prob", [0.05, 0.5, 0.99])
@pytest.mark.parametrize("completeness", [0.0, 0.6, 1.0])
def test_no_over_ceiling_case_is_ever_automated(amount, win_prob, completeness):
    """Sweep: nothing above the ceiling may be auto-actioned, at any win
    probability, any evidence level, either reason-code class."""
    for reason in (config.REASON_CODE, "10.4"):
        for order_data in (True, False):
            d = decision_engine.decide_four_way(
                win_prob=win_prob, amount=amount, evidence_completeness=completeness,
                reason_code=reason, has_order_data=order_data,
            )
            assert d.chosen == DecisionAction.ESCALATE


def test_demo_case_three_still_escalates():
    """Regression guard on the case the whole submission is built around:
    64% win probability, ₹42,000, evidence 67%. EV alone favours contesting;
    the ceiling must still send it to a human."""
    d = decision_engine.decide_four_way(
        win_prob=0.64, amount=42_000.0, evidence_completeness=0.67)
    assert d.chosen == DecisionAction.ESCALATE
    # And the EV genuinely did favour contesting — otherwise the case proves nothing.
    assert _opt(d, DecisionAction.CONTEST).expected_value_inr > -42_000.0


def test_just_under_ceiling_may_be_automated():
    """The ceiling is a boundary, not a blanket ban."""
    d = decision_engine.decide_four_way(
        win_prob=0.9, amount=config.HARD_CEILING_INR - 1_000, evidence_completeness=1.0)
    assert d.chosen in AUTOMATED_ACTIONS
    assert d.ceiling_blocked is False


# --- the evidence gate ------------------------------------------------------


def test_low_evidence_blocks_contest_only():
    """Evidence completeness is about the strength of the case you'd argue, so it
    gates CONTEST — but NOT refunding, which is the conservative action."""
    d = decision_engine.decide_four_way(
        win_prob=0.9, amount=2_000.0,
        evidence_completeness=config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO - 0.01,
    )
    assert _opt(d, DecisionAction.CONTEST).viable is False
    assert _opt(d, DecisionAction.AUTO_RESOLVE).viable is True
    assert d.evidence_blocked is True
    assert d.chosen != DecisionAction.CONTEST


# --- the VAMP asymmetry -----------------------------------------------------


def test_non_fraud_rdr_has_zero_vamp_cost():
    """13.1 is non-fraud, so an RDR resolution keeps it off the ratio."""
    d = decision_engine.decide_four_way(
        win_prob=0.7, amount=2_000.0, evidence_completeness=1.0, reason_code="13.1")
    assert _opt(d, DecisionAction.AUTO_RESOLVE).vamp_cost_inr == 0.0


def test_fraud_code_rdr_still_carries_vamp_cost():
    """10.4 is a fraud code — RDR does NOT exclude it from VAMP."""
    d = decision_engine.decide_four_way(
        win_prob=0.7, amount=2_000.0, evidence_completeness=1.0, reason_code="10.4")
    assert _opt(d, DecisionAction.AUTO_RESOLVE).vamp_cost_inr > 0.0


def test_rdr_is_a_weaker_lever_for_fraud_codes():
    """The asymmetry must actually move the EV, not just a label."""
    non_fraud = decision_engine.decide_four_way(
        win_prob=0.7, amount=2_000.0, evidence_completeness=1.0, reason_code="13.1")
    fraud = decision_engine.decide_four_way(
        win_prob=0.7, amount=2_000.0, evidence_completeness=1.0, reason_code="10.4")
    assert (_opt(non_fraud, DecisionAction.AUTO_RESOLVE).expected_value_inr
            > _opt(fraud, DecisionAction.AUTO_RESOLVE).expected_value_inr)


def test_deflect_not_viable_without_order_data():
    d = decision_engine.decide_four_way(
        win_prob=0.7, amount=2_000.0, evidence_completeness=1.0, has_order_data=False)
    assert _opt(d, DecisionAction.DEFLECT).viable is False


def test_chosen_action_is_always_viable():
    for amount in (500.0, 5_000.0, 24_000.0, 50_000.0):
        for wp in (0.1, 0.5, 0.95):
            d = decision_engine.decide_four_way(
                win_prob=wp, amount=amount, evidence_completeness=1.0)
            assert _opt(d, d.chosen).viable is True


def test_decision_carries_its_reasoning():
    d = decision_engine.decide_four_way(
        win_prob=0.5, amount=config.HARD_CEILING_INR + 1, evidence_completeness=1.0)
    assert d.reasons
    assert any("ceiling" in r for r in d.reasons)


# --- RDR optimizer ----------------------------------------------------------


def _fake_scored(n=50):
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(n)],
        "amount_inr": [500.0 + i * 100 for i in range(n)],
        "actual_won": [1 if i % 3 else 0 for i in range(n)],
        "win_probability": [0.7] * n,
        "system_action": ["AUTO_CONTEST" if i % 2 else "ESCALATE" for i in range(n)],
        "ev_contest": [100.0] * n,
        "ev_accept": [-500.0] * n,
        "evidence_completeness": [1.0] * n,
    })


def test_optimizer_returns_a_curve():
    curve = rdr_optimizer.optimize_rdr_threshold(_fake_scored(), n_candidates=10)
    assert len(curve) == 10
    assert "total_cost_inr" in curve.columns


def test_optimizer_never_recommends_above_the_hard_ceiling():
    """RDR is an automated money movement, so the sweep is bound by the same
    ceiling as every other automated action."""
    big = _fake_scored()
    big["amount_inr"] = big["amount_inr"] * 100  # push amounts far past the ceiling
    curve = rdr_optimizer.optimize_rdr_threshold(big, n_candidates=15)
    assert curve["amount_threshold_inr"].max() <= config.HARD_CEILING_INR


def test_higher_threshold_resolves_more_disputes():
    df = _fake_scored()
    low = rdr_optimizer.evaluate_rdr_threshold(df, 1_000.0)
    high = rdr_optimizer.evaluate_rdr_threshold(df, 4_000.0)
    assert high["disputes_auto_resolved"] > low["disputes_auto_resolved"]


def test_best_threshold_minimises_total_cost():
    curve = rdr_optimizer.optimize_rdr_threshold(_fake_scored(), n_candidates=15)
    best = rdr_optimizer.best_threshold(curve)
    assert best["total_cost_inr"] == curve["total_cost_inr"].min()


def test_non_fraud_rdr_buys_back_vamp_headroom():
    df = _fake_scored()
    res = rdr_optimizer.evaluate_rdr_threshold(df, 3_000.0, reason_code="13.1")
    assert res["vamp_headroom_after"] >= res["vamp_headroom_before"]


def test_fraud_code_rdr_buys_back_no_headroom():
    """The asymmetry, at portfolio level: refunding a fraud dispute does not
    remove it from the ratio."""
    df = _fake_scored()
    res = rdr_optimizer.evaluate_rdr_threshold(df, 3_000.0, reason_code="10.4")
    assert res["monthly_events_removed_from_ratio"] == 0
    assert res["vamp_headroom_after"] == res["vamp_headroom_before"]


def test_empty_input_returns_empty_curve():
    empty = _fake_scored(0)
    assert rdr_optimizer.optimize_rdr_threshold(empty).empty
    assert rdr_optimizer.best_threshold(pd.DataFrame()) == {}
