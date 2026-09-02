"""
Unit tests for the EV Decision Engine (src/ev_engine.py) — the component
the brief calls out as the core differentiator, so it's the one place we
test as if this were headed to production, not just a demo script.
"""
import pytest

from src import config, ev_engine
from src.ev_engine import Action


def test_ev_contest_formula_is_correct():
    # EV(contest) = amount * (2p - 1) - contest_cost, worked by hand.
    win_prob, amount, cost = 0.8, 10_000.0, 500.0
    expected = 10_000.0 * (2 * 0.8 - 1) - 500.0  # = 5500.0
    assert ev_engine.ev_contest(win_prob, amount, cost) == pytest.approx(expected)
    assert ev_engine.ev_contest(win_prob, amount, cost) == pytest.approx(5500.0)


def test_ev_accept_is_just_negative_amount():
    assert ev_engine.ev_accept(12_345.0) == -12_345.0


def test_hard_ceiling_blocks_regardless_of_high_confidence():
    # Even at 99% win probability, an amount above the hard ceiling must
    # escalate — the ceiling is a fixed rule, not a learned/overridable one.
    decision = ev_engine.decide(
        win_prob=0.99,
        amount=config.HARD_CEILING_INR + 1,
        evidence_completeness=1.0,
    )
    assert decision.action == Action.ESCALATE
    assert decision.ceiling_blocked is True


def test_low_evidence_completeness_blocks_auto_contest_even_if_ev_favors_it():
    # High win_prob and low amount would normally auto-contest, but
    # incomplete evidence must force escalation regardless of EV math.
    decision = ev_engine.decide(
        win_prob=0.9,
        amount=1_000.0,
        evidence_completeness=config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO - 0.01,
    )
    assert decision.action == Action.ESCALATE
    assert decision.evidence_blocked is True


def test_favorable_ev_within_ceiling_and_evidence_auto_contests():
    decision = ev_engine.decide(win_prob=0.85, amount=2_000.0, evidence_completeness=1.0)
    assert decision.action == Action.AUTO_CONTEST
    assert decision.ev_contest > decision.ev_accept
    assert decision.ceiling_blocked is False
    assert decision.evidence_blocked is False


def test_same_win_probability_can_flip_decision_purely_on_amount():
    # This is demo case #3 from the brief: a fixed-threshold system would
    # treat these identically (same win_prob), but EV math — which scales
    # with amount while contest_cost stays fixed — can and should diverge.
    low_amount_decision = ev_engine.decide(
        win_prob=0.58, amount=1_000.0, evidence_completeness=1.0
    )
    high_amount_decision = ev_engine.decide(
        win_prob=0.58, amount=config.HARD_CEILING_INR + 5_000, evidence_completeness=1.0
    )
    assert low_amount_decision.action == Action.AUTO_CONTEST
    assert high_amount_decision.action == Action.ESCALATE
    assert high_amount_decision.ceiling_blocked is True


def test_decide_rejects_invalid_win_probability():
    with pytest.raises(ValueError):
        ev_engine.decide(win_prob=1.5, amount=1_000.0, evidence_completeness=1.0)
    with pytest.raises(ValueError):
        ev_engine.decide(win_prob=-0.1, amount=1_000.0, evidence_completeness=1.0)


def test_decide_rejects_negative_amount():
    with pytest.raises(ValueError):
        ev_engine.decide(win_prob=0.5, amount=-1.0, evidence_completeness=1.0)
