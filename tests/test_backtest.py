"""Tests for the backtest and false-positive cost analysis (src/backtest.py)."""
import pandas as pd
import pytest

from src import config
from src.backtest import compute_false_positive_costs, run_backtest


def _make_scored_df(actions, outcomes, amounts):
    """Helper to build a minimal scored DataFrame for testing."""
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(len(actions))],
        "amount_inr": amounts,
        "actual_won": outcomes,
        "win_probability": [0.8] * len(actions),
        "system_action": actions,
        "ev_contest": [100.0] * len(actions),
        "ev_accept": [-1000.0] * len(actions),
        "evidence_completeness": [1.0] * len(actions),
    })


def test_all_auto_contest_all_win():
    """If system auto-contests everything and wins everything, no false positives."""
    df = _make_scored_df(
        actions=["AUTO_CONTEST"] * 5,
        outcomes=[1, 1, 1, 1, 1],
        amounts=[1000.0] * 5,
    )
    result = compute_false_positive_costs(df)
    assert result["false_positive_count"] == 0
    assert result["false_positive_cost_inr"] == 0
    assert result["true_positive_count"] == 5


def test_false_positive_cost_includes_contest_cost():
    """A false positive costs amount + contest_cost."""
    df = _make_scored_df(
        actions=["AUTO_CONTEST"],
        outcomes=[0],  # merchant lost
        amounts=[2000.0],
    )
    result = compute_false_positive_costs(df)
    assert result["false_positive_count"] == 1
    expected_cost = 2000.0 + config.CONTEST_COST_INR
    assert result["false_positive_cost_inr"] == pytest.approx(expected_cost)


def test_backtest_accept_all_loses_everything():
    """Accept-all strategy = negative sum of all amounts."""
    df = _make_scored_df(
        actions=["AUTO_CONTEST"] * 3,
        outcomes=[1, 0, 1],
        amounts=[1000.0, 2000.0, 3000.0],
    )
    result = run_backtest(df)
    assert result["strategy_accept_all_inr"] == pytest.approx(-6000.0)


def test_backtest_system_beats_accept_all():
    """With a reasonable win rate, the system should save money vs accepting all."""
    df = _make_scored_df(
        actions=["AUTO_CONTEST"] * 10,
        outcomes=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 70% win rate
        amounts=[1000.0] * 10,
    )
    result = run_backtest(df)
    assert result["savings_vs_accept_all_inr"] > 0


def test_backtest_contest_all_bears_labor_cost_for_every_case():
    """Contest-all requires a human to prepare every packet, so it must bear
    analyst labor cost on every single case, not just the auto-contested ones."""
    df = _make_scored_df(
        actions=["AUTO_CONTEST"] * 4,
        outcomes=[1, 1, 1, 1],
        amounts=[1000.0] * 4,
    )
    result = run_backtest(df)
    expected_contest_all = 4 * (1000.0 - config.CONTEST_COST_INR - config.ANALYST_LABOR_COST_PER_CASE_INR)
    assert result["strategy_contest_all_inr"] == pytest.approx(expected_contest_all)


def test_backtest_auto_contested_cases_bear_no_analyst_labor():
    """AUTO_CONTEST cases are handled by the system automatically — they
    must NOT bear the analyst labor cost the way Contest-All does."""
    df = _make_scored_df(
        actions=["AUTO_CONTEST"] * 4,
        outcomes=[1, 1, 1, 1],
        amounts=[1000.0] * 4,
    )
    result = run_backtest(df)
    expected_system = 4 * (1000.0 - config.CONTEST_COST_INR)
    assert result["strategy_system_inr"] == pytest.approx(expected_system)
    assert result["system_labor_cost_inr"] == 0.0
