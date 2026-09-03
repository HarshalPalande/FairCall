"""Tests for VAMP portfolio risk modelling (src/vamp.py)."""
import pytest

from src import config, vamp


def test_ratio_computed_correctly():
    status = vamp.compute_vamp_status(monthly_settled_txns=100_000, monthly_dispute_events=1_500)
    assert status.current_ratio == pytest.approx(0.015)


def test_excessive_flagged_at_threshold():
    status = vamp.compute_vamp_status(
        monthly_settled_txns=100_000, monthly_dispute_events=1_500, threshold=0.015
    )
    assert status.is_excessive is True


def test_below_threshold_not_excessive():
    status = vamp.compute_vamp_status(
        monthly_settled_txns=200_000, monthly_dispute_events=2_000, threshold=0.015
    )
    assert status.is_excessive is False
    assert status.headroom_events == 1_000  # 200000*0.015 = 3000, minus 2000


def test_below_monitoring_floor_has_zero_vamp_cost():
    """Merchants under 1,500 combined events/month aren't formally monitored."""
    status = vamp.compute_vamp_status(monthly_settled_txns=100_000, monthly_dispute_events=500)
    assert status.is_monitored is False
    assert vamp.marginal_vamp_cost(status) == 0.0


def test_marginal_cost_rises_as_headroom_shrinks():
    """The core behaviour: less headroom = higher marginal cost per dispute."""
    lots_of_headroom = vamp.compute_vamp_status(monthly_settled_txns=200_000, monthly_dispute_events=1_600)
    little_headroom = vamp.compute_vamp_status(monthly_settled_txns=200_000, monthly_dispute_events=2_900)

    cost_lots = vamp.marginal_vamp_cost(lots_of_headroom)
    cost_little = vamp.marginal_vamp_cost(little_headroom)

    assert cost_little > cost_lots


def test_excessive_merchant_pays_flat_penalty():
    status = vamp.compute_vamp_status(
        monthly_settled_txns=100_000, monthly_dispute_events=2_000, threshold=0.015
    )
    assert status.is_excessive is True
    assert vamp.marginal_vamp_cost(status) == config.VAMP_PENALTY_PER_DISPUTE_INR
