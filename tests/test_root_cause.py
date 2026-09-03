"""
Tests for the root-cause intelligence module.

The properties worth protecting here are the ones that make the analysis
honest rather than just plausible: that expected loss is ranked by MONEY (not
rate), that structurally-impossible evidence is excluded from "addressable"
gaps rather than inflating them, and that the observational and model-based
estimates stay separate rather than being silently blended.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, root_cause


def _frame(rows):
    return pd.DataFrame(rows)


def test_expected_loss_is_money_not_rate():
    """A high-volume segment with a decent win rate can lose more money than a
    small segment with a terrible one. Ranking must follow the money."""
    df = _frame(
        # 100 disputes, 90% won, 10 losses x 1000 = 10,000 lost
        [{"seg": "big", "won": 1, "amount_inr": 1000.0}] * 90
        + [{"seg": "big", "won": 0, "amount_inr": 1000.0}] * 10
        # 10 disputes, 10% won, 9 losses x 100 = 900 lost
        + [{"seg": "small", "won": 1, "amount_inr": 100.0}] * 1
        + [{"seg": "small", "won": 0, "amount_inr": 100.0}] * 9
    )
    out = root_cause.segment_loss_analysis(df, "seg")

    assert out.index[0] == "big", "ranking must be by expected loss, not win rate"
    assert out.loc["big", "expected_loss_inr"] == pytest.approx(10_000.0)
    assert out.loc["small", "expected_loss_inr"] == pytest.approx(900.0)
    # ...even though "small" has by far the worse win rate
    assert out.loc["small", "win_rate"] < out.loc["big", "win_rate"]


def test_expected_loss_shares_sum_to_one():
    df = _frame(
        [{"seg": "a", "won": 0, "amount_inr": 500.0}] * 4
        + [{"seg": "b", "won": 0, "amount_inr": 250.0}] * 4
    )
    out = root_cause.segment_loss_analysis(df, "seg")
    assert out["pct_of_total_loss"].sum() == pytest.approx(1.0)


def test_digital_delivery_cannot_be_asked_for_tracking_or_pod():
    """Recommending 'go collect a tracking number' for a digital download is
    recommending an impossible action. Those rows must be excluded from the
    addressable gap, not counted as merchant negligence."""
    df = _frame([
        {"shipping_method": "digital_delivery", "won": 0, "amount_inr": 1000.0,
         "has_tracking_number": 0, "has_signed_pod": 0, "has_delivery_confirmation": 1,
         "has_courier_communication": 1, "has_avs_match": 1},
        {"shipping_method": "standard", "won": 0, "amount_inr": 1000.0,
         "has_tracking_number": 0, "has_signed_pod": 0, "has_delivery_confirmation": 1,
         "has_courier_communication": 1, "has_avs_match": 1},
        {"shipping_method": "standard", "won": 1, "amount_inr": 1000.0,
         "has_tracking_number": 1, "has_signed_pod": 1, "has_delivery_confirmation": 1,
         "has_courier_communication": 1, "has_avs_match": 1},
    ])
    out = root_cause.evidence_gap_analysis(df).set_index("evidence_type")

    assert out.loc["tracking_number", "missing_structurally_impossible"] == 1
    assert out.loc["tracking_number", "missing_addressable"] == 1  # only the standard-shipping one
    assert out.loc["signed_pod", "missing_structurally_impossible"] == 1
    # delivery_confirmation is collectable for digital orders, so nothing is excluded
    assert out.loc["delivery_confirmation", "missing_structurally_impossible"] == 0


def test_structural_mask_only_applies_to_parcel_evidence():
    df = _frame([{"shipping_method": "digital_delivery"}, {"shipping_method": "standard"}])
    assert root_cause.structurally_impossible_mask(df, "tracking_number").tolist() == [True, False]
    assert root_cause.structurally_impossible_mask(df, "signed_pod").tolist() == [True, False]
    # AVS and courier comms are not parcel-dependent
    assert root_cause.structurally_impossible_mask(df, "avs_match").tolist() == [False, False]
    assert root_cause.structurally_impossible_mask(df, "courier_communication").tolist() == [False, False]


def test_amount_at_stake_counts_only_addressable_rows():
    """Money 'at stake' behind an evidence gap must exclude the rows where the
    evidence could never have been collected -- otherwise the headline number
    is inflated by cases nobody could have acted on."""
    df = _frame([
        {"shipping_method": "digital_delivery", "won": 0, "amount_inr": 9999.0,
         "has_tracking_number": 0, "has_signed_pod": 1, "has_delivery_confirmation": 1,
         "has_courier_communication": 1, "has_avs_match": 1},
        {"shipping_method": "standard", "won": 0, "amount_inr": 100.0,
         "has_tracking_number": 0, "has_signed_pod": 1, "has_delivery_confirmation": 1,
         "has_courier_communication": 1, "has_avs_match": 1},
    ])
    out = root_cause.evidence_gap_analysis(df).set_index("evidence_type")
    # the 9,999 digital-delivery row must NOT be counted
    assert out.loc["tracking_number", "amount_at_stake_inr"] == pytest.approx(100.0)


def test_late_filing_gap_is_directional_and_costed():
    df = _frame(
        [{"late_filing": 1, "won": 0, "amount_inr": 1000.0}] * 5
        + [{"late_filing": 1, "won": 1, "amount_inr": 1000.0}] * 5
        + [{"late_filing": 0, "won": 1, "amount_inr": 1000.0}] * 9
        + [{"late_filing": 0, "won": 0, "amount_inr": 1000.0}] * 1
    )
    out = root_cause.late_filing_analysis(df)
    assert out["late_count"] == 10
    assert out["late_share"] == pytest.approx(0.5)
    assert out["win_rate_late"] == pytest.approx(0.5)
    assert out["win_rate_on_time"] == pytest.approx(0.9)
    assert out["gap_pp"] == pytest.approx(40.0)
    assert out["expected_loss_late_inr"] == pytest.approx(5000.0)


def test_observational_gap_is_reported_separately_from_any_causal_claim():
    """The observational columns must exist under names that say what they are.
    This is a guard against someone later renaming these to something that
    reads as a causal 'recoverable revenue' promise -- the whole module
    docstring exists because that number would be confounded here."""
    df = _frame([
        {"shipping_method": "standard", "won": 1, "amount_inr": 100.0,
         "has_tracking_number": 1, "has_signed_pod": 1, "has_delivery_confirmation": 1,
         "has_courier_communication": 1, "has_avs_match": 1},
        {"shipping_method": "standard", "won": 0, "amount_inr": 100.0,
         "has_tracking_number": 0, "has_signed_pod": 0, "has_delivery_confirmation": 0,
         "has_courier_communication": 0, "has_avs_match": 0},
    ])
    cols = set(root_cause.evidence_gap_analysis(df).columns)
    assert "observational_gap_pp" in cols
    assert "amount_at_stake_inr" in cols
    # nothing in the observational table may claim to be recoverable/causal
    assert not any("recover" in c.lower() or "causal" in c.lower() for c in cols)
