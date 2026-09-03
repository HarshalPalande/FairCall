"""Tests for the rule-based Evidence Completeness Checker (src/evidence.py)."""
import pytest

from src.evidence import completeness_score, evaluate_evidence


def test_full_evidence_gives_100_percent():
    flags = {
        "tracking_number": 1,
        "delivery_confirmation": 1,
        "signed_pod": 1,
        "courier_communication": 1,
        "avs_match": 1,
    }
    result = completeness_score(flags)
    assert result.completeness_score == 1.0
    assert result.missing_documents == []


def test_no_evidence_gives_zero():
    flags = {
        "tracking_number": 0,
        "delivery_confirmation": 0,
        "signed_pod": 0,
        "courier_communication": 0,
        "avs_match": 0,
    }
    result = completeness_score(flags)
    assert result.completeness_score == 0.0
    assert len(result.missing_documents) == 3  # 3 required docs for reason code 13.1


def test_partial_evidence_score():
    flags = {
        "tracking_number": 1,
        "delivery_confirmation": 0,
        "signed_pod": 0,
        "courier_communication": 1,
        "avs_match": 1,
    }
    result = completeness_score(flags)
    assert result.completeness_score == pytest.approx(1 / 3, abs=0.01)
    assert "delivery_confirmation" in result.missing_documents
    assert "signed_pod" in result.missing_documents


def test_digital_delivery_signed_pod_inconsistency():
    flags = {
        "tracking_number": 0,
        "delivery_confirmation": 1,
        "signed_pod": 1,
        "courier_communication": 0,
        "avs_match": 1,
    }
    result = evaluate_evidence(flags, shipping_method="digital_delivery")
    assert result.consistency_ok is False
    assert any("digital_delivery" in note for note in result.consistency_notes)


def test_signed_pod_without_tracking_inconsistency():
    flags = {
        "tracking_number": 0,
        "delivery_confirmation": 1,
        "signed_pod": 1,
        "courier_communication": 0,
        "avs_match": 0,
    }
    result = evaluate_evidence(flags, shipping_method="standard")
    assert result.consistency_ok is False
    assert any("tracking_number" in note for note in result.consistency_notes)


def test_consistent_evidence_passes():
    flags = {
        "tracking_number": 1,
        "delivery_confirmation": 1,
        "signed_pod": 1,
        "courier_communication": 0,
        "avs_match": 0,
    }
    result = evaluate_evidence(flags, shipping_method="standard")
    assert result.consistency_ok is True
    assert result.consistency_notes == []
