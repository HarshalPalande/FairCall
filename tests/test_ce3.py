"""Tests for the CE3.0 qualification rules engine (src/ce3.py)."""
from datetime import datetime, timedelta

from src import ce3
from src.ce3 import PriorTransaction

DISPUTE_DATE = datetime(2026, 9, 1)


def _disputed(**overrides):
    base = {
        "date": DISPUTE_DATE,
        "ip_address": "203.0.113.45",
        "device_id": "dev_a1b2c3",
        "account_login_id": "user_9981",
        "shipping_address": "12 MG Road, Pune",
    }
    base.update(overrides)
    return base


def _prior(txn_id, days_ago, disputed=False, **overrides):
    base = {
        "ip_address": "203.0.113.45",
        "device_id": "dev_a1b2c3",
        "account_login_id": "user_9981",
        "shipping_address": "12 MG Road, Pune",
    }
    base.update(overrides)
    return PriorTransaction(
        transaction_id=txn_id,
        date=DISPUTE_DATE - timedelta(days=days_ago),
        was_disputed=disputed,
        **base,
    )


def test_qualifies_with_two_matching_priors_in_window():
    result = ce3.check_ce3_qualification(_disputed(), [_prior("A", 180), _prior("B", 200)], "10.4")
    assert result.qualifies is True
    assert len(result.matching_elements) >= 2


def test_rejects_wrong_reason_code():
    result = ce3.check_ce3_qualification(_disputed(), [_prior("A", 180), _prior("B", 200)], "13.1")
    assert result.qualifies is False
    assert result.checks["reason_code_eligible"] is False


def test_rejects_priors_too_recent():
    """Priors under 120 days don't count."""
    result = ce3.check_ce3_qualification(_disputed(), [_prior("A", 30), _prior("B", 60)], "10.4")
    assert result.qualifies is False
    assert result.checks["prior_transactions_in_window"] == 0


def test_rejects_priors_too_old():
    """Priors over 365 days don't count."""
    result = ce3.check_ce3_qualification(_disputed(), [_prior("A", 400), _prior("B", 500)], "10.4")
    assert result.qualifies is False
    assert result.checks["prior_transactions_in_window"] == 0


def test_rejects_disputed_prior():
    """A prior that was itself disputed is not eligible."""
    result = ce3.check_ce3_qualification(
        _disputed(),
        [_prior("A", 180), _prior("B", 200, disputed=True)],
        "10.4",
    )
    assert result.qualifies is False


def test_rejects_only_one_prior():
    result = ce3.check_ce3_qualification(_disputed(), [_prior("A", 180)], "10.4")
    assert result.qualifies is False


def test_requires_ip_or_device_anchor():
    """Matching only on address + login is insufficient — need IP or device."""
    disputed = _disputed(ip_address="198.51.100.7", device_id="dev_different")
    priors = [
        _prior("A", 180, ip_address="203.0.113.45", device_id="dev_a1b2c3"),
        _prior("B", 200, ip_address="192.0.2.99", device_id="dev_other"),
    ]
    result = ce3.check_ce3_qualification(disputed, priors, "10.4")
    assert result.qualifies is False
    assert result.checks.get("has_anchor_element") is False


def test_deterministic_same_input_same_output():
    """A compliance-adjacent engine must be deterministic."""
    args = (_disputed(), [_prior("A", 180), _prior("B", 200)], "10.4")
    r1 = ce3.check_ce3_qualification(*args)
    r2 = ce3.check_ce3_qualification(*args)
    assert r1.qualifies == r2.qualifies
    assert r1.matching_elements == r2.matching_elements
