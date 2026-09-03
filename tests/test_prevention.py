"""Tests for the Dispute Prevention Score model (src/prevention.py)."""
import json
import pickle

import pytest

from src import config
from src.prevention import build_prevention_features, build_transaction_dataset, score_transaction


def test_transaction_dataset_has_both_classes():
    data = build_transaction_dataset(seed=config.SEED)
    assert data["became_dispute"].sum() > 0
    assert (data["became_dispute"] == 0).sum() > 0
    # Should have ~5:1 ratio normal:dispute
    ratio = (data["became_dispute"] == 0).sum() / data["became_dispute"].sum()
    assert 4.5 < ratio < 5.5


def test_transaction_dataset_is_sorted_by_date():
    data = build_transaction_dataset(seed=config.SEED)
    dates = data["transaction_date"].values
    assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


def test_prevention_features_have_no_evidence_columns():
    """Prevention model must NOT use evidence flags — those aren't available at transaction time."""
    data = build_transaction_dataset(seed=config.SEED)
    X = build_prevention_features(data)
    evidence_cols = [col for col in X.columns if "has_" in col]
    assert evidence_cols == [], f"Evidence columns found in prevention features: {evidence_cols}"


def test_prevention_features_have_no_dispute_columns():
    """Prevention model must NOT use dispute-related columns."""
    data = build_transaction_dataset(seed=config.SEED)
    X = build_prevention_features(data)
    forbidden = [col for col in X.columns if any(x in col.lower() for x in ["dispute", "won", "late_filing"])]
    assert forbidden == [], f"Dispute columns found in prevention features: {forbidden}"


def test_score_transaction_returns_valid_tier():
    try:
        with open(config.MODELS_DIR / "prevention_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(config.MODELS_DIR / "prevention_feature_cols.json") as f:
            feature_cols = json.load(f)
    except FileNotFoundError:
        pytest.skip("Prevention model not trained yet — run `make train` first")

    txn = {
        "amount_inr": 5000.0,
        "merchant_category": "electronics",
        "device_type": "mobile",
        "shipping_method": "standard",
    }
    result = score_transaction(txn, model, feature_cols)
    assert result["risk_tier"] in ("HIGH", "MEDIUM", "LOW")
    assert 0.0 <= result["dispute_risk_score"] <= 1.0
    assert len(result["recommendation"]) > 0
