"""
End-to-end tests for the scoring pipeline (src/pipeline.py).

Every other test file exercises one component in isolation — the EV engine, the
evidence checker, the prevention dataset, the backtest arithmetic. None of them
test that `score_dispute()` actually wires those components together, which is
the thing the Streamlit UI, `scripts/demo.py` and `src/backtest.py` all call.
That's the product; this file tests the product.

The model here is trained inside the fixture on a small slice rather than loaded
from `models/`, so these tests run on a cold clone with no `make train` and stay
fast (~3s) while still using the real training path from src/.
"""
import json

import pytest
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

from src import audit, config, data_gen, pipeline
from src.evidence import evaluate_evidence
from src.features import build_feature_frame

BASE_CASE = {
    "transaction_id": "TEST-001",
    "customer_id": "CUST000001",
    "amount_inr": 5_000.0,
    "merchant_category": "electronics",
    "device_type": "mobile",
    "shipping_method": "standard",
    "late_filing": 0,
    "account_age_days_at_dispute": 180,
}

FULL_EVIDENCE = {f"has_{e}": 1 for e in config.EVIDENCE_TYPES}
NO_EVIDENCE = {f"has_{e}": 0 for e in config.EVIDENCE_TYPES}


@pytest.fixture(scope="module")
def artifacts():
    """Train a small detector + calibrator through the real code path in src/."""
    df = data_gen.generate_disputes(n=3_000, seed=config.SEED)
    X, df_feat = build_feature_frame(df)
    y = df_feat[config.LABEL_COL]

    split = int(len(X) * 0.8)
    model = xgb.XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.1, eval_metric="aucpr",
        random_state=config.SEED,
    )
    model.fit(X.iloc[:split], y.iloc[:split])

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(X.iloc[split:])[:, 1], y.iloc[split:])

    return model, calibrator, list(X.columns), df_feat.iloc[:split]


@pytest.fixture
def audit_log(tmp_path, monkeypatch):
    """Redirect the pipeline's audit writes to a temp log so tests never touch
    the real audit_log/decisions.jsonl."""
    log_path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(
        pipeline, "log_decision", lambda body: audit.log_decision(body, log_path=log_path)
    )
    return log_path


def _score(case_extra, artifacts, write_audit=False):
    model, calibrator, feature_cols, history = artifacts
    case = {**BASE_CASE, **case_extra}
    return pipeline.score_dispute(case, model, calibrator, feature_cols, history, write_audit=write_audit)


def test_score_dispute_returns_a_complete_result(artifacts):
    result = _score(FULL_EVIDENCE, artifacts)

    assert set(result) == {
        "transaction_id", "win_probability", "evidence", "counterfactuals", "ev_decision"
    }
    assert result["transaction_id"] == "TEST-001"
    assert 0.0 <= result["win_probability"] <= 1.0

    ev = result["ev_decision"]
    assert ev["action"].value in ("AUTO_CONTEST", "ESCALATE")
    assert ev["amount"] == BASE_CASE["amount_inr"]
    assert ev["reasons"], "a decision must always carry its reasoning"


def test_evidence_result_matches_the_standalone_checker(artifacts):
    """The pipeline must not compute completeness differently from the
    rule-based checker the README says is usable on its own."""
    case_extra = {**NO_EVIDENCE, "has_tracking_number": 1}
    result = _score(case_extra, artifacts)

    standalone = evaluate_evidence(
        {e: case_extra.get(f"has_{e}", 0) for e in config.EVIDENCE_TYPES},
        BASE_CASE["shipping_method"],
    )
    assert result["evidence"]["completeness_score"] == standalone.completeness_score
    assert result["evidence"]["missing_documents"] == standalone.missing_documents


def test_more_evidence_raises_win_probability(artifacts):
    """Directional sanity on the wiring: the same dispute with a full evidence
    packet must not score worse than with none."""
    none = _score(NO_EVIDENCE, artifacts)
    full = _score(FULL_EVIDENCE, artifacts)
    assert full["win_probability"] > none["win_probability"]


def test_missing_evidence_blocks_auto_contest(artifacts):
    """Zero evidence must escalate regardless of anything else — the gate the
    EV engine unit tests assert in isolation, here through the real pipeline."""
    result = _score(NO_EVIDENCE, artifacts)
    ev = result["ev_decision"]
    assert ev["action"].value == "ESCALATE"
    assert ev["evidence_blocked"] is True


def test_hard_ceiling_blocks_auto_contest_through_the_pipeline(artifacts):
    """Same case, same evidence, only the amount changes — the ₹ ceiling must
    flip the decision. This is demo case #3, asserted rather than narrated."""
    under = _score({**FULL_EVIDENCE, "amount_inr": config.HARD_CEILING_INR - 1_000}, artifacts)
    over = _score({**FULL_EVIDENCE, "amount_inr": config.HARD_CEILING_INR + 1_000}, artifacts)

    assert over["ev_decision"]["ceiling_blocked"] is True
    assert over["ev_decision"]["action"].value == "ESCALATE"
    assert under["ev_decision"]["ceiling_blocked"] is False


def test_counterfactuals_only_offered_for_missing_evidence(artifacts):
    """No point recommending a document the merchant already has."""
    result = _score({**NO_EVIDENCE, "has_tracking_number": 1}, artifacts)
    offered = {c["evidence_type"] for c in result["counterfactuals"]}

    assert "tracking_number" not in offered
    assert offered, "some missing evidence should have been rankable"
    for c in result["counterfactuals"]:
        assert c["current_value"] == 0
        assert c["delta"] == pytest.approx(c["what_if_prob"] - c["baseline_prob"], abs=1e-4)


def test_counterfactuals_are_ranked_by_impact(artifacts):
    result = _score(NO_EVIDENCE, artifacts)
    deltas = [c["delta"] for c in result["counterfactuals"]]
    assert deltas == sorted(deltas, reverse=True)


def test_pipeline_writes_a_verifiable_audit_entry(artifacts, audit_log):
    """The decision and the audit record must agree, and the chain must verify."""
    result = _score(FULL_EVIDENCE, artifacts, write_audit=True)

    ok, msg = audit.verify_chain(audit_log)
    assert ok is True, msg

    entry = json.loads(audit_log.read_text().splitlines()[-1])
    body = entry["body"]
    assert body["transaction_id"] == "TEST-001"
    assert body["action"] == result["ev_decision"]["action"].value
    assert body["win_probability"] == result["win_probability"]
    assert body["ev_math"]["ev_contest"] == result["ev_decision"]["ev_contest"]
    assert "timestamp" in body


def test_audit_entry_is_written_before_any_downstream_action(artifacts, audit_log):
    """Two scored disputes produce two chained entries, in order."""
    _score(FULL_EVIDENCE, artifacts, write_audit=True)
    _score(NO_EVIDENCE, artifacts, write_audit=True)

    lines = audit_log.read_text().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert second["prev_hash"] == first["entry_hash"]

    ok, _ = audit.verify_chain(audit_log)
    assert ok is True


def test_no_audit_entry_when_write_audit_is_false(artifacts, audit_log):
    """src/backtest.py scores thousands of disputes with write_audit=False —
    that must genuinely write nothing."""
    _score(FULL_EVIDENCE, artifacts, write_audit=False)
    assert not audit_log.exists()
