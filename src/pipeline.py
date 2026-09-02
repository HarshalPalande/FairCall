"""
End-to-end scoring pipeline for a single dispute: ties together the
detector, calibrator, evidence checker, counterfactual engine, EV decision
engine, and audit log. This is what both scripts/demo.py and the Streamlit
UI call — one code path for both, so the demo can never drift from what a
real request would do.
"""
import json
import pickle
from dataclasses import asdict

import pandas as pd

from src import config, ev_engine
from src.audit import log_decision
from src.counterfactual import rank_evidence_impact, score_with_calibration
from src.evidence import evaluate_evidence
from src.features import feature_vector_for_case


def load_artifacts():
    with open(config.MODELS_DIR / "detector.pkl", "rb") as f:
        model = pickle.load(f)
    with open(config.MODELS_DIR / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(config.MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    history_df = pd.read_csv(config.DATA_DIR / "history_train.csv", parse_dates=["dispute_date", "transaction_date"])
    return model, calibrator, feature_cols, history_df


def score_dispute(raw_case: dict, model, calibrator, feature_cols, history_df, write_audit=True) -> dict:
    """raw_case: dict with amount_inr, shipping_method, merchant_category,
    device_type, late_filing, account_age_days_at_dispute, customer_id,
    and has_<evidence_type> flags."""
    evidence_flags = {e: raw_case.get(f"has_{e}", 0) for e in config.EVIDENCE_TYPES}

    evidence_result = evaluate_evidence(evidence_flags, raw_case["shipping_method"])

    feature_row = feature_vector_for_case(raw_case, history_df)
    X_row = feature_row.reindex(feature_cols).fillna(0).to_frame().T
    win_prob = score_with_calibration(model, calibrator, X_row)

    counterfactuals = rank_evidence_impact(model, calibrator, feature_row, feature_cols)

    decision = ev_engine.decide(
        win_prob=win_prob,
        amount=raw_case["amount_inr"],
        evidence_completeness=evidence_result.completeness_score,
    )

    result = {
        "transaction_id": raw_case.get("transaction_id", "UNKNOWN"),
        "win_probability": round(win_prob, 4),
        "evidence": asdict(evidence_result),
        "counterfactuals": [asdict(c) for c in counterfactuals],
        "ev_decision": asdict(decision),
    }

    if write_audit:
        log_decision({
            "transaction_id": result["transaction_id"],
            "input_case": {k: v for k, v in raw_case.items() if k != "customer_id" or True},
            "win_probability": result["win_probability"],
            "evidence_completeness": evidence_result.completeness_score,
            "evidence_missing": evidence_result.missing_documents,
            "ev_math": {"ev_contest": decision.ev_contest, "ev_accept": decision.ev_accept},
            "action": decision.action.value,
            "reasons": decision.reasons,
        })

    return result
