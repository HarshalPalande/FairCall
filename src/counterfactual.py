"""
Counterfactual Engine — "what evidence would help most?"

Takes a scored dispute's feature vector, flips one evidence flag at a time
(0->1 for missing docs), re-scores with the SAME calibrated model, and
ranks evidence types by how much they'd move the win probability. This is
the "58% -> 84% with delivery proof" hero moment from the brief: it's a
targeted recommendation, not just a number.
"""
from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass
class CounterfactualResult:
    evidence_type: str
    current_value: int
    baseline_prob: float
    what_if_prob: float
    delta: float


def score_with_calibration(model, calibrator, X_row: pd.DataFrame) -> float:
    raw = model.predict_proba(X_row)[:, 1][0]
    return float(calibrator.predict([raw])[0])


def rank_evidence_impact(model, calibrator, feature_row: pd.Series, feature_cols: list) -> list[CounterfactualResult]:
    """feature_row: a single feature Series (as produced by
    features.feature_vector_for_case), already reindexed to feature_cols."""
    X_row = feature_row.reindex(feature_cols).fillna(0).to_frame().T
    baseline_prob = score_with_calibration(model, calibrator, X_row)

    results = []
    for etype in config.EVIDENCE_TYPES:
        col = f"has_{etype}"
        if col not in feature_cols:
            continue
        current_value = int(feature_row.get(col, 0))
        if current_value == 1:
            continue  # already present — no counterfactual to offer
        X_cf = X_row.copy()
        X_cf[col] = 1
        what_if_prob = score_with_calibration(model, calibrator, X_cf)
        results.append(CounterfactualResult(
            evidence_type=etype,
            current_value=current_value,
            baseline_prob=round(baseline_prob, 4),
            what_if_prob=round(what_if_prob, 4),
            delta=round(what_if_prob - baseline_prob, 4),
        ))
    results.sort(key=lambda r: r.delta, reverse=True)
    return results
