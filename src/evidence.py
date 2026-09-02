"""
Evidence Completeness Checker — rule-based, no ML.

Litmus test from the project brief: "if the ML model disappeared tomorrow,
would the product still be useful?" This module is the proof: it answers
"is this evidence packet complete and consistent for reason code 13.1?"
using nothing but a lookup table and arithmetic. It has standalone value
and is what the Counterfactual Engine and EV Decision Engine both consult
before any model score is even considered.
"""
from dataclasses import dataclass, field

from src import config


@dataclass
class CompletenessResult:
    reason_code: str
    completeness_score: float
    missing_documents: list = field(default_factory=list)
    present_documents: list = field(default_factory=list)
    consistency_ok: bool = True
    consistency_notes: list = field(default_factory=list)


def completeness_score(evidence_flags: dict, reason_code: str = config.REASON_CODE) -> CompletenessResult:
    """evidence_flags: dict like {'tracking_number': 1, 'delivery_confirmation': 0, ...}"""
    required = config.REQUIRED_EVIDENCE_FOR_13_1 if reason_code == "13.1" else config.EVIDENCE_TYPES
    present = [e for e in required if evidence_flags.get(e, 0) == 1]
    missing = [e for e in required if evidence_flags.get(e, 0) == 0]
    score = len(present) / len(required) if required else 0.0
    return CompletenessResult(
        reason_code=reason_code,
        completeness_score=round(score, 4),
        missing_documents=missing,
        present_documents=present,
    )


def consistency_check(evidence_flags: dict, shipping_method: str) -> tuple[bool, list]:
    """Cheap, explainable sanity rules that catch obviously-manipulated or
    self-contradictory evidence packets. Not a fraud model — just guards
    against nonsensical submissions reaching the EV engine."""
    notes = []
    ok = True
    if shipping_method == "digital_delivery" and evidence_flags.get("signed_pod", 0) == 1:
        ok = False
        notes.append("signed_pod present for a digital_delivery order — physical POD is not possible here.")
    if evidence_flags.get("signed_pod", 0) == 1 and evidence_flags.get("tracking_number", 0) == 0:
        ok = False
        notes.append("signed_pod present without a tracking_number — a courier cannot produce a POD without a shipment record.")
    return ok, notes


def evaluate_evidence(evidence_flags: dict, shipping_method: str, reason_code: str = config.REASON_CODE) -> CompletenessResult:
    result = completeness_score(evidence_flags, reason_code)
    ok, notes = consistency_check(evidence_flags, shipping_method)
    result.consistency_ok = ok
    result.consistency_notes = notes
    return result
