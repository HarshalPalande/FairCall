"""
Central configuration: seed, paths, and the cost table the EV engine runs on.

All monetary values below are PLACEHOLDER assumptions for demo purposes.
In production these would be sourced from finance/risk ops, not invented by
an engineer. We say so here and in the README rather than hiding it.
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
AUDIT_LOG_PATH = ROOT / "audit_log" / "decisions.jsonl"

for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, AUDIT_LOG_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

REASON_CODE = "13.1"  # Visa 13.1: "Merchandise / Services Not Received"

N_DISPUTES = 20_000
DATE_RANGE_DAYS = 540  # ~18 months of synthetic history

# Evidence document types required/relevant for reason code 13.1.
# Order matters for nothing; presence of each is a binary feature.
EVIDENCE_TYPES = [
    "tracking_number",
    "delivery_confirmation",
    "signed_pod",          # signed proof of delivery
    "courier_communication",
    "avs_match",            # address verification service match at purchase
]

# Rule-based: which docs actually matter for THIS reason code, per card-network
# guidance for "not received" disputes. Used by the Evidence Completeness
# Checker (src/evidence.py) independent of any ML model.
REQUIRED_EVIDENCE_FOR_13_1 = [
    "tracking_number",
    "delivery_confirmation",
    "signed_pod",
]

# --- EV Decision Engine cost table (PLACEHOLDER, see README) ---------------
# Fixed ₹ cost to prepare + submit a contest packet (analyst/system labor).
CONTEST_LABOR_COST_INR = 150.0
# Card-network/acquirer administrative fee charged on a contested dispute,
# levied regardless of who wins.
CHARGEBACK_ADMIN_FEE_INR = 350.0
CONTEST_COST_INR = CONTEST_LABOR_COST_INR + CHARGEBACK_ADMIN_FEE_INR  # 500.0

# Hard ceiling: disputes above this amount ALWAYS escalate to a human,
# no matter how favorable the EV math looks. This is a fixed rule, not a
# learned parameter, and cannot be overridden by model confidence.
HARD_CEILING_INR = 25_000.0

# Minimum evidence completeness (see src/evidence.py) required before the
# system is allowed to auto-contest, even if EV math is favorable. Below
# this, the case escalates regardless of predicted win probability.
MIN_EVIDENCE_COMPLETENESS_FOR_AUTO = 0.6

TIME_COL = "dispute_date"
LABEL_COL = "won"
