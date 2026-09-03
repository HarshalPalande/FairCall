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

# Additional cost for a HUMAN analyst to manually prepare and submit a
# contest evidence packet: reviewing the dispute, gathering documents,
# writing the response, submitting through the acquirer portal. This is on
# top of CONTEST_COST_INR (which already covers the acquirer admin fee and
# a baseline packet-assembly cost) — it's the extra cost specific to a human
# doing that assembly by hand instead of the system doing it automatically
# for an AUTO_CONTEST case. Used by the backtest (src/backtest.py) to give
# "contest everything blindly" its real, non-zero labor cost instead of
# implicitly assuming evidence packets prepare themselves for free.
# PLACEHOLDER — same caveat as the rest of this cost table: in production,
# sourced from ops team time-tracking data, not invented by an engineer.
ANALYST_LABOR_COST_PER_CASE_INR = 200.0

TIME_COL = "dispute_date"
LABEL_COL = "won"

# --- VAMP (Visa Acquirer Monitoring Program) parameters -------------------
# Effective 1 April 2026, the merchant "Excessive" VAMP threshold dropped from
# 2.2% to 1.5% in US/Canada/EU/APAC/LATAM (CEMEA remains 2.2%).
#
# VAMP Ratio = (TC40 fraud reports + TC15 disputes) / settled CNP transactions
#
# KEY POINT (see README): this ratio counts disputes FILED, not disputes LOST.
# Winning a representment does not remove the dispute from the ratio. Only
# pre-dispute deflection (CE3.0 via Order Insight, RDR) keeps it off.
VAMP_EXCESSIVE_THRESHOLD = 0.015          # 1.5% as of 1 April 2026
VAMP_MONITORING_FLOOR_EVENTS = 1500       # min combined events/month to be monitored
VAMP_PENALTY_PER_DISPUTE_USD = 8.0        # $8 per disputed/fraudulent txn at Excessive
USD_TO_INR = 88.0                         # PLACEHOLDER FX rate — disclosed, not sourced
VAMP_PENALTY_PER_DISPUTE_INR = VAMP_PENALTY_PER_DISPUTE_USD * USD_TO_INR

# Simulated merchant portfolio context for the demo. In production these would
# come from the merchant's actual settled-transaction and dispute counts.
# PLACEHOLDER values, disclosed as such.
MERCHANT_MONTHLY_SETTLED_TXNS = 200_000
MERCHANT_MONTHLY_DISPUTE_EVENTS = 2_600   # ~1.30% — deliberately near the 1.5% line

# --- Visa Compelling Evidence 3.0 (CE3.0) --------------------------------
# CE3.0 applies to Visa reason code 10.4 (Fraud — Card Absent Environment).
# It lets a merchant block a first-party fraud dispute by proving the cardholder
# had a legitimate prior transaction history with them.
#
# NOTE ON SCOPE: the rest of this project is scoped to reason code 13.1.
# CE3.0 is implemented here to show the rule-engine pattern extending to a
# second reason code — it is NOT applicable to 13.1 disputes.
CE3_REASON_CODE = "10.4"

# Prior transactions must fall in this window relative to the disputed transaction.
CE3_MIN_PRIOR_AGE_DAYS = 120
CE3_MAX_PRIOR_AGE_DAYS = 365

# Minimum number of prior undisputed transactions required.
CE3_MIN_PRIOR_TRANSACTIONS = 2

# Data elements that can be matched across transactions.
CE3_MATCHABLE_ELEMENTS = [
    "ip_address",
    "device_id",
    "account_login_id",
    "shipping_address",
]

# At least one matched element MUST come from this set.
CE3_ANCHOR_ELEMENTS = ["ip_address", "device_id"]

# Minimum total number of matching elements.
CE3_MIN_MATCHING_ELEMENTS = 2

# --- Visa RDR (Rapid Dispute Resolution) ----------------------------------
# RDR auto-resolves eligible pre-disputes against merchant-configured rules:
# the cardholder is refunded immediately and no chargeback posts.
#
# CRITICAL ASYMMETRY (see README):
#   - Non-fraud pre-disputes resolved via RDR are EXCLUDED from the VAMP ratio.
#   - Fraud (10.4) disputes still COUNT toward VAMP even when RDR refunds them.
# So RDR is a strong lever for non-fraud codes like 13.1 and a much weaker one
# for fraud codes.
#
# PLACEHOLDER cost assumptions — same disclosure standard as the rest of the
# cost table: set by us, not sourced from Razorpay finance/risk ops.
RDR_RESOLUTION_FEE_INR = 250.0

# Visa permits a limited number of merchant-configured RDR rule scenarios.
RDR_MAX_RULE_SCENARIOS = 10

# Reason codes for which an RDR resolution keeps the event off the VAMP ratio.
RDR_VAMP_EXCLUDED_REASON_CODES = ["13.1", "13.2", "13.3", "12.6"]

# --- Order Insight (pre-dispute deflection) -------------------------------
# Order Insight pushes transaction detail to the issuer at cardholder inquiry,
# resolving billing-confusion cases before a dispute is filed at all. It only
# works when the merchant actually HAS rich order data to push.
ORDER_INSIGHT_COST_INR = 40.0
# Probability a deflection attempt succeeds given complete order data.
# PLACEHOLDER — in production this is measured from actual deflection rates.
ORDER_INSIGHT_BASE_DEFLECT_RATE = 0.35
