"""
Visa Compelling Evidence 3.0 (CE3.0) qualification engine.

Pure rule-based, no ML. Determines whether a disputed transaction qualifies for
CE3.0 treatment, which — if it does — blocks the dispute pre-emptively and shifts
liability to the issuer, keeping the event off the merchant's VAMP ratio entirely.

THE RULES (from Visa's published CE3.0 criteria):
  1. Applies to reason code 10.4 (Fraud — Card Absent Environment) only.
  2. Need at least 2 prior UNDISPUTED transactions from the same cardholder.
  3. Those transactions must be 120-365 days before the disputed transaction.
  4. At least 2 data elements must match across the disputed transaction and the
     prior transactions.
  5. At least one matching element must be IP address or device ID/fingerprint.

This is deliberately a deterministic checker: given the same inputs it always
returns the same answer, and every rejection reason is explicit. That matters for
a compliance-adjacent feature — a merchant needs to know exactly why a dispute
did or did not qualify, not a probability.

SCOPE NOTE: CE3.0 applies to reason code 10.4. The rest of this project (the
detector, EV engine, evidence checker) is scoped to reason code 13.1. This
module does not extend or change that scope — it demonstrates the same
rule-engine pattern (src/evidence.py) applied to a second, higher-value reason
code where the entire mechanism is deterministic rules, not a model at all.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src import config


@dataclass
class PriorTransaction:
    transaction_id: str
    date: datetime
    was_disputed: bool
    ip_address: str = None
    device_id: str = None
    account_login_id: str = None
    shipping_address: str = None


@dataclass
class CE3Result:
    qualifies: bool
    reason_code: str
    matching_elements: list = field(default_factory=list)
    qualifying_transactions: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)


def _elements_matching(disputed: dict, prior: PriorTransaction) -> list:
    """Which data elements match between the disputed txn and a prior one."""
    matches = []
    for element in config.CE3_MATCHABLE_ELEMENTS:
        disputed_val = disputed.get(element)
        prior_val = getattr(prior, element, None)
        if disputed_val and prior_val and disputed_val == prior_val:
            matches.append(element)
    return matches


def check_ce3_qualification(
    disputed_transaction: dict,
    prior_transactions: list,
    reason_code: str,
) -> CE3Result:
    """
    disputed_transaction: dict with keys 'date' (datetime) plus any of
        ip_address, device_id, account_login_id, shipping_address.
    prior_transactions: list of PriorTransaction.
    """
    result = CE3Result(qualifies=False, reason_code=reason_code)

    # --- Rule 1: reason code must be 10.4 ---
    if reason_code != config.CE3_REASON_CODE:
        result.failures.append(
            f"CE3.0 applies only to reason code {config.CE3_REASON_CODE} "
            f"(Fraud — Card Absent Environment). This dispute is {reason_code}."
        )
        result.checks["reason_code_eligible"] = False
        return result
    result.checks["reason_code_eligible"] = True

    disputed_date = disputed_transaction["date"]
    window_start = disputed_date - timedelta(days=config.CE3_MAX_PRIOR_AGE_DAYS)
    window_end = disputed_date - timedelta(days=config.CE3_MIN_PRIOR_AGE_DAYS)

    # --- Rules 2 & 3: undisputed prior transactions inside the date window ---
    eligible = []
    for txn in prior_transactions:
        if txn.was_disputed:
            continue
        if not (window_start <= txn.date <= window_end):
            continue
        eligible.append(txn)

    result.checks["prior_transactions_in_window"] = len(eligible)

    if len(eligible) < config.CE3_MIN_PRIOR_TRANSACTIONS:
        result.failures.append(
            f"Found {len(eligible)} eligible prior undisputed transaction(s) in the "
            f"{config.CE3_MIN_PRIOR_AGE_DAYS}-{config.CE3_MAX_PRIOR_AGE_DAYS} day "
            f"window; CE3.0 requires at least {config.CE3_MIN_PRIOR_TRANSACTIONS}."
        )
        return result

    # --- Rules 4 & 5: matching data elements ---
    # Find elements that match across the disputed txn AND at least 2 priors.
    element_match_counts = {e: 0 for e in config.CE3_MATCHABLE_ELEMENTS}
    per_txn_matches = {}

    for txn in eligible:
        matches = _elements_matching(disputed_transaction, txn)
        per_txn_matches[txn.transaction_id] = matches
        for m in matches:
            element_match_counts[m] += 1

    consistent_elements = [
        e for e, count in element_match_counts.items() if count >= config.CE3_MIN_PRIOR_TRANSACTIONS
    ]
    result.matching_elements = consistent_elements
    result.checks["matching_elements_count"] = len(consistent_elements)

    if len(consistent_elements) < config.CE3_MIN_MATCHING_ELEMENTS:
        result.failures.append(
            f"Only {len(consistent_elements)} data element(s) match consistently across "
            f"{config.CE3_MIN_PRIOR_TRANSACTIONS}+ prior transactions "
            f"({', '.join(consistent_elements) if consistent_elements else 'none'}); "
            f"CE3.0 requires at least {config.CE3_MIN_MATCHING_ELEMENTS}."
        )
        return result

    has_anchor = any(e in config.CE3_ANCHOR_ELEMENTS for e in consistent_elements)
    result.checks["has_anchor_element"] = has_anchor

    if not has_anchor:
        result.failures.append(
            f"At least one matching element must be IP address or device ID/fingerprint. "
            f"Matched elements were: {', '.join(consistent_elements)}."
        )
        return result

    # --- Qualified ---
    result.qualifies = True
    result.qualifying_transactions = [
        t.transaction_id
        for t in eligible
        if any(m in consistent_elements for m in per_txn_matches[t.transaction_id])
    ][: config.CE3_MIN_PRIOR_TRANSACTIONS]

    return result


def ce3_outcome_summary(result: CE3Result) -> dict:
    """Business-facing summary of what qualification means."""
    if result.qualifies:
        return {
            "status": "QUALIFIES",
            "effect": "Dispute blocked pre-emptively; liability shifts to the issuer.",
            "vamp_impact": "Event does NOT count toward the merchant's VAMP ratio.",
            "action": "Submit CE3.0 evidence via Order Insight before the dispute is filed.",
        }
    return {
        "status": "DOES NOT QUALIFY",
        "effect": "Dispute proceeds through the standard representment path.",
        "vamp_impact": "Event WILL count toward the merchant's VAMP ratio regardless of outcome.",
        "action": "Route to the standard EV Decision Engine.",
    }
