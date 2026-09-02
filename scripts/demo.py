"""
Runs the 3 demo cases from the project brief through the full pipeline and
prints a walkthrough. Requires `make train` (or `python -m src.model`) to
have been run first so models/ and data/history_train.csv exist.

    python -m scripts.demo
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline import load_artifacts, score_dispute

CASES = {
    "1. Clear auto-contest (high confidence, complete evidence, low amount)": {
        "transaction_id": "DEMO-001",
        "customer_id": "CUST000835",  # high-trust customer: 10/11 prior disputes won
        "amount_inr": 1_800.0,
        "merchant_category": "electronics",
        "device_type": "mobile",
        "shipping_method": "standard",
        "late_filing": 0,
        "account_age_days_at_dispute": 240,
        "has_tracking_number": 1,
        "has_delivery_confirmation": 1,
        "has_signed_pod": 1,
        "has_courier_communication": 1,
        "has_avs_match": 1,
    },
    "2. Clear escalate (low confidence / missing evidence)": {
        "transaction_id": "DEMO-002",
        "customer_id": "CUST000099",
        "amount_inr": 4_500.0,
        "merchant_category": "fashion",
        "device_type": "mobile",
        "shipping_method": "standard",
        "late_filing": 1,
        "account_age_days_at_dispute": 12,
        "has_tracking_number": 0,
        "has_delivery_confirmation": 0,
        "has_signed_pod": 0,
        "has_courier_communication": 0,
        "has_avs_match": 0,
    },
    "3. The moment that matters: high amount -> hard ceiling overrides favorable EV math, where a naive fixed-probability threshold would auto-act": {
        "transaction_id": "DEMO-003",
        "customer_id": "CUST000250",
        "amount_inr": 42_000.0,
        "merchant_category": "electronics",
        "device_type": "desktop",
        "shipping_method": "express",
        "late_filing": 0,
        "account_age_days_at_dispute": 180,
        "has_tracking_number": 1,
        "has_delivery_confirmation": 1,
        "has_signed_pod": 0,
        "has_courier_communication": 1,
        "has_avs_match": 1,
    },
}


def main():
    model, calibrator, feature_cols, history_df = load_artifacts()

    for title, case in CASES.items():
        result = score_dispute(case, model, calibrator, feature_cols, history_df)
        print("=" * 100)
        print(title)
        print("-" * 100)
        print(f"Win probability (calibrated): {result['win_probability']:.2%}")
        ev = result["ev_decision"]
        print(f"EV(contest) = ₹{ev['ev_contest']:,.2f}   EV(accept) = ₹{ev['ev_accept']:,.2f}")
        print(f"Decision: {ev['action']}   (ceiling_blocked={ev['ceiling_blocked']}, evidence_blocked={ev['evidence_blocked']})")
        for r in ev["reasons"]:
            print(f"  reason: {r}")
        naive_action = "AUTO_CONTEST" if result["win_probability"] >= 0.5 else "ESCALATE"
        if naive_action != ev["action"].split(".")[-1]:
            print(
                f"  >> DIVERGES from a naive fixed-threshold system (win_prob >= 50% => contest), "
                f"which would have chosen {naive_action} here."
            )
        ecv = result["evidence"]
        print(f"Evidence completeness: {ecv['completeness_score']:.0%}  missing: {ecv['missing_documents'] or 'none'}")
        if result["counterfactuals"]:
            top = result["counterfactuals"][0]
            print(
                f"Top counterfactual: adding '{top['evidence_type']}' -> "
                f"{top['baseline_prob']:.0%} -> {top['what_if_prob']:.0%} "
                f"(+{top['delta']:.0%})"
            )
        print()

    ok, msg = __import__("src.audit", fromlist=["verify_chain"]).verify_chain()
    print(f"Audit log chain check: {'OK' if ok else 'FAILED'} — {msg}")
    print(f"Audit log location: {config.AUDIT_LOG_PATH}")


if __name__ == "__main__":
    main()
