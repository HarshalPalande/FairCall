"""
Runs the demo cases through the full pipeline and prints a walkthrough.

Each case is scored twice on purpose: once by the original two-output EV engine
(CONTEST vs ACCEPT) and once by the four-way engine that also prices DEFLECT and
AUTO_RESOLVE against the merchant's VAMP exposure. The contrast is the point --
where the two disagree is where portfolio risk changes the answer.

Requires `make train` (or `python -m src.model`) to have been run first, so that
models/ and data/history_train.csv exist.

    python -m scripts.demo
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, decision_engine, vamp
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
    "4. The RDR moment: we'd probably WIN at representment, but refunding is cheaper once VAMP is priced in": {
        "transaction_id": "DEMO-004",
        "customer_id": "CUST002701",  # real customer from data/disputes.csv; evidence set to full
        # here (not the row's actual flags) so the case isolates the VAMP effect from
        # evidence-completeness noise -- the win probability itself is the model's,
        # not hand-picked.
        "amount_inr": 3_101.81,
        "merchant_category": "electronics",
        "device_type": "desktop",
        "shipping_method": "standard",
        "late_filing": 1,
        "account_age_days_at_dispute": 5,
        "has_tracking_number": 1,
        "has_delivery_confirmation": 1,
        "has_signed_pod": 1,
        "has_courier_communication": 1,
        "has_avs_match": 1,
    },
}



def print_four_way(decision, two_way_action):
    """Print the four-way engine's pricing next to what the two-way engine said.

    Prints every option, not just the winner, because the argument this project
    makes is about the comparison -- a reader has to see DEFLECT and AUTO_RESOLVE
    being priced to believe the choice between them means anything.
    """
    print("  --- four-way engine (DEFLECT / AUTO_RESOLVE / CONTEST / ESCALATE) ---")
    for o in decision.options:
        mark = ">>" if o.action == decision.chosen else "  "
        if o.ev_comparable:
            ev = f"EV ₹{o.expected_value_inr:>12,.2f}"
        else:
            ev = f"   (not EV-scored)      "
        status = "" if o.viable else "  BLOCKED"
        print(f"  {mark} {o.action.value:<13} {ev}   VAMP ₹{o.vamp_cost_inr:>9,.2f}{status}")
        if not o.viable and o.blocked_reason:
            print(f"       -> {o.blocked_reason}")
    print(f"  Chosen: {decision.chosen.value}")
    for r in decision.reasons:
        print(f"    {r}")
    if decision.chosen.value != two_way_action:
        print(
            f"  >> The two-way engine said {two_way_action}. Pricing VAMP exposure and the "
            f"pre-dispute levers changes the answer to {decision.chosen.value}."
        )


def main():
    model, calibrator, feature_cols, history_df = load_artifacts()

    vamp_status = vamp.compute_vamp_status()
    print("=" * 100)
    print("PORTFOLIO CONTEXT (the same numbers the four-way engine prices against)")
    print("-" * 100)
    print(f"Monthly settled CNP transactions : {config.MERCHANT_MONTHLY_SETTLED_TXNS:,}")
    print(f"Monthly dispute events           : {config.MERCHANT_MONTHLY_DISPUTE_EVENTS:,}")
    print(f"VAMP ratio                       : {vamp_status.current_ratio:.3%}  "
          f"(Excessive threshold {config.VAMP_EXCESSIVE_THRESHOLD:.2%})")
    print(f"Headroom before Excessive        : {vamp_status.headroom_events:,} dispute events")
    print(f"Marginal cost of ONE more dispute: ₹{vamp.marginal_vamp_cost(vamp_status):,.2f}")
    print()

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

        decision = decision_engine.decide_four_way(
            win_prob=result["win_probability"],
            amount=case["amount_inr"],
            evidence_completeness=ecv["completeness_score"],
            reason_code=config.REASON_CODE,
            vamp_status=vamp_status,
        )
        print_four_way(decision, ev["action"].split(".")[-1])
        print()

    ok, msg = __import__("src.audit", fromlist=["verify_chain"]).verify_chain()
    print(f"Audit log chain check: {'OK' if ok else 'FAILED'} — {msg}")
    print(f"Audit log location: {config.AUDIT_LOG_PATH}")


if __name__ == "__main__":
    main()
