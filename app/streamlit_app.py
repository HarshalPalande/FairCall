"""
Analyst UI — one screen: dispute in, prediction + EV math + missing
evidence + recommendation out. Run with:

    streamlit run app/streamlit_app.py

Calls the exact same src.pipeline.score_dispute() used by scripts/demo.py
and the test suite — there is one scoring code path, not a UI-specific copy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import config, decision_engine, vamp
from src.audit import verify_chain
from src.pipeline import load_artifacts, score_dispute

st.set_page_config(page_title="AI Risk Manager — Dispute Copilot", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.title("AI Risk Manager — Chargeback Evidence Copilot")
st.caption(
    f"Scoped to Visa reason code {config.REASON_CODE} (\"Merchandise / Services Not Received\") only. "
    "Trained and evaluated on synthetic data — see README for why."
)

try:
    model, calibrator, feature_cols, history_df = get_artifacts()
except FileNotFoundError:
    st.error("No trained model found. Run `make train` (or `python -m src.model`) first.")
    st.stop()

col_input, col_output = st.columns([1, 1.4])

with col_input:
    st.subheader("Dispute")
    amount = st.number_input("Amount (₹)", min_value=100.0, max_value=200_000.0, value=5_000.0, step=100.0)
    late_filing = st.checkbox("Filed late (>30 days after transaction)", value=False)
    account_age = st.slider("Customer account age at dispute (days)", 0, 700, 120)
    merchant_category = st.selectbox(
        "Merchant category", ["electronics", "fashion", "grocery", "home", "beauty", "digital_goods"]
    )
    device_type = st.selectbox("Device type", ["mobile", "desktop", "app"])
    shipping_method = st.selectbox("Shipping method", ["standard", "express", "digital_delivery"])

    st.markdown("**Evidence on file**")
    evidence_inputs = {}
    for etype in config.EVIDENCE_TYPES:
        default = etype not in ("signed_pod",)
        evidence_inputs[f"has_{etype}"] = int(
            st.checkbox(etype.replace("_", " ").title(), value=default, key=etype)
        )

    customer_id = st.selectbox(
        "Customer (for prior-history features — demo uses real synthetic customers)",
        sorted(history_df["customer_id"].unique())[:200],
    )

    run = st.button("Score dispute", type="primary")

with col_output:
    st.subheader("Result")
    if run:
        case = {
            "transaction_id": "UI-CASE",
            "customer_id": customer_id,
            "amount_inr": amount,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "shipping_method": shipping_method,
            "late_filing": int(late_filing),
            "account_age_days_at_dispute": account_age,
            **evidence_inputs,
        }
        result = score_dispute(case, model, calibrator, feature_cols, history_df)
        ev = result["ev_decision"]
        ecv = result["evidence"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Win probability", f"{result['win_probability']:.0%}")
        m2.metric("Evidence completeness", f"{ecv['completeness_score']:.0%}")
        m3.metric(
            "Decision",
            ev["action"].value if hasattr(ev["action"], "value") else str(ev["action"]).split(".")[-1],
        )

        if ev["ceiling_blocked"]:
            st.warning(f"Blocked by hard ₹ ceiling (₹{config.HARD_CEILING_INR:,.0f}) — always escalates, regardless of confidence.")
        if ev["evidence_blocked"]:
            st.warning(f"Blocked by minimum evidence completeness ({config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO:.0%}).")

        st.markdown("**EV math**")
        st.write(f"EV(contest) = ₹{ev['ev_contest']:,.2f}    EV(accept) = ₹{ev['ev_accept']:,.2f}")
        for r in ev["reasons"]:
            st.write(f"- {r}")

        vamp_status = vamp.compute_vamp_status()
        marginal_vamp = vamp.marginal_vamp_cost(vamp_status)

        st.markdown("**Portfolio (VAMP) cost**")
        st.write(
            f"This dispute adds ₹{marginal_vamp:,.2f} of VAMP portfolio risk. "
            f"Current ratio: {vamp_status.current_ratio * 100:.3f}% "
            f"(threshold {vamp_status.threshold * 100:.2f}%, "
            f"{vamp_status.headroom_events:,} disputes of headroom)."
        )
        st.caption(
            "Note: this cost applies whether the dispute is contested or accepted — the VAMP "
            "ratio counts disputes filed, not lost. Only prevention avoids it. See the VAMP Risk page."
        )

        st.markdown("---")
        st.markdown("**Four-way decision comparison**")
        st.caption(
            "The original engine chooses contest vs. escalate. Visa's pre-dispute "
            "infrastructure offers more levers, and they differ in VAMP consequence. "
            "ESCALATE is not EV-scored — it is what happens when no automated action "
            "is permitted."
        )

        four_way = decision_engine.decide_four_way(
            win_prob=result["win_probability"],
            amount=amount,
            evidence_completeness=ecv["completeness_score"],
            reason_code=config.REASON_CODE,
        )

        rows = []
        for opt in four_way.options:
            rows.append({
                "Action": opt.action.value,
                "EV (₹)": (f"{opt.expected_value_inr:,.0f}" if opt.ev_comparable
                           else f"({opt.expected_value_inr:,.0f} if accepted)"),
                "VAMP cost (₹)": f"{opt.vamp_cost_inr:,.0f}",
                "Viable": "✅" if opt.viable else "❌",
                "Why": opt.blocked_reason or opt.rationale,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if four_way.chosen == decision_engine.DecisionAction.ESCALATE:
            st.warning(f"**Chosen: `{four_way.chosen.value}`** — no automated action permitted.")
        else:
            st.success(f"**Chosen: `{four_way.chosen.value}`**")
        for r in four_way.reasons:
            st.write(f"- {r}")

        if ecv["missing_documents"]:
            st.markdown("**Missing required evidence**")
            st.write(", ".join(ecv["missing_documents"]))
        if ecv["consistency_notes"]:
            st.markdown("**Consistency warnings**")
            for n in ecv["consistency_notes"]:
                st.write(f"⚠️ {n}")

        if result["counterfactuals"]:
            st.markdown("**What evidence would help most? (counterfactual re-scoring)**")
            cf_df = pd.DataFrame(result["counterfactuals"])
            cf_df["baseline_prob"] = (cf_df["baseline_prob"] * 100).round(1)
            cf_df["what_if_prob"] = (cf_df["what_if_prob"] * 100).round(1)
            cf_df["delta_pp"] = (cf_df["delta"] * 100).round(1)
            st.dataframe(
                cf_df[["evidence_type", "baseline_prob", "what_if_prob", "delta_pp"]].rename(
                    columns={
                        "evidence_type": "Evidence type",
                        "baseline_prob": "Current win %",
                        "what_if_prob": "If added win %",
                        "delta_pp": "Δ (pp)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            top = result["counterfactuals"][0]
            st.info(
                f"Recommendation: gather **{top['evidence_type'].replace('_', ' ')}** before contesting — "
                f"raises win probability from {top['baseline_prob']:.0%} to {top['what_if_prob']:.0%}."
            )

        ok, msg = verify_chain()
        st.caption(f"Audit log chain check: {'✅ OK' if ok else '❌ FAILED'} — {msg}")
    else:
        st.write("Fill in the dispute on the left and click **Score dispute**.")

st.divider()
with st.expander("Defense-only enforcement (what this system will NOT do)"):
    st.markdown(
        "- Never auto-submits evidence without a human review gate (out of scope for this demo build — see README).\n"
        "- Never contacts the disputing customer.\n"
        "- Hard ₹ ceiling cannot be bypassed by model confidence.\n"
        "- Every scored decision is written to an append-only, hash-chained audit log."
    )
