"""
Human Review Queue — analyst view for ESCALATE cases.
Ranked by EV(contest) descending so highest-value cases surface first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config, data_gen
from src.audit import log_decision
from src.backtest import score_test_set
from src.bootstrap import ensure_trained
from src.model import time_respecting_split
from src.pipeline import load_artifacts

st.set_page_config(page_title="Review Queue", layout="wide")
ensure_trained()
st.title("Analyst Review Queue")
st.caption("ESCALATE cases ranked by potential recovery value — highest-impact decisions surface first.")


@st.cache_resource
def get_escalated_cases():
    model, calibrator, feature_cols, history_df = load_artifacts()
    raw = data_gen.generate_disputes()
    _, _, test_raw = time_respecting_split(raw)
    # Score a subset for demo speed.
    scored = score_test_set(test_raw.head(200), model, calibrator, feature_cols, history_df)
    escalated = scored[scored["system_action"] == "ESCALATE"].copy()
    escalated = escalated.sort_values("ev_contest", ascending=False)
    return escalated


try:
    queue = get_escalated_cases()
except FileNotFoundError:
    st.error("No trained model found. Run `make train` first.")
    st.stop()

if "review_decisions" not in st.session_state:
    # transaction_id -> "CONTEST" | "ACCEPT", human decisions made THIS session.
    # Not persisted beyond it -- a real analyst tool would write these to a
    # database, not browser-tab state. What IS persisted, immediately, on
    # every decision below, is the audit log entry itself (audit_log/decisions.jsonl),
    # which is the actual record of what happened, not this in-memory dict.
    st.session_state.review_decisions = {}

resolved_count = len(st.session_state.review_decisions)
m1, m2 = st.columns(2)
m1.metric("Cases pending review", len(queue) - resolved_count)
m2.metric("Reviewed this session", resolved_count)

for i, (_, case) in enumerate(queue.head(15).iterrows()):
    txn_id = case["transaction_id"]
    decision = st.session_state.review_decisions.get(txn_id)

    label = f"{txn_id} — ₹{case['amount_inr']:,.0f} — Win prob: {case['win_probability']:.0%}"
    if decision:
        label += f"  —  reviewed: {decision}"

    with st.expander(label, expanded=(decision is None and i == 0)):
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Probability", f"{case['win_probability']:.0%}")
        c2.metric("EV(contest)", f"₹{case['ev_contest']:,.0f}")
        c3.metric("Evidence", f"{case['evidence_completeness']:.0%}")

        reasons = []
        if case["amount_inr"] > config.HARD_CEILING_INR:
            reasons.append(f"Amount ₹{case['amount_inr']:,.0f} exceeds hard ceiling ₹{config.HARD_CEILING_INR:,.0f}")
        if case["evidence_completeness"] < config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO:
            reasons.append(
                f"Evidence completeness {case['evidence_completeness']:.0%} below minimum "
                f"{config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO:.0%}"
            )
        if case["ev_contest"] <= case["ev_accept"]:
            reasons.append(f"EV(contest) ₹{case['ev_contest']:,.0f} does not exceed EV(accept) ₹{case['ev_accept']:,.0f}")

        if reasons:
            st.markdown("**Why escalated:**")
            for r in reasons:
                st.write(f"- {r}")

        st.markdown(f"**EV math:** Contest = ₹{case['ev_contest']:,.2f} vs Accept = ₹{case['ev_accept']:,.2f}")

        def _record(action, txn_id=txn_id, case=case):
            st.session_state.review_decisions[txn_id] = action
            # This is the actual effect of the click -- a real, hash-chained
            # audit entry, not a UI-only state change. It's a human decision on
            # an ESCALATE case, not the model's own call, so it's tagged
            # distinctly from the "two_way"/"four_way" engine entries elsewhere.
            log_decision({
                "transaction_id": txn_id,
                "engine": "human_review",
                "amount_inr": float(case["amount_inr"]),
                "win_probability": float(case["win_probability"]),
                "evidence_completeness": float(case["evidence_completeness"]),
                "ev_contest": float(case["ev_contest"]),
                "ev_accept": float(case["ev_accept"]),
                "analyst_decision": action,
            })

        if decision == "CONTEST":
            st.success("Analyst decision recorded: **contest** this dispute. Logged to the audit trail.")
        elif decision == "ACCEPT":
            st.info("Analyst decision recorded: **accept the loss**. Logged to the audit trail.")
        else:
            col_a, col_b = st.columns(2)
            if col_a.button("Contest", key=f"contest_{i}", type="primary"):
                _record("CONTEST")
                st.rerun()
            if col_b.button("Accept loss", key=f"accept_{i}"):
                _record("ACCEPT")
                st.rerun()
