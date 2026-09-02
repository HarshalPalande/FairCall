"""
Human Review Queue — analyst view for ESCALATE cases.
Ranked by EV(contest) descending so highest-value cases surface first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config, data_gen
from src.backtest import score_test_set
from src.model import time_respecting_split
from src.pipeline import load_artifacts

st.set_page_config(page_title="Review Queue", layout="wide")
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

st.metric("Cases pending review", len(queue))

for i, (_, case) in enumerate(queue.head(15).iterrows()):
    with st.expander(
        f"{case['transaction_id']} — ₹{case['amount_inr']:,.0f} — "
        f"Win prob: {case['win_probability']:.0%}",
        expanded=(i == 0),
    ):
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

        col_a, col_b = st.columns(2)
        col_a.button("Contest", key=f"contest_{i}", type="primary")
        col_b.button("Accept loss", key=f"accept_{i}")
