"""
Cost Sensitivity Dashboard — shows how the system's behavior changes when
business parameters change. Drag the sliders and see the auto-contest rate,
savings, and false-positive cost respond in real time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src import config, data_gen, ev_engine
from src.backtest import score_test_set
from src.model import time_respecting_split
from src.pipeline import load_artifacts

st.set_page_config(page_title="Cost Sensitivity", layout="wide")
st.title("Cost Sensitivity Dashboard")
st.caption(
    "How do the system's decisions change when business parameters change? "
    "Drag the sliders and see the auto-contest rate, savings, and false-positive cost respond."
)


@st.cache_resource
def load_and_score():
    model, calibrator, feature_cols, history_df = load_artifacts()
    raw = data_gen.generate_disputes()
    _, _, test_raw = time_respecting_split(raw)
    # Score a subset for interactive speed.
    scored = score_test_set(test_raw.head(500), model, calibrator, feature_cols, history_df)
    return scored


try:
    base_scored = load_and_score()
except FileNotFoundError:
    st.error("No trained model found. Run `make train` first.")
    st.stop()

st.markdown("---")

col_sliders, col_results = st.columns([1, 1.5])

with col_sliders:
    st.subheader("Business Parameters")

    contest_cost = st.slider(
        "Contest cost (₹) — labor + admin fee",
        min_value=100, max_value=2000, value=int(config.CONTEST_COST_INR), step=50,
        help="Total cost to prepare and submit a contest: analyst labor + card network admin fee",
    )

    hard_ceiling = st.slider(
        "Hard ceiling (₹) — max amount for auto-contest",
        min_value=5000, max_value=100000, value=int(config.HARD_CEILING_INR), step=5000,
        help="Disputes above this amount ALWAYS escalate to a human, regardless of win probability",
    )

    min_evidence = st.slider(
        "Minimum evidence completeness for auto-contest",
        min_value=0.0, max_value=1.0, value=config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO, step=0.1,
        help="Evidence completeness threshold below which the system escalates regardless of EV",
    )

    analyst_labor = st.slider(
        "Analyst labor cost per case (₹)",
        min_value=50, max_value=500, value=int(config.ANALYST_LABOR_COST_PER_CASE_INR), step=25,
        help="Cost for a human analyst to manually prepare a contest evidence packet",
    )

with col_results:
    st.subheader("Impact")

    re_decided = []
    for _, row in base_scored.iterrows():
        decision = ev_engine.decide(
            win_prob=row["win_probability"],
            amount=row["amount_inr"],
            evidence_completeness=row["evidence_completeness"],
            contest_cost=contest_cost,
            ceiling=hard_ceiling,
            min_completeness=min_evidence,
        )
        re_decided.append({**row.to_dict(), "new_action": decision.action.value, "new_ev_contest": decision.ev_contest})
    re_df = pd.DataFrame(re_decided)

    new_auto = re_df[re_df["new_action"] == "AUTO_CONTEST"]
    new_escalate = re_df[re_df["new_action"] == "ESCALATE"]

    auto_rate = len(new_auto) / len(re_df) if len(re_df) > 0 else 0

    new_fp = new_auto[new_auto["actual_won"] == 0]
    new_tp = new_auto[new_auto["actual_won"] == 1]
    fp_rate = len(new_fp) / len(new_auto) if len(new_auto) > 0 else 0
    fp_cost = (new_fp["amount_inr"] + contest_cost).sum()
    tp_value = (new_tp["amount_inr"] - contest_cost).sum()
    net_auto = tp_value - fp_cost

    m1, m2 = st.columns(2)
    m1.metric("Auto-contest rate", f"{auto_rate:.0%}")
    m2.metric("Cases auto-handled", f"{len(new_auto)} / {len(re_df)}")

    m3, m4 = st.columns(2)
    m3.metric("False positive rate", f"{fp_rate:.1%}")
    m4.metric("False positive cost", f"₹{fp_cost:,.0f}")

    m5, m6 = st.columns(2)
    m5.metric("Net value (auto decisions)", f"₹{net_auto:,.0f}")
    m6.metric("Cases needing analyst", f"{len(new_escalate)}")

    st.markdown("---")

    st.markdown("**How this compares to default settings:**")
    default_auto = len(base_scored[base_scored["system_action"] == "AUTO_CONTEST"])
    default_escalate = len(base_scored[base_scored["system_action"] == "ESCALATE"])

    comparison = pd.DataFrame({
        "Metric": ["Auto-contest", "Escalate", "Auto-contest rate"],
        "Default": [default_auto, default_escalate, f"{default_auto / len(base_scored):.0%}"],
        "With your settings": [len(new_auto), len(new_escalate), f"{auto_rate:.0%}"],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**What this shows:** the system isn't brittle — it adapts to different business "
        "parameters. A merchant with expensive analyst labor benefits from a higher auto-contest "
        "rate (lower ceiling, lower evidence threshold). A merchant processing high-value "
        "transactions benefits from a higher ceiling with more human oversight."
    )
