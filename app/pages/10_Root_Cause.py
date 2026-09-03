"""
Root Cause Intelligence — what should this merchant CHANGE?

Every other page here is predictive or decisive. This one is prescriptive:
it looks across the whole dispute population and ranks what to fix, by money.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src import config, data_gen, root_cause
from src.pipeline import load_artifacts

st.set_page_config(page_title="Root Cause Intelligence", layout="wide")
st.title("Root Cause Intelligence")
st.caption(
    "Not 'will we win this dispute?' — 'what should this merchant change?' "
    "Ranked by rupees lost, because a remediation budget is finite and the "
    "worst win rate is usually not the biggest bill."
)


@st.cache_data(show_spinner="Loading dispute population...")
def load_disputes():
    return data_gen.generate_disputes()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Re-scoring counterfactuals against the calibrated model...")
def get_counterfactual_value(_model, _calibrator, feature_cols, sample_size):
    df = load_disputes()
    return root_cause.counterfactual_evidence_value(
        df, _model, _calibrator, feature_cols, sample_size=sample_size
    )


disputes = load_disputes()

st.markdown("---")
st.subheader("1. Where the money actually goes")
st.caption(
    "Expected loss = amount × probability of losing, summed per segment. Sorted "
    "by that, not by win rate — the two give different answers, which is the point."
)

seg_col = st.selectbox(
    "Break down by", ["shipping_method", "merchant_category", "device_type"],
    help="All three are transaction attributes the merchant controls or can act on.",
)
seg = root_cause.segment_loss_analysis(disputes, seg_col)

display = seg.reset_index().rename(columns={
    seg_col: seg_col.replace("_", " ").title(),
    "disputes": "Disputes",
    "win_rate": "Win rate",
    "avg_amount_inr": "Avg amount ₹",
    "expected_loss_inr": "Expected loss ₹",
    "pct_of_total_loss": "% of total loss",
})
st.dataframe(
    display, hide_index=True, use_container_width=True,
    column_config={
        "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Avg amount ₹": st.column_config.NumberColumn(format="%.0f"),
        "Expected loss ₹": st.column_config.NumberColumn(format="%.0f"),
        "% of total loss": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

worst_rate = seg["win_rate"].idxmin()
biggest_loss = seg["expected_loss_inr"].idxmax()
if worst_rate != biggest_loss:
    st.info(
        f"**These disagree, and that's the useful part.** `{worst_rate}` has the worst "
        f"win rate ({seg.loc[worst_rate, 'win_rate']:.1%}), but `{biggest_loss}` costs "
        f"more money (₹{seg.loc[biggest_loss, 'expected_loss_inr']:,.0f}, "
        f"{seg.loc[biggest_loss, 'pct_of_total_loss']:.0%} of all losses) purely on volume. "
        f"A merchant fixing things in win-rate order would fix `{worst_rate}` first and "
        f"leave most of the money on the table."
    )
else:
    st.info(
        f"`{biggest_loss}` is both the worst win rate and the biggest bill "
        f"(₹{seg.loc[biggest_loss, 'expected_loss_inr']:,.0f}) — fix it first."
    )

st.markdown("---")
st.subheader("2. Evidence gaps — observed vs. what the model says they're worth")

st.caption(
    "Two estimates side by side, deliberately not blended. The observational gap is "
    "what the population shows; the counterfactual is what the calibrated model "
    "predicts when the same dispute is re-scored with only that flag flipped."
)

gap = root_cause.evidence_gap_analysis(disputes)

model, calibrator, feature_cols, history_df = get_artifacts()
sample_size = st.slider(
    "Counterfactual sample size", min_value=100, max_value=800, value=300, step=100,
    help="Each sampled dispute is re-scored once per missing evidence type. "
         "Larger = steadier estimate, slower.",
)
cf = get_counterfactual_value(model, calibrator, feature_cols, sample_size)

merged = gap.merge(cf[["evidence_type", "mean_predicted_lift_pp", "sampled_missing"]],
                   on="evidence_type", how="left")
merged["overstatement"] = merged["observational_gap_pp"] / merged["mean_predicted_lift_pp"]

show = merged[[
    "evidence_type", "missing_addressable", "missing_structurally_impossible",
    "observational_gap_pp", "mean_predicted_lift_pp", "overstatement", "amount_at_stake_inr",
]].rename(columns={
    "evidence_type": "Evidence",
    "missing_addressable": "Missing (addressable)",
    "missing_structurally_impossible": "Missing (impossible)",
    "observational_gap_pp": "Observed gap (pp)",
    "mean_predicted_lift_pp": "Model counterfactual (pp)",
    "overstatement": "Observed ÷ model",
    "amount_at_stake_inr": "₹ at stake",
})
st.dataframe(
    show, hide_index=True, use_container_width=True,
    column_config={
        "Observed gap (pp)": st.column_config.NumberColumn(format="%.1f"),
        "Model counterfactual (pp)": st.column_config.NumberColumn(format="%.1f"),
        "Observed ÷ model": st.column_config.NumberColumn(format="%.1fx"),
        "₹ at stake": st.column_config.NumberColumn(format="%.0f"),
    },
)

worst_overstated = merged.loc[merged["overstatement"].idxmax()]
st.warning(
    f"**Why both columns are here.** For `{worst_overstated['evidence_type']}` the "
    f"observational gap ({worst_overstated['observational_gap_pp']:.1f}pp) is "
    f"**{worst_overstated['overstatement']:.1f}× larger** than the model's causal estimate "
    f"({worst_overstated['mean_predicted_lift_pp']:.1f}pp). In this dataset evidence "
    f"availability and win probability are *both* driven by the same latent customer "
    f"trust (`src/data_gen.py`: `evidence_base_rate = 0.35 + 0.5*trust`, and `win_logit` "
    f"carries `1.8*(trust-0.5)`), so the raw difference is confounded by construction. "
    f"A dashboard quoting the observational number as 'recoverable revenue' would send "
    f"this merchant to fix the wrong thing."
)

impossible_total = int(gap["missing_structurally_impossible"].sum())
if impossible_total:
    st.caption(
        f"**{impossible_total:,} 'missing evidence' rows were excluded as structurally "
        f"impossible** — digital-delivery orders have no parcel, so there is no tracking "
        f"number or signed proof of delivery to go collect. Counting them would inflate "
        f"the addressable gap with actions nobody could take."
    )

st.markdown("---")
st.subheader("3. The most actionable finding: filing late")

late = root_cause.late_filing_analysis(disputes)
l1, l2, l3 = st.columns(3)
l1.metric("Disputes filed late", f"{late['late_count']:,}", f"{late['late_share']:.0%} of all disputes")
l2.metric("Win rate when late", f"{late['win_rate_late']:.1%}",
          f"{late['gap_pp']:.1f}pp below on-time", delta_color="inverse")
l3.metric("Expected loss on late filings", f"₹{late['expected_loss_late_inr']:,.0f}")

st.success(
    f"**Unlike evidence availability, this one has no structural blocker and no "
    f"confound to argue about.** {late['late_share']:.0%} of disputes are filed late, "
    f"and late filings win {late['gap_pp']:.1f} percentage points less often. Filing "
    f"promptly is entirely a process change — no new data, no new vendor, no model. "
    f"That makes it the first thing to fix, even though it isn't the biggest single "
    f"number on this page."
)

st.markdown("---")
st.caption(
    "All figures computed from the synthetic dispute population (`src/data_gen.py`), "
    "with the counterfactual column re-scored by the same calibrated model used "
    "everywhere else in this project (`src/counterfactual.py`). These are patterns in "
    "a disclosed simulation, not measured real-world merchant behaviour — the method "
    "is the deliverable here, not the specific rupee figures."
)
