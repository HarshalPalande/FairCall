"""
RDR Rule Optimizer — what amount threshold should the merchant configure for
automatic pre-dispute resolution?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config, data_gen, rdr_optimizer, vamp
from src.backtest import score_test_set
from src.model import time_respecting_split
from src.pipeline import load_artifacts

st.set_page_config(page_title="RDR Optimizer", layout="wide")
st.title("RDR Rule Optimizer")
st.caption(
    "Visa RDR auto-refunds disputes matching merchant-configured rules before a "
    "chargeback posts. For non-fraud reason codes the event is excluded from the "
    "VAMP ratio. This finds the amount threshold that minimises total cost."
)

st.info(
    "**The trade-off:** auto-resolving more disputes costs more in refunds but buys "
    "back VAMP headroom. Auto-resolving fewer preserves revenue but risks crossing the "
    "Excessive threshold — which applies penalties to the merchant's entire dispute "
    "volume, not just the marginal case."
)


@st.cache_resource
def get_scored():
    model, calibrator, feature_cols, history_df = load_artifacts()
    raw = data_gen.generate_disputes()
    _, _, test_raw = time_respecting_split(raw)
    return score_test_set(test_raw.head(400), model, calibrator, feature_cols, history_df)


try:
    scored = get_scored()
except FileNotFoundError:
    st.error("No trained model found. Run `make train` (or `make demo`) first.")
    st.stop()
except Exception as e:  # pragma: no cover - UI guard
    st.error(f"Could not load scored disputes: {e}")
    st.stop()

st.markdown("---")

col_cfg, col_res = st.columns([1, 1.6])

with col_cfg:
    st.subheader("Portfolio Context")
    settled = st.number_input(
        "Monthly settled CNP transactions", min_value=1_000, max_value=5_000_000,
        value=config.MERCHANT_MONTHLY_SETTLED_TXNS, step=10_000,
    )
    disputes = st.number_input(
        "Monthly dispute events", min_value=0, max_value=100_000,
        value=config.MERCHANT_MONTHLY_DISPUTE_EVENTS, step=100,
    )
    reason_code = st.selectbox(
        "Reason code",
        [config.REASON_CODE, "10.4"],
        help="Non-fraud codes (13.x) are excluded from VAMP when RDR-resolved. "
             "Fraud code 10.4 is NOT — it counts even when refunded.",
    )

    vamp_status = vamp.compute_vamp_status(
        monthly_settled_txns=settled, monthly_dispute_events=disputes
    )

    st.metric("Current VAMP ratio", f"{vamp_status.current_ratio * 100:.3f}%")
    st.metric("Headroom", f"{vamp_status.headroom_events:,} disputes")

    if reason_code in config.RDR_VAMP_EXCLUDED_REASON_CODES:
        st.success("✅ RDR resolutions for this code are excluded from the VAMP ratio.")
    else:
        st.warning(
            "⚠️ Fraud code — RDR resolutions still count toward VAMP. The optimizer "
            "reflects that RDR is a much weaker lever here."
        )

curve = rdr_optimizer.optimize_rdr_threshold(
    scored, reason_code=reason_code, vamp_status=vamp_status
)
best = rdr_optimizer.best_threshold(curve)

with col_res:
    st.subheader("Recommended Rule")

    if not best:
        st.warning("No data to optimize over.")
    else:
        st.success(
            f"**Auto-resolve disputes at or below ₹{best['amount_threshold_inr']:,.0f}**"
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Disputes auto-resolved", f"{int(best['disputes_auto_resolved'])}",
                  f"{best['pct_auto_resolved'] * 100:.0f}% of volume")
        r2.metric("Refund outlay", f"₹{best['refund_outlay_inr']:,.0f}")
        r3.metric("Total cost", f"₹{best['total_cost_inr']:,.0f}")

        st.markdown("---")
        st.markdown("**VAMP impact of this rule**")
        v1, v2 = st.columns(2)
        v1.metric("VAMP ratio", f"{best['vamp_ratio_after_pct']:.3f}%",
                  f"{best['vamp_ratio_after_pct'] - best['vamp_ratio_before_pct']:+.3f} pp")
        v2.metric("Headroom", f"{int(best['vamp_headroom_after']):,}",
                  f"{int(best['vamp_headroom_after'] - best['vamp_headroom_before']):+,}")

        if best["crosses_threshold_before"] and not best["crosses_threshold_after"]:
            st.success("🎯 This rule brings the merchant back UNDER the Excessive threshold.")

st.markdown("---")
st.subheader("Cost Curve")
st.caption(
    "Total cost across candidate thresholds. The full curve is shown rather than just "
    "the optimum, so the trade-off is inspectable."
)

if not curve.empty:
    st.line_chart(
        curve[["amount_threshold_inr", "total_cost_inr"]].set_index("amount_threshold_inr")
    )

    with st.expander("Full sweep data"):
        display = curve[[
            "amount_threshold_inr", "disputes_auto_resolved", "pct_auto_resolved",
            "refund_outlay_inr", "total_cost_inr",
            "vamp_ratio_after_pct", "vamp_headroom_after",
        ]].copy()
        display.columns = [
            "Threshold ₹", "Auto-resolved", "% volume", "Refund outlay ₹",
            "Total cost ₹", "VAMP ratio %", "Headroom",
        ]
        # pct_auto_resolved is a fraction (0-1); the column header says "%", and the
        # metric above reports it as one. Scale here so the table agrees with its own
        # label and with the headline figure instead of reading 100x low.
        display["% volume"] = display["% volume"] * 100
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Threshold ₹": st.column_config.NumberColumn(format="%.0f"),
                "Auto-resolved": st.column_config.NumberColumn(format="%d"),
                "% volume": st.column_config.NumberColumn(format="%.1f%%"),
                "Refund outlay ₹": st.column_config.NumberColumn(format="%.0f"),
                "Total cost ₹": st.column_config.NumberColumn(format="%.0f"),
                "VAMP ratio %": st.column_config.NumberColumn(format="%.3f"),
                "Headroom": st.column_config.NumberColumn(format="%d"),
            },
        )

st.divider()
st.warning(
    f"**The sweep is capped at the hard ceiling (₹{config.HARD_CEILING_INR:,.0f}).** "
    "RDR is an automated money movement, so it is bound by the same "
    "no-large-automated-decision-without-a-human rule as every other automated action "
    "in the decision engine. Recommending a higher threshold would recommend something "
    "the engine will refuse to execute."
)
st.caption(
    "Escalated disputes are priced here as accepted losses — a deliberately pessimistic "
    "floor, not a prediction, since a human analyst would contest some of them. That "
    "makes the remaining-slice cost an upper bound and biases the optimizer mildly "
    "toward auto-resolving. src/backtest.py models analyst behaviour explicitly; this "
    "page does not."
)
st.caption(
    "RDR fee, Order Insight cost, deflection rate and the merchant portfolio figures are "
    "PLACEHOLDER assumptions, disclosed as such in config.py and the README. The VAMP "
    "thresholds and the fraud/non-fraud exclusion asymmetry are from Visa's published "
    "program rules."
)
