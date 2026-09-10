"""
VAMP Portfolio Risk — the cost dimension the EV engine originally missed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src import config, vamp

st.set_page_config(page_title="VAMP Risk", layout="wide")
st.title("VAMP Portfolio Risk")
st.caption(
    "Visa Acquirer Monitoring Program — the portfolio-level cost of a dispute, "
    "which the transaction-level EV math alone does not capture."
)

st.markdown("---")

col_in, col_out = st.columns([1, 1.4])

with col_in:
    st.subheader("Merchant Portfolio")
    st.caption("Simulated values — in production these come from settled transaction data.")
    settled = st.number_input(
        "Monthly settled CNP transactions",
        min_value=1_000, max_value=5_000_000,
        value=config.MERCHANT_MONTHLY_SETTLED_TXNS, step=10_000,
    )
    disputes = st.number_input(
        "Monthly dispute events (TC40 fraud + TC15 disputes)",
        min_value=0, max_value=100_000,
        value=config.MERCHANT_MONTHLY_DISPUTE_EVENTS, step=100,
    )
    threshold_pct = st.slider(
        "VAMP Excessive threshold (%)",
        min_value=0.5, max_value=3.0,
        value=config.VAMP_EXCESSIVE_THRESHOLD * 100, step=0.1,
        help="1.5% for US/Canada/EU/APAC/LATAM since 1 April 2026. CEMEA remains 2.2%.",
    )

status = vamp.compute_vamp_status(
    monthly_settled_txns=settled,
    monthly_dispute_events=disputes,
    threshold=threshold_pct / 100,
)
marginal = vamp.marginal_vamp_cost(status)

with col_out:
    st.subheader("Standing")

    ratio_pct = status.current_ratio * 100
    threshold_pct_disp = status.threshold * 100
    gauge_max = max(threshold_pct_disp * 1.6, ratio_pct * 1.1, 0.1)
    ratio_frac = min(ratio_pct / gauge_max, 1.0) * 100
    threshold_frac = min(threshold_pct_disp / gauge_max, 1.0) * 100

    st.markdown(
        f"""
<div style="margin-top:6px;">
  <div style="position:relative; height:22px; border-radius:6px; overflow:visible;
              background:linear-gradient(90deg, #22c55e 0%, #eab308 {threshold_frac * 0.7:.1f}%,
              #ef4444 {threshold_frac:.1f}%, #7f1d1d 100%);">
    <div style="position:absolute; left:{threshold_frac:.1f}%; top:-4px; bottom:-4px;
                width:2px; background:#111; opacity:0.55;"></div>
    <div style="position:absolute; left:{ratio_frac:.1f}%; top:-6px; width:0; height:0;
                border-left:7px solid transparent; border-right:7px solid transparent;
                border-top:10px solid #111; transform:translateX(-7px);"></div>
  </div>
  <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:2px;
              opacity:0.75;">
    <span>0%</span><span>Excessive threshold {threshold_pct_disp:.2f}% →</span><span>{gauge_max:.2f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    m1.metric("Current VAMP ratio", f"{status.current_ratio * 100:.3f}%")
    m2.metric("Threshold", f"{status.threshold * 100:.2f}%")

    if status.is_excessive:
        st.error(
            f"**EXCESSIVE** — above the {status.threshold * 100:.2f}% threshold. "
            f"₹{config.VAMP_PENALTY_PER_DISPUTE_INR:,.0f} (${config.VAMP_PENALTY_PER_DISPUTE_USD:.0f}) "
            f"per disputed transaction applies, with no warning tier, plus reserve "
            f"requirements and processing termination risk."
        )
    elif not status.is_monitored:
        st.info(
            f"Below the VAMP monitoring floor ({config.VAMP_MONITORING_FLOOR_EVENTS:,} "
            f"combined events/month). Not formally monitored — though acquirers often "
            f"apply stricter internal limits."
        )
    else:
        pct_used = status.current_ratio / status.threshold
        if pct_used > 0.85:
            st.warning(
                f"**{status.headroom_events:,} disputes of headroom** before crossing "
                f"into Excessive. That is {pct_used:.0%} of the allowance consumed."
            )
        else:
            st.success(
                f"**{status.headroom_events:,} disputes of headroom.** "
                f"{pct_used:.0%} of the allowance consumed."
            )

    st.markdown("---")
    st.markdown("**Marginal cost of one additional dispute**")
    st.metric("VAMP risk cost per dispute", f"₹{marginal:,.2f}")
    st.caption(
        "Modelled as the portfolio-wide penalty exposure weighted by the probability "
        "that this dispute is the one that crosses the threshold (1 / headroom). "
        "Rises sharply as headroom shrinks."
    )

st.markdown("---")
st.subheader("Why this changes the decision framing")

st.markdown(
    """
The transaction-level EV engine asks: *will we win this dispute?*

VAMP asks a different question: *how many disputes can we afford to have at all?*

**The critical asymmetry:** the VAMP ratio counts disputes **filed**, not disputes
**lost**. A merchant who contests and wins still carries that dispute in their ratio.
Contesting protects the transaction value; it does nothing for portfolio standing.
"""
)

comparison = pd.DataFrame({
    "Outcome": [
        "Dispute filed -> contested -> WON",
        "Dispute filed -> contested -> LOST",
        "Dispute filed -> accepted",
        "Dispute PREVENTED (evidence collected proactively)",
    ],
    "Transaction ₹": ["Recovered", "Lost", "Lost", "Never at risk"],
    "Counts toward VAMP ratio?": ["YES", "YES", "YES", "NO"],
})
st.dataframe(comparison, hide_index=True, use_container_width=True)

st.info(
    "**This is why the prevention layer matters strategically, not just financially.** "
    "It is the only branch that keeps an event off the VAMP ratio. Every other path — "
    "including winning — leaves portfolio risk untouched."
)

st.markdown("---")
st.caption(
    "VAMP thresholds and penalty structure are from Visa's published program rules. "
    "The merchant portfolio figures above are simulated for demonstration. The "
    "USD-INR rate is a placeholder, disclosed as such in config.py."
)
