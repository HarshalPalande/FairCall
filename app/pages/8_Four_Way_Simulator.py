"""
Four-Way Policy Simulator — play with the decision surface directly.

Every other page in this app scores a real dispute record. This one doesn't.
It calls src.ev_engine.decide() (the original two-way engine) and
src.decision_engine.decide_four_way() (DEFLECT / AUTO_RESOLVE / CONTEST / ESCALATE)
on whatever win probability, amount, evidence level and portfolio state you type
in — live, on every rerun. The point is to make the four-way engine's central claim
something you can personally break rather than something you're told about: that
pricing VAMP exposure can flip the answer even when the two-way engine is confident.

Opens already on a case where the two engines disagree, so the disagreement is the
first thing you see, not something you have to find.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config, decision_engine, ev_engine, vamp

st.set_page_config(page_title="Four-Way Simulator", layout="wide")
st.title("Four-Way Policy Simulator")
st.caption(
    "Move any input. Both engines re-run live: the original two-way engine "
    "(CONTEST vs. ESCALATE) and the four-way engine (DEFLECT / AUTO_RESOLVE / "
    "CONTEST / ESCALATE, priced against VAMP). Where they disagree is the whole "
    "point of the four-way engine."
)
st.markdown("---")

col_dispute, col_portfolio = st.columns([1, 1])

with col_dispute:
    st.subheader("This dispute")
    amount = st.number_input(
        "Amount (₹)", min_value=100.0, max_value=200_000.0,
        value=3_100.0, step=100.0,  # whole-rupee default -- a fractional default renders
        # with a locale-dependent decimal separator (e.g. "3101,81") on some browsers
    )
    win_prob_pct = st.slider(
        "Win probability at representment (%)",
        min_value=0, max_value=100, value=69,
        help="What the calibrated model says P(win the contest) is, for this dispute.",
    )
    evidence_pct = st.slider(
        "Evidence completeness (%)",
        min_value=0, max_value=100, value=100,
        help=f"Gates CONTEST only, at {config.MIN_EVIDENCE_COMPLETENESS_FOR_AUTO:.0%}. "
             "Does NOT gate AUTO_RESOLVE — refunding is the conservative action.",
    )
    reason_code = st.selectbox(
        "Reason code",
        options=[config.REASON_CODE, config.CE3_REASON_CODE],
        format_func=lambda rc: (
            f"{rc} — Merchandise not received (non-fraud, RDR excluded from VAMP)"
            if rc == config.REASON_CODE else
            f"{rc} — Fraud, card-absent (RDR still counts toward VAMP)"
        ),
    )
    has_order_data = st.checkbox(
        "Rich order data available (required for DEFLECT / Order Insight)",
        value=True,
    )

with col_portfolio:
    st.subheader("Portfolio state")
    st.caption("What the merchant's VAMP standing looks like this month.")
    settled = st.number_input(
        "Monthly settled CNP transactions",
        min_value=1_000, max_value=5_000_000,
        value=config.MERCHANT_MONTHLY_SETTLED_TXNS, step=10_000,
    )
    dispute_events = st.slider(
        "Monthly dispute events",
        min_value=0, max_value=4_000,
        value=config.MERCHANT_MONTHLY_DISPUTE_EVENTS, step=25,
        help="The same dispute, same win probability, can get a different decision "
             "purely because portfolio headroom changed — try dragging this.",
    )

    vamp_status = vamp.compute_vamp_status(
        monthly_settled_txns=settled, monthly_dispute_events=dispute_events,
    )
    marginal_vamp = vamp.marginal_vamp_cost(vamp_status)

    ratio_pct = vamp_status.current_ratio * 100
    threshold_pct = vamp_status.threshold * 100
    gauge_max = max(threshold_pct * 1.6, ratio_pct * 1.1, 0.1)
    ratio_frac = min(ratio_pct / gauge_max, 1.0) * 100
    threshold_frac = min(threshold_pct / gauge_max, 1.0) * 100

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
    <span>0%</span><span>Excessive threshold {threshold_pct:.2f}% →</span><span>{gauge_max:.2f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("VAMP ratio", f"{ratio_pct:.3f}%")
    g2.metric("Headroom", f"{vamp_status.headroom_events:,}")
    g3.metric("Marginal VAMP cost", f"₹{marginal_vamp:,.0f}")
    if vamp_status.is_excessive:
        st.error("Portfolio is EXCESSIVE — flat per-dispute penalty applies regardless of this case.")

win_prob = win_prob_pct / 100
evidence_completeness = evidence_pct / 100

two_way = ev_engine.decide(win_prob, amount, evidence_completeness)
four_way = decision_engine.decide_four_way(
    win_prob=win_prob,
    amount=amount,
    evidence_completeness=evidence_completeness,
    reason_code=reason_code,
    has_order_data=has_order_data,
    vamp_status=vamp_status,
)

st.markdown("---")

two_way_label = two_way.action.value
four_way_label = four_way.chosen.value

if two_way_label != four_way_label:
    st.warning(
        f"### Two-way engine: `{two_way_label}`   →   Four-way engine: `{four_way_label}`\n"
        f"Pricing VAMP exposure and the pre-dispute levers changes the answer. "
        f"This is the case the two-output engine cannot express."
    )
else:
    st.success(f"### Both engines agree: `{four_way_label}`")

if (four_way.chosen == decision_engine.DecisionAction.AUTO_RESOLVE
        and win_prob > 0.5
        and two_way.action == ev_engine.Action.AUTO_CONTEST):
    st.info(
        f"**The RDR moment:** at {win_prob_pct}% win probability, representment would "
        f"probably WIN. The four-way engine refunds anyway — once the ₹{marginal_vamp:,.0f} "
        f"marginal VAMP cost is priced onto CONTEST but not onto AUTO_RESOLVE, refunding "
        f"scores higher. Contesting wins the fight and loses the portfolio."
    )

st.markdown("#### Every option, priced")

opts = four_way.options
ev_values = [o.expected_value_inr for o in opts if o.ev_comparable]
max_abs = max([abs(v) for v in ev_values] + [1.0])

_, c2_axis, _ = st.columns([1.3, 3, 1.2])
with c2_axis:
    st.markdown(
        '<div style="position:relative; height:16px;">'
        '<div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:#94a3b8;"></div>'
        '<div style="position:absolute; left:50%; top:0; font-size:11px; color:#94a3b8; '
        'transform:translateX(-50%);">₹0</div></div>',
        unsafe_allow_html=True,
    )

for o in opts:
    is_chosen = o.action == four_way.chosen
    if o.ev_comparable:
        # frac maxes at 100 (not 50): it fills the WHOLE half-row it's drawn in,
        # not half of it -- using 50 here was a bug that capped every bar at a
        # quarter of the row's width even at maximum magnitude.
        frac = min(abs(o.expected_value_inr) / max_abs, 1.0) * 100
        is_neg = o.expected_value_inr < 0
        bar_color = "#94a3b8" if not o.viable else ("#ef4444" if is_neg else "#22c55e")
        edge = "1px solid #111" if is_chosen else "none"
        if is_neg:
            bar_html = (
                f'<div style="display:flex; width:100%; height:22px;">'
                f'<div style="flex:1; display:flex; justify-content:flex-end;">'
                f'<div style="width:{frac}%; background:{bar_color}; opacity:{1.0 if o.viable else 0.35}; '
                f'border-radius:3px 0 0 3px; border:{edge};"></div></div>'
                f'<div style="flex:1;"></div></div>'
            )
        else:
            bar_html = (
                f'<div style="display:flex; width:100%; height:22px;">'
                f'<div style="flex:1;"></div>'
                f'<div style="flex:1; display:flex;">'
                f'<div style="width:{frac}%; background:{bar_color}; opacity:{1.0 if o.viable else 0.35}; '
                f'border-radius:0 3px 3px 0; border:{edge};"></div></div></div>'
            )
        ev_text = f"₹{o.expected_value_inr:,.0f}"
    else:
        bar_html = '<div style="height:22px;"></div>'
        ev_text = "not EV-scored"

    status_text = "  BLOCKED" if not o.viable else ""
    if is_chosen:
        label = (
            f'<span style="background:#dcfce7; color:#166534; font-weight:700; '
            f'padding:2px 8px; border-radius:4px;">{o.action.value} ◀ CHOSEN</span>'
        )
    else:
        label = f"<b>{o.action.value}</b>{status_text}"

    c1, c2, c3 = st.columns([1.3, 3, 1.2])
    with c1:
        st.markdown(label, unsafe_allow_html=True)
    with c2:
        st.markdown(bar_html, unsafe_allow_html=True)
    with c3:
        st.markdown(f"{ev_text}  ·  VAMP ₹{o.vamp_cost_inr:,.0f}")
    st.caption(o.blocked_reason or o.rationale)

st.markdown("---")
st.markdown("**Why the four-way engine chose what it chose:**")
for r in four_way.reasons:
    st.write(f"- {r}")

st.markdown("---")
st.caption(
    "This page calls the exact same `src.ev_engine.decide()` and "
    "`src.decision_engine.decide_four_way()` used by `make demo`, the review queue, "
    "and the test suite — there is no simulator-only copy of the decision logic."
)
