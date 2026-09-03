"""
Navigation router + custom sidebar.

Two constraints shape the odd-looking structure here:

1. `st.navigation`'s built-in sections render as flat headers — they can't
   collapse. So the built-in nav is hidden (`position="hidden"`) and the
   sidebar is rebuilt from real components: `st.page_link` for each entry and
   `st.expander` for the collapsible "Advanced" group (the expander is what
   provides the chevron and the actual open/close behaviour).

2. The nav is rendered AFTER `pg.run()`, which looks backwards but isn't.
   Every page file calls its own `st.set_page_config()`, and Streamlit allows
   exactly one per run, which must precede any delta. Drawing the sidebar
   first would emit deltas and break `set_page_config` on all ten pages.
   Running the page first, then adding sidebar content, satisfies the guard —
   and because this is the sidebar, call order doesn't affect where it lands.
   (Verified no page writes to the sidebar itself, so nothing competes.)

`st.navigation` itself is safe to call before `pg.run()`: it enqueues a
`navigation` ForwardMsg, not a delta, so it doesn't consume the
one-set_page_config-per-run budget.
"""
import streamlit as st

# --- what a first-time viewer should see first ----------------------------
demo_mode = st.Page(
    "pages/9_Demo_Mode.py", title="Demo Mode", url_path="Demo_Mode", default=True,
)
root_cause = st.Page(
    "pages/10_Root_Cause.py", title="Root Cause", url_path="Root_Cause",
)

# --- everything else ------------------------------------------------------
dispute_copilot = st.Page(
    "pages/0_Dispute_Copilot.py", title="Dispute Copilot", url_path="Dispute_Copilot",
)
four_way = st.Page(
    "pages/8_Four_Way_Simulator.py", title="Four-Way Simulator", url_path="Four_Way_Simulator",
)
prevention = st.Page(
    "pages/1_Prevention_Score.py", title="Prevention Score", url_path="Prevention_Score",
)
review_queue = st.Page(
    "pages/2_Review_Queue.py", title="Review Queue", url_path="Review_Queue",
)
rdr = st.Page(
    "pages/7_RDR_Optimizer.py", title="RDR Optimizer", url_path="RDR_Optimizer",
)
vamp = st.Page(
    "pages/5_VAMP_Risk.py", title="VAMP Risk", url_path="VAMP_Risk",
)
ce3 = st.Page(
    "pages/6_CE3_Qualification.py", title="CE3.0 Qualification", url_path="CE3_Qualification",
)
cost = st.Page(
    "pages/4_Cost_Sensitivity.py", title="Cost Sensitivity", url_path="Cost_Sensitivity",
)
razorpay = st.Page(
    "pages/3_Razorpay_Integration.py", title="Razorpay Integration", url_path="Razorpay_Integration",
)

PRIMARY = [demo_mode, root_cause]
ADVANCED = [
    dispute_copilot, four_way, prevention, review_queue,
    rdr, vamp, ce3, cost, razorpay,
]

pg = st.navigation(PRIMARY + ADVANCED, position="hidden")
pg.run()

# --- sidebar, drawn after the page for the set_page_config reason above ----
SIDEBAR_CSS = """
<style>
/* Font is deliberately NOT set anywhere here — everything inherits the app's
   existing typography. Only spacing, colour and layout are touched. */

section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
    padding: 6px 12px;
    border-radius: 8px;
    margin: 1px 0;
}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {
    font-size: 0.92rem;
    margin: 0;
}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
    background: rgba(120, 120, 120, 0.10);
}

/* "Advanced" group: strip the expander's card chrome so it reads as a nav
   section rather than a boxed widget. */
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    padding: 6px 12px !important;
    border-radius: 8px;
    font-weight: 600;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: rgba(120, 120, 120, 0.10);
}

/* Children: indented and hung off a vertical connector line, matching the
   tree look — muted until hovered or active. */
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding-left: 0 !important;
    margin-left: 20px;
    border-left: 1px solid rgba(130, 130, 130, 0.28);
}
section[data-testid="stSidebar"] [data-testid="stExpander"]
    [data-testid="stPageLink-NavLink"] {
    margin-left: 6px;
    opacity: 0.72;
}
section[data-testid="stSidebar"] [data-testid="stExpander"]
    [data-testid="stPageLink-NavLink"]:hover {
    opacity: 1;
}
</style>
"""

with st.sidebar:
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    for page in PRIMARY:
        st.page_link(page)
    # Open by default on every load — the chevron still closes it for the
    # current view. Streamlit expanders reset to this default on each rerun,
    # so navigating to another page reopens it, which is the intent here.
    with st.expander("Advanced", expanded=True):
        for page in ADVANCED:
            st.page_link(page)
