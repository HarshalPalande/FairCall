"""
Dispute Prevention Score — score a transaction at PAYMENT TIME,
before any dispute exists.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config
from src.prevention import score_transaction

st.set_page_config(page_title="Dispute Prevention Score", layout="wide")
st.title("Dispute Prevention Score")
st.caption(
    "Score a transaction at payment time — BEFORE any dispute exists. "
    "Flag high-risk transactions so the merchant can proactively collect evidence."
)


@st.cache_resource
def load_prevention_model():
    with open(config.MODELS_DIR / "prevention_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(config.MODELS_DIR / "prevention_feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


try:
    prevention_model, prevention_cols = load_prevention_model()
except FileNotFoundError:
    st.error("Prevention model not found. Run `make train` first.")
    st.stop()

col_in, col_out = st.columns([1, 1.4])

with col_in:
    st.subheader("Transaction Details")
    amount = st.number_input("Amount (₹)", min_value=100.0, max_value=200_000.0, value=8_000.0, step=100.0,
                              format="%.0f")  # avoids a locale-dependent decimal separator (e.g. "8000,00")
    merchant_category = st.selectbox(
        "Merchant category", ["electronics", "fashion", "grocery", "home", "beauty", "digital_goods"]
    )
    device_type = st.selectbox("Device type", ["mobile", "desktop", "app"])
    shipping_method = st.selectbox("Shipping method", ["standard", "express", "digital_delivery"])
    score_btn = st.button("Score transaction risk", type="primary")

with col_out:
    st.subheader("Risk Assessment")
    if score_btn:
        txn = {
            "amount_inr": amount,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "shipping_method": shipping_method,
        }
        result = score_transaction(txn, prevention_model, prevention_cols)

        tier = result["risk_tier"]
        score = result["dispute_risk_score"]

        m1, m2 = st.columns(2)
        m1.metric("Dispute Risk Score", f"{score:.1%}")

        if tier == "HIGH":
            m2.error(f"{tier} RISK")
        elif tier == "MEDIUM":
            m2.warning(f"{tier} RISK")
        else:
            m2.success(f"{tier} RISK")

        st.markdown("---")
        st.markdown(f"**Recommended action:** {result['recommendation']}")

        if tier in ("HIGH", "MEDIUM"):
            st.markdown("**Evidence to collect proactively:**")
            st.markdown("1. **Tracking number** — get this from the courier immediately after shipping")
            st.markdown("2. **Delivery confirmation** — request automated delivery notification")
            if shipping_method != "digital_delivery":
                st.markdown("3. **Signed proof of delivery** — request for high-value shipments")

            st.info(
                "If a dispute arrives later with this evidence already on file, "
                "the reactive system's win probability jumps significantly. Compare by scoring "
                "the same case on the Chargeback Copilot page with all evidence checked."
            )
        else:
            st.success("This transaction has low dispute risk. Standard monitoring is sufficient.")

        st.markdown("---")
        st.markdown("**Why this matters:**")
        st.markdown(
            f"If this ₹{amount:,.0f} transaction becomes a dispute with NO evidence on file, "
            f"the expected loss is ₹{amount:,.0f}. With proactive evidence collection, "
            f"the win probability typically improves substantially — turning a ₹{amount:,.0f} loss "
            f"into a ₹{amount - config.CONTEST_COST_INR:,.0f} recovery if the dispute is won."
        )
    else:
        st.write("Enter transaction details and click **Score transaction risk**.")

st.divider()
st.caption(
    "The prevention model uses ONLY features available at transaction time — "
    "no evidence flags, no information about THIS dispute (it hasn't happened yet). "
    "It does use this customer's own PRIOR transaction history (their past dispute "
    "rate) — that's genuinely known at payment time, not a leak. Lower metrics than "
    "the outcome model (see artifacts/prevention_metrics.json) are expected and "
    "honest, not a bug."
)
