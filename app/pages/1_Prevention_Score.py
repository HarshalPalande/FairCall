"""
Dispute Prevention Score — score a transaction at PAYMENT TIME,
before any dispute exists.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src import config, prevention
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


@st.cache_data(show_spinner="Finding illustrative customers...")
def find_example_customers():
    """Pick a few REAL customers out of the actual population to make the
    cust_prior_* signal tangible, instead of an anonymous customer_id dropdown
    nobody could read anything into. Requires at least 5 prior transactions so
    the rate shown isn't just noise from 1-2 data points."""
    pool = prevention.build_transaction_dataset(seed=config.SEED)
    stats = pool.groupby("customer_id").agg(count=("became_dispute", "count"),
                                             mean=("became_dispute", "mean"),
                                             avg_amount=("amount_inr", "mean"))
    stats = stats[stats["count"] >= 5]
    risky = stats.sort_values("mean", ascending=False).head(2)
    safe = stats.sort_values("mean", ascending=True).head(2)
    examples = []
    for cust_id, row in pd.concat([risky, safe]).iterrows():
        examples.append({
            "customer_id": cust_id,
            "prior_txn_count": int(row["count"]),
            "prior_flag_rate": float(row["mean"]),
            "prior_avg_amount": float(row["avg_amount"]),
        })
    return pool, examples


try:
    prevention_model, prevention_cols = load_prevention_model()
except FileNotFoundError:
    st.error("Prevention model not found. Run `make train` first.")
    st.stop()

customer_pool, example_customers = find_example_customers()

col_in, col_out = st.columns([1, 1.4])

with col_in:
    st.subheader("Transaction Details")

    customer_options = {"— New customer (no prior history) —": None}
    for ex in example_customers:
        label = (f"{ex['customer_id']} — {ex['prior_txn_count']} prior txns, "
                 f"{ex['prior_flag_rate']:.0%} became disputes")
        customer_options[label] = ex
    customer_label = st.selectbox(
        "Customer", list(customer_options.keys()),
        help="Pick a real customer from the training population to see how their "
             "OWN prior transaction history changes this score — or leave it on "
             "'New customer' to see the no-history fallback the model uses when "
             "it doesn't know who's paying.",
    )
    chosen_customer = customer_options[customer_label]

    # Default the amount to this customer's own typical spend, not a fixed value
    # for everyone -- an amount that's a big outlier vs. a customer's own history
    # is itself a real signal the model picks up on, which would otherwise swamp
    # and confound the prior-flag-rate signal this selector exists to demonstrate.
    # Keyed on the customer selection so it resets when you switch customers, but
    # stays freely editable for a given selection.
    default_amount = round(chosen_customer["prior_avg_amount"], -2) if chosen_customer else 8_000.0
    amount = st.number_input(
        "Amount (₹)", min_value=100.0, max_value=200_000.0, value=default_amount, step=100.0,
        format="%.0f", key=f"amount_{customer_label}",
    )
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
        if chosen_customer is not None:
            txn["customer_id"] = chosen_customer["customer_id"]
            result = score_transaction(txn, prevention_model, prevention_cols, history_df=customer_pool)
            st.caption(
                f"Scored using {chosen_customer['customer_id']}'s real prior history: "
                f"{chosen_customer['prior_txn_count']} prior transactions, "
                f"{chosen_customer['prior_flag_rate']:.0%} became disputes."
            )
        else:
            result = score_transaction(txn, prevention_model, prevention_cols)
            st.caption(
                "Scored with no customer history (the model's honest fallback for an "
                "unknown/new customer) — pick a customer on the left to see how their "
                "own prior transactions change this score."
            )

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
