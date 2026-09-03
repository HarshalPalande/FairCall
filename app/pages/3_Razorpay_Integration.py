"""
Razorpay Integration Demo — shows the scoring pipeline working with
Razorpay payment data. Demonstrates this was built for Razorpay's
ecosystem, not just as a generic buildathon project.
"""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config
from src.pipeline import load_artifacts, score_dispute
from src.prevention import score_transaction

st.set_page_config(page_title="Razorpay Integration", layout="wide")
st.title("Razorpay Payment -> Dispute Risk Pipeline")
st.caption(
    "Demonstrates end-to-end flow: Razorpay payment -> prevention score -> "
    "dispute scoring. Shows how this system plugs into Razorpay's payment ecosystem."
)

has_credentials = bool(os.environ.get("RAZORPAY_KEY_ID")) and bool(os.environ.get("RAZORPAY_KEY_SECRET"))


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_prevention_model():
    with open(config.MODELS_DIR / "prevention_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(config.MODELS_DIR / "prevention_feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


try:
    model, calibrator, feature_cols, history_df = get_artifacts()
    prevention_model, prevention_cols = get_prevention_model()
except FileNotFoundError:
    st.error("Models not found. Run `make train` first.")
    st.stop()

if has_credentials:
    st.success("Razorpay API credentials detected — live mode available")
    mode = st.radio("Mode", ["Live (fetch real payment)", "Demo (sample payments)"], horizontal=True)
else:
    st.info(
        "Running in demo mode with sample Razorpay-format payments. "
        "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET environment variables for live API access."
    )
    mode = "Demo (sample payments)"

SAMPLE_PAYMENTS = {
    "pay_QjR3k5K8xY2mNp — ₹8,500 electronics (mobile)": {
        "id": "pay_QjR3k5K8xY2mNp",
        "amount": 850000,  # Razorpay uses paise
        "currency": "INR",
        "status": "captured",
        "method": "upi",
        "description": "Samsung Galaxy Buds Pro",
        "email": "customer@example.com",
        "contact": "+919876543210",
        "notes": {"merchant_category": "electronics", "shipping_method": "standard"},
        "created_at": 1719849600,
    },
    "pay_LmN8p2R4wX7qAs — ₹2,200 fashion (desktop)": {
        "id": "pay_LmN8p2R4wX7qAs",
        "amount": 220000,
        "currency": "INR",
        "status": "captured",
        "method": "card",
        "description": "Nike Air Max 90",
        "email": "buyer@example.com",
        "contact": "+919123456789",
        "notes": {"merchant_category": "fashion", "shipping_method": "express"},
        "created_at": 1719936000,
    },
    "pay_Xk9mW3pL5nR2Bt — ₹42,000 electronics (desktop, high value)": {
        "id": "pay_Xk9mW3pL5nR2Bt",
        "amount": 4200000,
        "currency": "INR",
        "status": "captured",
        "method": "card",
        "description": "MacBook Air M3",
        "email": "premium@example.com",
        "contact": "+919988776655",
        "notes": {"merchant_category": "electronics", "shipping_method": "express"},
        "created_at": 1720022400,
    },
    "pay_Rp4nT7mK2xW9Cs — ₹650 grocery (app, low value)": {
        "id": "pay_Rp4nT7mK2xW9Cs",
        "amount": 65000,
        "currency": "INR",
        "status": "captured",
        "method": "upi",
        "description": "Monthly grocery order",
        "email": "weekly@example.com",
        "contact": "+919555444333",
        "notes": {"merchant_category": "grocery", "shipping_method": "standard"},
        "created_at": 1720108800,
    },
}

st.markdown("---")

if mode == "Demo (sample payments)":
    selected_key = st.selectbox("Select a sample Razorpay payment", list(SAMPLE_PAYMENTS.keys()))
    payment = SAMPLE_PAYMENTS[selected_key]
else:
    # Live mode — real Razorpay Test Mode API.
    import razorpay

    client = razorpay.Client(auth=(os.environ["RAZORPAY_KEY_ID"], os.environ["RAZORPAY_KEY_SECRET"]))

    fetch_method = st.radio(
        "How do you want to select a payment?",
        ["Browse recent payments", "Enter payment ID manually"],
        horizontal=True,
    )

    if fetch_method == "Browse recent payments":
        try:
            recent = client.payment.all({"count": 10})
            items = recent.get("items", [])
            if not items:
                st.warning("No payments found in your Test Mode account yet. Create one via Dashboard -> Payment Links.")
                st.stop()
            options = {
                f"{p['id']} — ₹{p['amount'] / 100:,.2f} — {p.get('method', 'unknown')} — {p.get('status', '')}": p
                for p in items
            }
            selected_label = st.selectbox("Select a recent payment", list(options.keys()))
            payment = options[selected_label]
        except Exception as e:
            st.error(f"Could not fetch payments: {e}")
            st.stop()
    else:
        payment_id = st.text_input("Enter Razorpay Payment ID", placeholder="pay_XXXXXXXXXXXXXX")
        if payment_id:
            try:
                payment = client.payment.fetch(payment_id)
            except Exception as e:
                st.error(f"Could not fetch payment: {e}")
                st.stop()
        else:
            st.write("Enter a payment ID to fetch from Razorpay.")
            st.stop()

col_raw, col_mapped = st.columns([1, 1])

with col_raw:
    st.subheader("Razorpay Payment Object")
    st.json({
        "id": payment["id"],
        "amount": payment["amount"],
        "currency": payment.get("currency", "INR"),
        "status": payment.get("status", "captured"),
        "method": payment.get("method", "card"),
        "description": payment.get("description", ""),
        "created_at": payment.get("created_at", ""),
    })

amount_inr = payment["amount"] / 100  # paise -> rupees
method = payment.get("method", "card")
notes = payment.get("notes", {})

device_map = {"upi": "mobile", "wallet": "mobile", "card": "desktop", "netbanking": "desktop", "emi": "desktop"}
device_type = device_map.get(method, "mobile")

has_notes = bool(notes.get("merchant_category")) and bool(notes.get("shipping_method"))

with col_mapped:
    st.subheader("Mapped to Pipeline Input")

    if has_notes:
        merchant_category = notes["merchant_category"]
        shipping_method = notes["shipping_method"]
    else:
        st.info(
            "This payment has no `notes.merchant_category` / `notes.shipping_method` set "
            "(real test payments created via Payment Links won't have these by default). "
            "Select them manually below — in production these would come from the order/cart data."
        )
        map_col1, map_col2 = st.columns(2)
        merchant_category = map_col1.selectbox(
            "Merchant category (manual — not in payment.notes)",
            ["electronics", "fashion", "grocery", "home", "beauty", "digital_goods"],
        )
        shipping_method = map_col2.selectbox(
            "Shipping method (manual — not in payment.notes)",
            ["standard", "express", "digital_delivery"],
        )

    mapping = {
        "amount_inr": f"₹{amount_inr:,.2f}",
        "source": f"payment.amount ({payment['amount']} paise) / 100",
        "device_type": f"{device_type} (from payment.method='{method}')",
        "merchant_category": f"{merchant_category}" + (" (from payment.notes)" if has_notes else " (manual)"),
        "shipping_method": f"{shipping_method}" + (" (from payment.notes)" if has_notes else " (manual)"),
    }
    for k, v in mapping.items():
        st.write(f"**{k}:** {v}")

st.markdown("---")

run = st.button("Run full pipeline on this payment", type="primary")

if run:
    st.subheader("Results")

    tab1, tab2 = st.tabs(["Prevention Score (at payment time)", "Dispute Score (if disputed)"])

    with tab1:
        st.markdown("*Scored at payment time — before any dispute exists*")
        txn = {
            "amount_inr": amount_inr,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "shipping_method": shipping_method,
        }
        prev_result = score_transaction(txn, prevention_model, prevention_cols)

        p1, p2 = st.columns(2)
        p1.metric("Dispute Risk Score", f"{prev_result['dispute_risk_score']:.1%}")
        tier = prev_result["risk_tier"]
        if tier == "HIGH":
            p2.error(f"{tier} RISK")
        elif tier == "MEDIUM":
            p2.warning(f"{tier} RISK")
        else:
            p2.success(f"{tier} RISK")

        st.markdown(f"**Action:** {prev_result['recommendation']}")

    with tab2:
        st.markdown("*Simulated: what happens if this payment becomes a dispute?*")

        st.markdown("**Scenario A: Dispute arrives with NO evidence on file**")
        case_no_evidence = {
            "transaction_id": payment["id"],
            "customer_id": "CUST000100",
            "amount_inr": amount_inr,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "shipping_method": shipping_method,
            "late_filing": 0,
            "account_age_days_at_dispute": 120,
            "has_tracking_number": 0,
            "has_delivery_confirmation": 0,
            "has_signed_pod": 0,
            "has_courier_communication": 0,
            "has_avs_match": 0,
        }
        result_no = score_dispute(case_no_evidence, model, calibrator, feature_cols, history_df, write_audit=False)
        ev_no = result_no["ev_decision"]

        def _action_str(action):
            return action.value if hasattr(action, "value") else str(action)

        a1, a2, a3 = st.columns(3)
        a1.metric("Win Probability", f"{result_no['win_probability']:.0%}")
        a2.metric("Decision", _action_str(ev_no["action"]))
        a3.metric("EV(contest)", f"₹{ev_no['ev_contest']:,.0f}")

        st.markdown("---")
        st.markdown("**Scenario B: Dispute arrives with FULL evidence (merchant followed prevention recommendation)**")
        case_full_evidence = {
            **case_no_evidence,
            "has_tracking_number": 1,
            "has_delivery_confirmation": 1,
            "has_signed_pod": 1 if shipping_method != "digital_delivery" else 0,
            "has_courier_communication": 1,
            "has_avs_match": 1,
        }
        result_full = score_dispute(case_full_evidence, model, calibrator, feature_cols, history_df, write_audit=False)
        ev_full = result_full["ev_decision"]

        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Win Probability", f"{result_full['win_probability']:.0%}",
            delta=f"+{(result_full['win_probability'] - result_no['win_probability']):.0%}",
        )
        b2.metric("Decision", _action_str(ev_full["action"]))
        b3.metric(
            "EV(contest)", f"₹{ev_full['ev_contest']:,.0f}",
            delta=f"+₹{ev_full['ev_contest'] - ev_no['ev_contest']:,.0f}",
        )

        delta_prob = result_full["win_probability"] - result_no["win_probability"]
        if delta_prob > 0.1:
            st.success(
                f"**The prevention layer's value**: by collecting evidence proactively, "
                f"win probability jumps from {result_no['win_probability']:.0%} to "
                f"{result_full['win_probability']:.0%} — turning a potential loss into a recovery."
            )

st.divider()
st.markdown(
    "**How this integrates with Razorpay:** In production, this system would hook into "
    "Razorpay's payment webhook (`payment.captured`). Every captured payment gets a prevention "
    "score in real time. High-risk payments trigger automatic evidence collection requests to "
    "the merchant via the Razorpay Dashboard or API. If a dispute later arrives via "
    "`payment.dispute.created` webhook, the scoring pipeline runs with whatever evidence is "
    "already on file."
)
