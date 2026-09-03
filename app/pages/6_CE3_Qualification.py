"""
CE3.0 Qualification Checker — deterministic rules engine for Visa
Compelling Evidence 3.0.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src import ce3, config
from src.ce3 import PriorTransaction

st.set_page_config(page_title="CE3.0 Qualification", layout="wide")
st.title("Visa Compelling Evidence 3.0 — Qualification Checker")
st.caption(
    "Deterministic rules engine. No ML. Determines whether a fraud dispute can be "
    "blocked pre-emptively under CE3.0, shifting liability to the issuer and keeping "
    "the event off the merchant's VAMP ratio."
)

st.warning(
    f"**Scope note:** CE3.0 applies to Visa reason code {config.CE3_REASON_CODE} "
    f"(Fraud — Card Absent Environment). The rest of this project is scoped to "
    f"reason code {config.REASON_CODE} (Merchandise Not Received). This page shows "
    f"the rule-engine pattern extending to a second reason code — it is not "
    f"applicable to 13.1 disputes."
)

st.markdown("---")

SCENARIOS = {
    "Qualifies — 2 priors, IP + device match": {
        "reason_code": "10.4",
        "disputed": {
            "date": datetime(2026, 9, 1),
            "ip_address": "203.0.113.45",
            "device_id": "dev_a1b2c3",
            "account_login_id": "user_9981",
            "shipping_address": "12 MG Road, Pune",
        },
        "priors": [
            PriorTransaction(
                "TXN-A", datetime(2026, 3, 1), False,
                "203.0.113.45", "dev_a1b2c3", "user_9981", "12 MG Road, Pune",
            ),
            PriorTransaction(
                "TXN-B", datetime(2026, 1, 15), False,
                "203.0.113.45", "dev_a1b2c3", "user_9981", "12 MG Road, Pune",
            ),
        ],
    },
    "Fails — priors too recent (under 120 days)": {
        "reason_code": "10.4",
        "disputed": {
            "date": datetime(2026, 9, 1),
            "ip_address": "203.0.113.45",
            "device_id": "dev_a1b2c3",
        },
        "priors": [
            PriorTransaction("TXN-C", datetime(2026, 8, 1), False, "203.0.113.45", "dev_a1b2c3"),
            PriorTransaction("TXN-D", datetime(2026, 7, 20), False, "203.0.113.45", "dev_a1b2c3"),
        ],
    },
    "Fails — no IP or device anchor (only address + login match)": {
        "reason_code": "10.4",
        "disputed": {
            "date": datetime(2026, 9, 1),
            "ip_address": "198.51.100.7",
            "device_id": "dev_zzz999",
            "account_login_id": "user_9981",
            "shipping_address": "12 MG Road, Pune",
        },
        "priors": [
            PriorTransaction(
                "TXN-E", datetime(2026, 3, 1), False,
                "203.0.113.45", "dev_a1b2c3", "user_9981", "12 MG Road, Pune",
            ),
            PriorTransaction(
                "TXN-F", datetime(2026, 1, 15), False,
                "192.0.2.99", "dev_b7x2y1", "user_9981", "12 MG Road, Pune",
            ),
        ],
    },
    "Fails — one prior was itself disputed": {
        "reason_code": "10.4",
        "disputed": {
            "date": datetime(2026, 9, 1),
            "ip_address": "203.0.113.45",
            "device_id": "dev_a1b2c3",
        },
        "priors": [
            PriorTransaction("TXN-G", datetime(2026, 3, 1), False, "203.0.113.45", "dev_a1b2c3"),
            PriorTransaction("TXN-H", datetime(2026, 1, 15), True, "203.0.113.45", "dev_a1b2c3"),
        ],
    },
    "Fails — wrong reason code (13.1)": {
        "reason_code": "13.1",
        "disputed": {
            "date": datetime(2026, 9, 1),
            "ip_address": "203.0.113.45",
            "device_id": "dev_a1b2c3",
        },
        "priors": [
            PriorTransaction("TXN-I", datetime(2026, 3, 1), False, "203.0.113.45", "dev_a1b2c3"),
            PriorTransaction("TXN-J", datetime(2026, 1, 15), False, "203.0.113.45", "dev_a1b2c3"),
        ],
    },
}

choice = st.selectbox("Select a scenario", list(SCENARIOS.keys()))
scenario = SCENARIOS[choice]

col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("Disputed Transaction")
    d = scenario["disputed"]
    st.write(f"**Reason code:** {scenario['reason_code']}")
    st.write(f"**Date:** {d['date'].date()}")
    for elem in config.CE3_MATCHABLE_ELEMENTS:
        if d.get(elem):
            st.write(f"**{elem}:** `{d[elem]}`")

    st.subheader("Prior Transactions")
    prior_rows = []
    for p in scenario["priors"]:
        age = (d["date"] - p.date).days
        prior_rows.append({
            "ID": p.transaction_id,
            "Date": p.date.date(),
            "Age (days)": age,
            "Disputed?": "Yes" if p.was_disputed else "No",
            "IP": p.ip_address or "—",
            "Device": p.device_id or "—",
        })
    st.dataframe(pd.DataFrame(prior_rows), hide_index=True, use_container_width=True)

with col_r:
    st.subheader("Qualification Result")

    result = ce3.check_ce3_qualification(scenario["disputed"], scenario["priors"], scenario["reason_code"])
    summary = ce3.ce3_outcome_summary(result)

    if result.qualifies:
        st.success(f"**{summary['status']}**")
    else:
        st.error(f"**{summary['status']}**")

    st.markdown(f"**Effect:** {summary['effect']}")
    st.markdown(f"**VAMP impact:** {summary['vamp_impact']}")
    st.markdown(f"**Action:** {summary['action']}")

    st.markdown("---")
    st.markdown("**Rule-by-rule checks**")

    checks_display = [{
        "Rule": f"Reason code is {config.CE3_REASON_CODE}",
        "Result": "PASS" if result.checks.get("reason_code_eligible") else "FAIL",
    }]
    if "prior_transactions_in_window" in result.checks:
        n = result.checks["prior_transactions_in_window"]
        checks_display.append({
            "Rule": (
                f">={config.CE3_MIN_PRIOR_TRANSACTIONS} undisputed priors in "
                f"{config.CE3_MIN_PRIOR_AGE_DAYS}-{config.CE3_MAX_PRIOR_AGE_DAYS} day window"
            ),
            "Result": f"{'PASS' if n >= config.CE3_MIN_PRIOR_TRANSACTIONS else 'FAIL'} (found {n})",
        })
    if "matching_elements_count" in result.checks:
        n = result.checks["matching_elements_count"]
        checks_display.append({
            "Rule": f">={config.CE3_MIN_MATCHING_ELEMENTS} data elements match consistently",
            "Result": f"{'PASS' if n >= config.CE3_MIN_MATCHING_ELEMENTS else 'FAIL'} (found {n})",
        })
    if "has_anchor_element" in result.checks:
        checks_display.append({
            "Rule": "At least one match is IP address or device ID",
            "Result": "PASS" if result.checks["has_anchor_element"] else "FAIL",
        })

    st.dataframe(pd.DataFrame(checks_display), hide_index=True, use_container_width=True)

    if result.matching_elements:
        st.markdown(f"**Matching elements:** {', '.join(result.matching_elements)}")
    if result.qualifying_transactions:
        st.markdown(f"**Qualifying priors:** {', '.join(result.qualifying_transactions)}")
    if result.failures:
        st.markdown("**Why it failed:**")
        for f in result.failures:
            st.write(f"- {f}")

st.divider()
st.markdown(
    """
**Why CE3.0 belongs in this system:** it is the clearest example of the project's core
thesis — prevention beats representment. A CE3.0-qualifying dispute is blocked before it
is ever filed, so the merchant keeps the transaction value *and* the event never touches
their VAMP ratio. Contesting the same dispute successfully would recover the money but
leave the portfolio risk in place.
"""
)
