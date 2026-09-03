"""
Demo Mode — 50 simulated merchant sessions, 100 fresh synthetic transactions,
scored live by the real trained prevention model.

WHAT THIS IS NOT: this does not claim the model catches every risky transaction.
It does not -- the Prevention Score page discloses that this model has materially
less signal than the reactive detector (no evidence flags exist yet at payment
time), and this page's own live reference figure (computed fresh below, not
copied from that page -- see the threshold note there) will show the same modest,
honest recall. This page does not hide that behind a rigged demo; it puts the
model in front of a fresh batch every time you click and reports whatever
actually happens, including the misses.

WHAT THIS IS: every batch is freshly generated (no fixed seed), sampled from
`src.prevention.build_transaction_dataset()` -- the exact generator the model
was trained and evaluated on, not a hand-tuned scenario set. "Agents" here means
100 transactions are attributed round-robin to 50 synthetic session IDs for
narrative flavor; there is no adaptive or adversarial logic trying to evade the
model, by design -- that would cross into offense-capable territory this
project deliberately stays out of. The difficulty tiers (Easy/Moderate/Extreme)
are computed AFTER scoring, from how far each case sat from the model's correct
answer -- they are not chosen in advance to flatter the result.
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import config, prevention

st.set_page_config(page_title="Demo Mode", layout="wide")
st.title("Demo Mode — Live Batch")
st.caption(
    "50 simulated sessions, 100 fresh synthetic transactions, scored live by the "
    "real trained prevention model. A new random batch every click -- not a canned "
    "replay -- so the results genuinely vary run to run."
)


@st.cache_resource
def load_prevention_model():
    with open(config.MODELS_DIR / "prevention_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(config.MODELS_DIR / "prevention_feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


@st.cache_data(show_spinner="Calibrating a reference threshold on a fresh 8,000-transaction pool...")
def compute_reference_threshold(_model, _feature_cols):
    """Calibrate the 'flagged' cutoff live, once per session, instead of trusting a
    stale value out of artifacts/prevention_metrics.json.

    That JSON's threshold is rank-derived from ONE specific 18,000-row test set --
    `sorted_probs[n_actual_disputes * 0.5]` -- which makes it an artifact of that
    exact evaluation run's score distribution, not a portable cutoff. Reusing it
    against a freshly-generated pool with a different score distribution produced
    a live recall ~2x the disclosed figure in testing here -- a methodology bug,
    not honest run-to-run noise. Recomputing the same procedure fresh, on a large
    reference pool, right now, keeps the comparison self-consistent instead of
    comparing two different eras of data under one number.
    """
    ref_seed = config.SEED + 7  # fixed so the threshold is stable within a session, distinct from training's seed
    pool = prevention.build_transaction_dataset(seed=ref_seed)
    ref = pool.sample(n=8_000, random_state=ref_seed).reset_index(drop=True)
    X = prevention.build_prevention_features(ref).reindex(columns=_feature_cols, fill_value=0)
    probs = _model.predict_proba(X)[:, 1]
    truth = ref["became_dispute"].values

    n_pos = int(truth.sum())
    sorted_probs = np.sort(probs)[::-1]
    idx = min(int(n_pos * 0.5), len(sorted_probs) - 1)
    threshold = float(sorted_probs[idx])

    flagged = probs >= threshold
    tp = int(((flagged == 1) & (truth == 1)).sum())
    fp = int(((flagged == 1) & (truth == 0)).sum())
    fn = int(((flagged == 0) & (truth == 1)).sum())
    ref_recall = tp / (tp + fn) if (tp + fn) else 0.0
    ref_precision = tp / (tp + fp) if (tp + fp) else 0.0
    return threshold, ref_recall, ref_precision


try:
    prevention_model, prevention_cols = load_prevention_model()
except FileNotFoundError:
    st.error("Prevention model not found. Run `make train` first.")
    st.stop()

threshold, ref_recall, ref_precision = compute_reference_threshold(prevention_model, prevention_cols)

st.info(
    f"**Reference point, computed live** on a fresh 8,000-transaction pool (not the "
    f"README's numbers — those use a threshold tied to one specific old test set; "
    f"this page recalibrates fresh so the comparison stays apples-to-apples): "
    f"**{ref_recall:.0%} recall, {ref_precision:.0%} precision** at risk score ≥ "
    f"{threshold:.2f}. A 100-case batch is noisy — expect it to bounce around this, "
    f"not sit exactly on it."
)

run = st.button("▶ Run a new batch (50 agents, 100 payments)", type="primary")
st.caption(
    "**Tier** and **Outcome** use two different, deliberately different bars: "
    "Tier is the product's own cheap-to-act-on advisory line (≥0.30 = HIGH, "
    "\"collect evidence now\") — Outcome/CAUGHT uses this page's stricter, "
    "calibrated research threshold above. A `HIGH` tier next to a `MISSED` or "
    "`cleared` Outcome is expected, not a contradiction: the product would still "
    "tell a merchant to act on it, this page just doesn't count it as a catch."
)

if run:
    seed = int(time.time_ns() % (2**31 - 1))  # unseeded from config.SEED on purpose -- every run differs
    pool = prevention.build_transaction_dataset(seed=seed)
    batch = pool.sample(n=100, random_state=seed % (2**31 - 1)).reset_index(drop=True)
    batch = batch.sample(frac=1, random_state=(seed + 1) % (2**31 - 1)).reset_index(drop=True)
    batch["agent_id"] = [f"AGENT-{(i % 50) + 1:02d}" for i in range(len(batch))]

    progress = st.progress(0.0, text="Agents submitting payments...")
    live_table = st.empty()
    live_metrics = st.empty()

    results = []
    tp = fp = fn = tn = 0
    CHUNK = 10
    for i, row in batch.iterrows():
        txn = {
            "amount_inr": row["amount_inr"],
            "transaction_date": row["transaction_date"],
            "merchant_category": row["merchant_category"],
            "device_type": row["device_type"],
            "shipping_method": row["shipping_method"],
        }
        scored = prevention.score_transaction(txn, prevention_model, prevention_cols)
        risk_score = scored["dispute_risk_score"]
        truth = int(row["became_dispute"])
        flagged = risk_score >= threshold

        if flagged and truth == 1:
            outcome, tp = "✅ CAUGHT", tp + 1
        elif not flagged and truth == 1:
            outcome, fn = "❌ MISSED", fn + 1
        elif flagged and truth == 0:
            outcome, fp = "⚠️ FALSE ALARM", fp + 1
        else:
            outcome, tn = "· cleared", tn + 1

        # Difficulty is a SIGNED margin from the threshold, not distance from the 0/1
        # endpoints -- a fixed-cutoff version of this (bucketing raw risk_score vs.
        # truth on 0.3/0.6) collapsed "Moderate" to look identical to "Extreme"
        # whenever the calibrated threshold sat high, because most of that band fell
        # on the same (missed) side of it either way. margin > 0 means the score
        # landed on the correct side of the threshold; its magnitude, normalized by
        # the room available on that side, gives a 0..1 "how confidently right" scale.
        # Using unsigned distance here would be wrong: it would rate a confidently-
        # WRONG case (e.g. genuine risk scored near 0) as "easy" merely for being far
        # from the boundary, when it's actually the disguised-risk case that fooled
        # the model -- exactly what "Extreme" is supposed to capture.
        margin = (risk_score - threshold) if truth == 1 else (threshold - risk_score)
        margin_denom = ((1 - threshold) if truth == 1 else threshold) if margin >= 0 else (
            threshold if truth == 1 else (1 - threshold)
        )
        margin_norm = max(min(margin / max(margin_denom, 1e-6), 1.0), -1.0)
        difficulty_val = (1 - margin_norm) / 2
        difficulty = "Easy" if difficulty_val < 0.3 else ("Moderate" if difficulty_val < 0.6 else "Extreme")

        results.append({
            "Agent": row["agent_id"],
            "Amount ₹": f"{row['amount_inr']:,.0f}",
            "Category": row["merchant_category"],
            "Risk score": f"{risk_score:.2f}",
            "Tier": scored["risk_tier"],
            "Actually risky?": "Yes" if truth == 1 else "No",
            "Outcome": outcome,
            "Difficulty": difficulty,
        })

        if (i + 1) % CHUNK == 0 or i == len(batch) - 1:
            progress.progress((i + 1) / len(batch), text=f"Agents submitting payments... {i + 1}/100")
            live_table.dataframe(pd.DataFrame(results[-CHUNK:]), hide_index=True, use_container_width=True)
            with live_metrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Caught", tp)
                m2.metric("Missed", fn)
                m3.metric("False alarms", fp)
                m4.metric("Cleared correctly", tn)
            time.sleep(0.12)

    progress.empty()
    st.markdown("---")
    st.subheader("Final tally")
    st.caption("Caught / Missed / False alarms / Cleared counts are the same numbers shown live above.")

    total_risky = tp + fn
    total_flagged = tp + fp
    batch_recall = tp / total_risky if total_risky else 0.0
    batch_precision = tp / total_flagged if total_flagged else 0.0

    g1, g2 = st.columns(2)
    g1.metric("This batch's recall", f"{batch_recall:.0%}",
              f"{total_risky} of 100 were genuinely risky")
    g2.metric("This batch's precision", f"{batch_precision:.0%}",
              f"{total_flagged} flagged total")

    st.caption(
        f"Reference (8,000-case pool, same threshold): {ref_recall:.0%} recall / "
        f"{ref_precision:.0%} precision. This batch: {batch_recall:.0%} / "
        f"{batch_precision:.0%}. Different every run because the 100-case sample is "
        f"fresh every run — that's honest small-sample variance around the same "
        f"model, same threshold, not a different model performing better or worse."
    )

    st.markdown("#### By difficulty (computed after scoring, not chosen in advance)")
    df_results = pd.DataFrame(results)
    diff_summary = []
    for tier in ["Easy", "Moderate", "Extreme"]:
        sub = df_results[df_results["Difficulty"] == tier]
        risky_sub = sub[sub["Actually risky?"] == "Yes"]
        caught_sub = risky_sub[risky_sub["Outcome"] == "✅ CAUGHT"]
        diff_summary.append({
            "Difficulty": tier,
            "Cases": len(sub),
            "Genuinely risky": len(risky_sub),
            "Caught": f"{len(caught_sub)}/{len(risky_sub)}" if len(risky_sub) else "—",
        })
    st.dataframe(pd.DataFrame(diff_summary), hide_index=True, use_container_width=True)
    st.caption(
        "Easy = the model's score strongly agreed with the truth. Extreme = a genuinely "
        "risky transaction that looked clean, or a clean one that looked risky — the cases "
        "transaction-time features alone can't reliably separate. Expect Extreme recall to "
        "be the weakest number on this page; that's the honest shape of a harder problem, "
        "not a bug."
    )

    with st.expander("Full batch (all 100)"):
        st.dataframe(df_results, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption(
    "Uses `src.prevention.build_transaction_dataset()` and `src.prevention.score_transaction()` "
    "-- the same generator and the same trained model as the Prevention Score page and "
    "`make train`. No scenario here is hand-picked to make the model look better than its "
    "disclosed evaluation."
)
