"""
Deployment bootstrap: trains all model artifacts on first use if they aren't
already on disk.

WHY THIS EXISTS: models/*.pkl and data/*.csv are deliberately gitignored (see
README — no binary blobs committed to git; everything reproducible from a
seeded generator instead, exactly what `make train` does locally). That's the
right call for local development, but it means a fresh deploy — Streamlit
Community Cloud clones the repo, runs `pip install -r requirements.txt`, and
starts the app; it never runs `make train` — begins with none of them
present. On Community Cloud specifically, the filesystem is also ephemeral
across sleep/wake cycles (apps sleep after 12h idle and wake into a brand new
container), so this isn't a one-time bootstrap problem either — it recurs
every time the app wakes from sleep.

Every Streamlit page that needs a trained model calls `ensure_trained()` from
here instead of independently checking for missing files. Because
`st.cache_resource`'s cache key is the function object itself, importing this
one function everywhere guarantees training runs at most once per awake
container, not once per page a visitor happens to open first.

Not imported by scripts/ or tests/ — those already have, and rely on, an
explicit `make train` step and a hard FileNotFoundError if it's skipped. This
module exists purely to make the deployed *app* self-sufficient.
"""
import streamlit as st

from src import config


@st.cache_resource(show_spinner="First request on this instance — training models (~15-30s), then cached for every visitor until this instance sleeps...")
def ensure_trained() -> bool:
    """Trains everything the app's pages need (detector, calibrator,
    prevention model) only if it isn't already on disk. Returns True once
    artifacts are confirmed present, so callers can ignore the return value
    and just rely on the cache having run this at least once.

    Calls src.model.train(fast=True): a smaller synthetic dataset, fewer
    trees, and no SHAP/plot generation (nothing here reads those PNGs at
    runtime) -- full local `make train` produced every number in the README
    and is untouched. A first deploy attempt trained on the full, non-fast
    settings and got CPU-throttled by Streamlit Community Cloud's free tier
    before finishing; fast=True exists specifically so a cold start fits
    inside that budget. The live app's own freshly-trained metrics will
    therefore differ slightly from the README's documented numbers -- a
    disclosed tradeoff, not a silent inconsistency."""
    required = [
        config.MODELS_DIR / "detector.pkl",
        config.MODELS_DIR / "calibrator.pkl",
        config.MODELS_DIR / "feature_cols.json",
        config.DATA_DIR / "history_train.csv",
        config.MODELS_DIR / "prevention_model.pkl",
        config.MODELS_DIR / "prevention_feature_cols.json",
    ]
    if not all(p.exists() for p in required):
        from src import model as model_module

        model_module.train(verbose=False, fast=True)
    return True
