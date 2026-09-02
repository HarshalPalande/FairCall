"""
Feature store: turns raw dispute rows into a model-ready feature matrix.

The per-customer aggregate ("UID-style") features are computed as EXPANDING,
STRICTLY-PAST-ONLY statistics: a dispute at time T only ever sees that
customer's disputes with dispute_date < T. This is what makes the later
time-respecting train/calibration/test split meaningful — if we computed
these aggregates over the full dataset first, we'd leak future outcomes
(and future win-rate) backward into earlier rows, which is a classic and
easy-to-miss chargeback-modeling bug.
"""
import numpy as np
import pandas as pd

from src import config

CATEGORICAL_COLS = ["merchant_category", "device_type", "shipping_method"]
EVIDENCE_COLS = [f"has_{e}" for e in config.EVIDENCE_TYPES]

RAW_NUMERIC_COLS = [
    "amount_inr",
    "late_filing",
    "account_age_days_at_dispute",
]

UID_AGGREGATE_COLS = [
    "cust_prior_dispute_count",
    "cust_prior_win_rate",
    "cust_prior_avg_amount",
    "amount_dev_from_cust_avg",
]


def add_uid_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly-past per-customer aggregate features. df must already be
    sorted by dispute_date (the caller / pipeline guarantees this)."""
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    g = df.groupby("customer_id")

    prior_count = g.cumcount()
    # cumulative sum/mean of `won` and `amount` EXCLUDING current row via shift
    won_cumsum = g["won"].cumsum() - df["won"]
    win_rate = won_cumsum / prior_count.replace(0, np.nan)
    global_prior = df["won"].expanding().mean().shift(1).fillna(df["won"].mean())

    amount_cumsum = g["amount_inr"].cumsum() - df["amount_inr"]
    prior_avg_amount = amount_cumsum / prior_count.replace(0, np.nan)

    df = df.copy()
    df["cust_prior_dispute_count"] = prior_count
    df["cust_prior_win_rate"] = win_rate.fillna(global_prior)
    df["cust_prior_avg_amount"] = prior_avg_amount.fillna(df["amount_inr"].median())
    df["amount_dev_from_cust_avg"] = (
        df["amount_inr"] - df["cust_prior_avg_amount"]
    ) / df["cust_prior_avg_amount"].replace(0, np.nan)
    df["amount_dev_from_cust_avg"] = df["amount_dev_from_cust_avg"].fillna(0.0)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering: UID aggregates + one-hot categoricals.
    Returns a numeric-only frame ready for XGBoost, index-aligned to df."""
    df = add_uid_aggregates(df)
    feat_cols = RAW_NUMERIC_COLS + UID_AGGREGATE_COLS + EVIDENCE_COLS
    X = df[feat_cols].copy()
    X = pd.concat([X, pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)], axis=1)
    return X, df


def feature_vector_for_case(raw_row: dict, history_df: pd.DataFrame) -> pd.Series:
    """Build a single-row feature vector for a live dispute, using only
    `history_df` (disputes strictly before this one) for UID aggregates.
    Used by the scoring pipeline / counterfactual engine, not training."""
    cust_hist = history_df[history_df["customer_id"] == raw_row["customer_id"]]
    prior_count = len(cust_hist)
    win_rate = cust_hist["won"].mean() if prior_count > 0 else history_df["won"].mean()
    prior_avg_amount = (
        cust_hist["amount_inr"].mean() if prior_count > 0 else history_df["amount_inr"].median()
    )
    dev = (raw_row["amount_inr"] - prior_avg_amount) / prior_avg_amount if prior_avg_amount else 0.0

    row = {
        "amount_inr": raw_row["amount_inr"],
        "late_filing": raw_row["late_filing"],
        "account_age_days_at_dispute": raw_row["account_age_days_at_dispute"],
        "cust_prior_dispute_count": prior_count,
        "cust_prior_win_rate": win_rate,
        "cust_prior_avg_amount": prior_avg_amount,
        "amount_dev_from_cust_avg": dev,
    }
    for e in config.EVIDENCE_TYPES:
        row[f"has_{e}"] = raw_row.get(f"has_{e}", 0)
    for col in CATEGORICAL_COLS:
        val = raw_row[col]
        row[f"{col}_{val}"] = 1
    return pd.Series(row)
