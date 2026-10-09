"""
Feature Engineering Module
Builds customer-level snapshots from cleaned transaction data.

For a given cutoff date t:
  - PAST 180 DAYS window  →  RFM + extended features (X)
  - NEXT 30 DAYS window   →  binary repurchase label (y)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _rfm_features(grp: pd.DataFrame, cutoff: pd.Timestamp, window_days: int = 180) -> pd.Series:
    """Compute per-customer features over the past `window_days`."""
    start = cutoff - pd.Timedelta(days=window_days)
    hist = grp[(grp["InvoiceDate"] >= start) & (grp["InvoiceDate"] < cutoff)]

    if hist.empty:
        return pd.Series(dtype=float)

    invoices = hist.drop_duplicates("InvoiceNo")
    last_date = hist["InvoiceDate"].max()

    recency = (cutoff - last_date).days
    frequency = invoices["InvoiceNo"].nunique()
    monetary = hist["Revenue"].sum()
    aov = monetary / frequency if frequency > 0 else 0.0
    product_diversity = hist["StockCode"].nunique()

    # Shorter windows
    start30 = cutoff - pd.Timedelta(days=30)
    start90 = cutoff - pd.Timedelta(days=90)
    count_30d = hist[hist["InvoiceDate"] >= start30]["InvoiceNo"].nunique()
    count_90d = hist[hist["InvoiceDate"] >= start90]["InvoiceNo"].nunique()

    return pd.Series({
        "recency_days": recency,
        "frequency": frequency,
        "monetary": monetary,
        "aov": aov,
        "product_diversity": product_diversity,
        "count_30d": count_30d,
        "count_90d": count_90d,
    })


def _repurchase_label(grp: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int = 30) -> int:
    """1 if the customer made at least one purchase in (cutoff, cutoff + horizon]."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    future = grp[(grp["InvoiceDate"] > cutoff) & (grp["InvoiceDate"] <= end)]
    return int(len(future) > 0)


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int = 180,
    horizon_days: int = 30,
    min_transactions: int = 1,
) -> pd.DataFrame:
    """
    Build a customer-level snapshot for a given cutoff date.

    Parameters
    ----------
    df           : clean transaction dataframe
    cutoff       : reference date t
    window_days  : lookback window for features (default 180)
    horizon_days : forward window for label (default 30)
    min_transactions : minimum invoices in lookback to include customer

    Returns
    -------
    DataFrame with one row per qualifying customer.
    """
    logger.info(f"Building snapshot at cutoff={cutoff.date()}")

    # Only keep customers who have history in the lookback window
    start = cutoff - pd.Timedelta(days=window_days)
    eligible = df[(df["InvoiceDate"] >= start) & (df["InvoiceDate"] < cutoff)]["CustomerID"].unique()
    logger.info(f"  Eligible customers with lookback history: {len(eligible):,}")

    grouped = df[df["CustomerID"].isin(eligible)].groupby("CustomerID")

    records = []
    for cust_id, grp in grouped:
        feats = _rfm_features(grp, cutoff, window_days)
        if feats.empty or feats["frequency"] < min_transactions:
            continue
        label = _repurchase_label(grp, cutoff, horizon_days)
        row = {"CustomerID": cust_id, **feats.to_dict(), "repurchase": label}
        records.append(row)

    snap = pd.DataFrame(records)
    snap["cutoff_date"] = cutoff.date()

    logger.info(
        f"  Snapshot size: {len(snap):,} customers | "
        f"repurchase rate: {snap['repurchase'].mean():.2%}"
    )
    return snap.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-snapshot builder (time-based splits)
# ---------------------------------------------------------------------------

def build_all_snapshots(
    df: pd.DataFrame,
    cutoff_dates: list[pd.Timestamp],
    window_days: int = 180,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Build and concatenate snapshots for multiple cutoff dates."""
    frames = []
    for cutoff in cutoff_dates:
        snap = build_snapshot(df, cutoff, window_days, horizon_days)
        if not snap.empty:
            frames.append(snap)
    return pd.concat(frames, ignore_index=True)


def get_eligible_cutoff_dates(
    df: pd.DataFrame,
    window_days: int = 180,
    horizon_days: int = 30,
    freq: str = "MS",  # month-start
) -> list[pd.Timestamp]:
    """
    Return all candidate cutoff dates where both lookback and label
    windows are fully contained within the dataset.
    """
    data_start = df["InvoiceDate"].min()
    data_end = df["InvoiceDate"].max()

    earliest_cutoff = data_start + pd.Timedelta(days=window_days)
    latest_cutoff = data_end - pd.Timedelta(days=horizon_days)

    all_dates = pd.date_range(start=earliest_cutoff, end=latest_cutoff, freq=freq)
    return list(all_dates)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary",
    "aov",
    "product_diversity",
    "count_30d",
    "count_90d",
]

LABEL_COL = "repurchase"


if __name__ == "__main__":
    import sys

    data_path = Path("data/processed/clean_data.parquet")
    if not data_path.exists():
        print(f"Clean data not found at {data_path}. Run data_cleaning.py first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    eligible = get_eligible_cutoff_dates(df)
    print(f"Eligible cutoff dates: {len(eligible)}")
    for d in eligible:
        print(f"  {d.date()}")

    # Build a single snapshot as demo
    cutoff = eligible[len(eligible) // 2]
    snap = build_snapshot(df, cutoff)
    out = Path("data/processed/snapshot_demo.parquet")
    snap.to_parquet(out, index=False)
    print(f"\nDemo snapshot saved to {out}")
    print(snap.describe())
