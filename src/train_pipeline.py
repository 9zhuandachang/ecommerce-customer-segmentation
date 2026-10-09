"""
Training Pipeline (Offline)
Runs all steps: clean → features → segmentation → model experiments → save.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# Make src importable when run as script
sys.path.insert(0, str(Path(__file__).parent))

from data_cleaning import load_raw_data, clean_data, print_audit
from features import (
    build_snapshot, get_eligible_cutoff_dates,
    FEATURE_COLS, LABEL_COL,
)
from segmentation import (
    fit_segmentation, assign_segments, label_segments,
    compute_segment_profiles, save_segmentation,
    SEGMENT_FEATURES,
)
from modelling import (
    run_experiments, save_model, select_best_model,
    time_split, FEATURE_COLS as MODEL_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
CLEAN_DATA_PATH = Path("data/processed/clean_data.parquet")
SNAPSHOTS_PATH = Path("data/processed/snapshots.parquet")
SEG_PIPELINE_PATH = Path("models/segmentation_pipeline.pkl")
MODEL_DIR = Path("models")
RESULTS_PATH = Path("data/processed/experiment_results.pkl")


# ---------------------------------------------------------------------------
# Step 1 – Data preparation
# ---------------------------------------------------------------------------

def step_prepare_data() -> pd.DataFrame:
    if CLEAN_DATA_PATH.exists():
        logger.info("Clean data already exists, loading from cache.")
        df = pd.read_parquet(CLEAN_DATA_PATH)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        return df

    raw = load_raw_data(str(RAW_DATA_PATH))
    clean, audit = clean_data(raw)
    print_audit(audit)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_DATA_PATH, index=False)
    logger.info(f"Saved clean data to {CLEAN_DATA_PATH}")
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])
    return clean


# ---------------------------------------------------------------------------
# Step 2 – Build snapshots
# ---------------------------------------------------------------------------

def step_build_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if SNAPSHOTS_PATH.exists():
        logger.info("Snapshots already exist, loading from cache.")
        snap = pd.read_parquet(SNAPSHOTS_PATH)
        return snap

    eligible = get_eligible_cutoff_dates(df)
    logger.info(f"Building {len(eligible)} snapshots...")

    frames = []
    for cutoff in eligible:
        snap = build_snapshot(df, cutoff)
        if not snap.empty:
            frames.append(snap)

    snapshots = pd.concat(frames, ignore_index=True)
    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(SNAPSHOTS_PATH, index=False)
    logger.info(f"Saved {len(snapshots):,} customer-snapshots to {SNAPSHOTS_PATH}")
    return snapshots


# ---------------------------------------------------------------------------
# Step 3 – Segmentation (fit on training data only)
# ---------------------------------------------------------------------------

def step_segmentation(snapshots: pd.DataFrame) -> tuple:
    train, val, test = time_split(snapshots)
    X_train_seg = train[SEGMENT_FEATURES].fillna(train[SEGMENT_FEATURES].median())

    logger.info("Fitting K-Means segmentation (k=3..6)...")
    seg_result = fit_segmentation(X_train_seg)

    save_segmentation(seg_result["best_pipeline"], str(SEG_PIPELINE_PATH))

    # Assign segments to ALL snapshots
    X_all_seg = snapshots[SEGMENT_FEATURES].fillna(snapshots[SEGMENT_FEATURES].median())
    labels = assign_segments(seg_result["best_pipeline"], X_all_seg)
    snapshots = snapshots.copy()
    snapshots["segment"] = labels
    snapshots, name_map = label_segments(snapshots, SEGMENT_FEATURES)

    # Save k-selection results for dashboard
    k_results = [
        {"k": r["k"], "silhouette": r["silhouette"],
         "min_segment": r["min_segment"], "max_segment": r["max_segment"]}
        for r in seg_result["results"]
    ]
    joblib.dump(k_results, MODEL_DIR / "k_selection_results.pkl")
    joblib.dump(name_map, MODEL_DIR / "segment_name_map.pkl")

    # Update parquet with segment info
    snapshots.to_parquet(SNAPSHOTS_PATH, index=False)
    logger.info(f"Segments saved. Best k={seg_result['best_k']}")

    return snapshots, train, val, test, seg_result


# ---------------------------------------------------------------------------
# Step 4 – Model experiments
# ---------------------------------------------------------------------------

def step_modelling(snapshots: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    logger.info("Running six model experiments...")
    results = run_experiments(train, val, test, feature_cols=MODEL_FEATURES)

    best_name = select_best_model(results)
    best_model = results[best_name]["model"]

    save_model(best_model, str(MODEL_DIR / "best_model.pkl"))
    joblib.dump(results, str(RESULTS_PATH))
    joblib.dump(best_name, str(MODEL_DIR / "best_model_name.pkl"))

    logger.info(f"\n=== Experiment Summary ===")
    for name, r in results.items():
        v = r["val"]
        t = r["test"]
        logger.info(
            f"  {name:<30} | val_auc={v['roc_auc']} val_pr={v['pr_auc']} "
            f"| test_auc={t['roc_auc']} test_pr={t['pr_auc']}"
        )

    return results


# ---------------------------------------------------------------------------
# Step 5 – Score production snapshot (latest cutoff)
# ---------------------------------------------------------------------------

def step_score_latest(df: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    """Score customers at the latest valid cutoff and save for dashboard."""
    from features import get_eligible_cutoff_dates, build_snapshot

    eligible = get_eligible_cutoff_dates(df)
    if not eligible:
        return

    latest_cutoff = eligible[-1]
    prod_snap = build_snapshot(df, latest_cutoff)

    # Load saved pipeline and model
    seg_pipe = joblib.load(SEG_PIPELINE_PATH)
    best_model = joblib.load(MODEL_DIR / "best_model.pkl")
    name_map = joblib.load(MODEL_DIR / "segment_name_map.pkl")

    X_seg = prod_snap[SEGMENT_FEATURES].fillna(prod_snap[SEGMENT_FEATURES].median())
    prod_snap["segment"] = seg_pipe.predict(X_seg)
    prod_snap["segment_name"] = prod_snap["segment"].map(name_map)

    X_feat = prod_snap[MODEL_FEATURES]
    if hasattr(best_model, "predict_proba"):
        prod_snap["repurchase_score"] = best_model.predict_proba(X_feat)[:, 1]
    else:
        prod_snap["repurchase_score"] = best_model.predict(X_feat).astype(float)

    prod_snap["rank"] = prod_snap["repurchase_score"].rank(ascending=False, method="first").astype(int)

    out = Path("data/processed/scored_customers.parquet")
    prod_snap.to_parquet(out, index=False)
    logger.info(f"Scored {len(prod_snap):,} customers → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Starting offline training pipeline")
    logger.info("=" * 60)

    if not RAW_DATA_PATH.exists():
        logger.error(
            f"Raw data not found at {RAW_DATA_PATH}\n"
            "Please download 'online_retail_II.xlsx' from:\n"
            "  https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            f"and place it at {RAW_DATA_PATH.absolute()}"
        )
        sys.exit(1)

    df = step_prepare_data()
    snapshots = step_build_snapshots(df)
    snapshots, train, val, test, seg_result = step_segmentation(snapshots)
    results = step_modelling(snapshots, train, val, test)
    step_score_latest(df, snapshots)

    logger.info("=" * 60)
    logger.info("Pipeline complete. Run:  streamlit run app/dashboard.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
