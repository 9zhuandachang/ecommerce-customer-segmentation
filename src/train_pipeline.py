"""Offline prototype training from audited customer snapshots."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import load_modeling_snapshots, partition_modeling_snapshots
from src.modelling import (
    FEATURE_COLS as MODEL_FEATURES,
    run_experiments,
    save_model,
    select_best_model,
)
from src.segmentation import (
    SEGMENT_FEATURES,
    assign_segments,
    fit_segmentation,
    label_segments,
    save_segmentation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE_SNAPSHOTS_PATH = (
    PROJECT_ROOT / "data/processed/customer_snapshot_pipeline/customer_snapshots.csv"
)
SNAPSHOTS_PATH = PROJECT_ROOT / "data/processed/snapshots.parquet"
SCORED_PATH = PROJECT_ROOT / "data/processed/scored_customers.parquet"
RESULTS_PATH = PROJECT_ROOT / "data/processed/experiment_results.pkl"
MODEL_DIR = PROJECT_ROOT / "models"
SEG_PIPELINE_PATH = MODEL_DIR / "segmentation_pipeline.pkl"


def step_load_snapshots() -> pd.DataFrame:
    """Adapt the canonical snapshot CSV without recomputing cleaning or labels."""
    if not SOURCE_SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_SNAPSHOTS_PATH}. Run build_customer_snapshots.py first."
        )
    snapshots = load_modeling_snapshots(SOURCE_SNAPSHOTS_PATH)
    partition_modeling_snapshots(snapshots)
    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(SNAPSHOTS_PATH, index=False)
    logger.info("Loaded %s audited customer snapshots", f"{len(snapshots):,}")
    return snapshots


def step_segmentation(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on the recorded train partition and assign all snapshot segments."""
    train, validation, test = partition_modeling_snapshots(snapshots)
    train_medians = train[SEGMENT_FEATURES].median()
    x_train = train[SEGMENT_FEATURES].fillna(train_medians)

    logger.info("Fitting K-means candidates k=3..6 on training snapshots")
    result = fit_segmentation(x_train)
    save_segmentation(result["best_pipeline"], str(SEG_PIPELINE_PATH))

    x_all = snapshots[SEGMENT_FEATURES].fillna(train_medians)
    snapshots = snapshots.copy()
    snapshots["segment"] = assign_segments(result["best_pipeline"], x_all)

    named_train = train.copy()
    named_train["segment"] = assign_segments(result["best_pipeline"], x_train)
    _, name_map = label_segments(named_train, SEGMENT_FEATURES)
    snapshots["segment_name"] = snapshots["segment"].map(name_map)

    k_results = [
        {
            "k": row["k"],
            "silhouette": row["silhouette"],
            "min_segment": row["min_segment"],
            "max_segment": row["max_segment"],
        }
        for row in result["results"]
    ]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(k_results, MODEL_DIR / "k_selection_results.pkl")
    joblib.dump(name_map, MODEL_DIR / "segment_name_map.pkl")
    snapshots.to_parquet(SNAPSHOTS_PATH, index=False)
    logger.info("Saved segmentation with k=%s", result["best_k"])
    return snapshots, train, validation, test


def step_modelling(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Run the existing model prototype on the same recorded partitions."""
    results = run_experiments(
        train, validation, test, feature_cols=MODEL_FEATURES
    )
    best_name = select_best_model(results, metric="pr_auc")
    save_model(results[best_name]["model"], str(MODEL_DIR / "best_model.pkl"))
    joblib.dump(results, RESULTS_PATH)
    joblib.dump(best_name, MODEL_DIR / "best_model_name.pkl")

    for name, row in results.items():
        logger.info(
            "%s: validation AP=%s, test AP=%s",
            name,
            row["val"]["pr_auc"],
            row["test"]["pr_auc"],
        )
    return results


def step_score_snapshots(snapshots: pd.DataFrame) -> None:
    """Use the saved classifier for historical review at each supported cutoff."""
    best_model = joblib.load(MODEL_DIR / "best_model.pkl")
    scored = snapshots.copy()
    scored["repurchase_score"] = best_model.predict_proba(
        scored[MODEL_FEATURES]
    )[:, 1]
    scored["rank"] = (
        scored.groupby("cutoff_date")["repurchase_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    scored.to_parquet(SCORED_PATH, index=False)
    logger.info("Saved scores for %s snapshots", f"{len(scored):,}")


def main() -> None:
    snapshots = step_load_snapshots()
    snapshots, train, validation, test = step_segmentation(snapshots)
    step_modelling(train, validation, test)
    step_score_snapshots(snapshots)
    logger.info("Prototype training complete. Run: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
