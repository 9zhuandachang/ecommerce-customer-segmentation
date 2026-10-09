"""
Segmentation Module
K-Means clustering on RFM features.
Compares k = 3..6 using silhouette score, segment size and interpretability.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)

SEGMENT_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "aov",
    "product_diversity",
]


def build_segmentation_pipeline(k: int, random_state: int = 42) -> Pipeline:
    """Return an unfitted sklearn Pipeline: RobustScaler + KMeans(k)."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init="auto")),
    ])


def fit_segmentation(
    X: pd.DataFrame,
    k_range: range = range(3, 7),
    random_state: int = 42,
) -> dict:
    """
    Fit KMeans for each k in k_range.
    Returns dict with keys: best_k, best_pipeline, results (list of per-k dicts).
    """
    results = []

    for k in k_range:
        pipe = build_segmentation_pipeline(k, random_state)
        labels = pipe.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(len(X), 5000), random_state=random_state)
        counts = pd.Series(labels).value_counts()
        min_size = int(counts.min())
        max_size = int(counts.max())
        logger.info(f"  k={k}  silhouette={sil:.4f}  min_segment={min_size}  max_segment={max_size}")
        results.append({
            "k": k,
            "silhouette": sil,
            "min_segment": min_size,
            "max_segment": max_size,
            "pipeline": pipe,
            "labels": labels,
        })

    # Select best k by silhouette score
    best = max(results, key=lambda r: r["silhouette"])
    logger.info(f"Best k selected: {best['k']}  (silhouette={best['silhouette']:.4f})")

    return {
        "best_k": best["k"],
        "best_pipeline": best["pipeline"],
        "best_labels": best["labels"],
        "results": results,
    }


def assign_segments(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Assign segment labels using a fitted pipeline."""
    return pipeline.predict(X)


def label_segments(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Add human-readable segment names based on centroid characteristics.
    Segments are named by ranking their median recency (ascending) and
    median frequency (descending).
    """
    df = df.copy()
    profile = df.groupby("segment")[feature_cols].median()

    # Score: low recency + high frequency + high monetary = Champions
    profile["rfm_score"] = (
        -profile["recency_days"].rank()
        + profile["frequency"].rank()
        + profile["monetary"].rank()
    )
    rank = profile["rfm_score"].rank(ascending=False).astype(int)

    name_map = {}
    for seg, r in rank.items():
        if r == 1:
            name_map[seg] = "Champions"
        elif r == 2:
            name_map[seg] = "Loyal"
        elif r == 3:
            name_map[seg] = "At Risk"
        else:
            name_map[seg] = f"Segment {r}"

    df["segment_name"] = df["segment"].map(name_map)
    return df, name_map


def compute_segment_profiles(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Aggregate per-segment statistics for dashboard display."""
    profile = df.groupby(["segment", "segment_name"]).agg(
        size=("CustomerID", "count"),
        recency_median=("recency_days", "median"),
        frequency_median=("frequency", "median"),
        monetary_median=("monetary", "median"),
        aov_median=("aov", "median"),
        diversity_median=("product_diversity", "median"),
        repurchase_rate=("repurchase", "mean"),
    ).reset_index()
    return profile


def save_segmentation(pipeline: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Segmentation pipeline saved to {path}")


def load_segmentation(path: str) -> Pipeline:
    return joblib.load(path)
