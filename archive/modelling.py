"""
Modelling Module
Six experiments:
  1. Logistic Regression (LR)
  2. Gradient-Boosted Decision Trees (LightGBM / GBDT)
  3. Soft-voting Ensemble  w=0.25  → 0.25*LR + 0.75*GBDT
  4. Soft-voting Ensemble  w=0.50  → 0.50*LR + 0.50*GBDT
  5. Soft-voting Ensemble  w=0.75  → 0.75*LR + 0.25*GBDT
  6. [Baseline] Majority-class dummy
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss,
)
from sklearn.dummy import DummyClassifier
import lightgbm as lgb
import joblib

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary",
    "aov",
    "product_diversity",
    "count_30d",
    "count_90d",
]


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])


def build_gbdt_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )),
    ])


def build_dummy_pipeline() -> Pipeline:
    return Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])


# ---------------------------------------------------------------------------
# Soft-voting ensemble (post-fit)
# ---------------------------------------------------------------------------

class SoftVotingEnsemble:
    """Combines LR and GBDT probability scores with a fixed weight w for LR."""

    def __init__(self, lr_pipeline: Pipeline, gbdt_pipeline: Pipeline, w: float):
        assert 0.0 <= w <= 1.0, "w must be in [0, 1]"
        self.lr = lr_pipeline
        self.gbdt = gbdt_pipeline
        self.w = w

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_lr = self.lr.predict_proba(X)[:, 1]
        p_gbdt = self.gbdt.predict_proba(X)[:, 1]
        p_ensemble = self.w * p_lr + (1 - self.w) * p_gbdt
        return np.column_stack([1 - p_ensemble, p_ensemble])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Time-based train/val/test split
# ---------------------------------------------------------------------------

def time_split(
    snapshots: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by cutoff_date chronologically.
    Earlier label windows finish before the next partition begins.
    """
    dates = sorted(snapshots["cutoff_date"].unique())
    n = len(dates)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_dates = dates[:n_train]
    val_dates = dates[n_train: n_train + n_val]
    test_dates = dates[n_train + n_val:]

    if not test_dates:
        # fallback: use last date as test
        test_dates = [dates[-1]]
        val_dates = dates[n_train: -1] or [dates[n_train - 1]]

    train = snapshots[snapshots["cutoff_date"].isin(train_dates)]
    val = snapshots[snapshots["cutoff_date"].isin(val_dates)]
    test = snapshots[snapshots["cutoff_date"].isin(test_dates)]

    logger.info(
        f"Split: train={len(train)} val={len(val)} test={len(test)} "
        f"({len(train_dates)} / {len(val_dates)} / {len(test_dates)} cutoffs)"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Run all six experiments
# ---------------------------------------------------------------------------

def run_experiments(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    label_col: str = "repurchase",
) -> dict:
    """
    Train and evaluate all six experiments.
    Returns a dict with 'results' (per-model metrics) and fitted model objects.
    """
    X_train = train[feature_cols]
    y_train = train[label_col]
    X_val = val[feature_cols]
    y_val = val[label_col]
    X_test = test[feature_cols]
    y_test = test[label_col]

    # ---- Base models ----
    logger.info("Training Logistic Regression...")
    lr_pipe = build_lr_pipeline()
    lr_pipe.fit(X_train, y_train)

    logger.info("Training LightGBM (GBDT)...")
    gbdt_pipe = build_gbdt_pipeline()
    gbdt_pipe.fit(X_train, y_train)

    logger.info("Training Dummy baseline...")
    dummy_pipe = build_dummy_pipeline()
    dummy_pipe.fit(X_train, y_train)

    # ---- Ensembles ----
    ensembles = {
        "Ensemble_w025": SoftVotingEnsemble(lr_pipe, gbdt_pipe, w=0.25),
        "Ensemble_w050": SoftVotingEnsemble(lr_pipe, gbdt_pipe, w=0.50),
        "Ensemble_w075": SoftVotingEnsemble(lr_pipe, gbdt_pipe, w=0.75),
    }

    all_models = {
        "Logistic Regression": lr_pipe,
        "LightGBM (GBDT)": gbdt_pipe,
        "Dummy Baseline": dummy_pipe,
        **ensembles,
    }

    results = {}
    for name, model in all_models.items():
        if hasattr(model, "predict_proba"):
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
        else:
            val_prob = model.predict(X_val).astype(float)
            test_prob = model.predict(X_test).astype(float)

        val_metrics = evaluate(y_val.values, val_prob)
        test_metrics = evaluate(y_test.values, test_prob)
        logger.info(f"  {name}: val_auc={val_metrics['roc_auc']} test_auc={test_metrics['roc_auc']}")
        results[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "model": model,
            "val_prob": val_prob,
            "test_prob": test_prob,
        }

    return results


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)


def select_best_model(results: dict, metric: str = "roc_auc", split: str = "val"):
    """Return the name of the best model by validation metric."""
    best_name = max(
        [k for k in results if k != "Dummy Baseline"],
        key=lambda k: results[k][split][metric],
    )
    logger.info(f"Best model: {best_name} ({split}_{metric}={results[best_name][split][metric]})")
    return best_name
