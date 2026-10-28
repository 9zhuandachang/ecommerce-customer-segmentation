"""Monthly cutoff schedule, customer snapshots, and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_cleaning import CANONICAL_COLUMNS


SNAPSHOT_TO_MODEL_COLUMNS = {
    "snapshot_date": "cutoff_date",
    "customer_id": "CustomerID",
    "invoice_frequency_180d": "frequency",
    "monetary_value_180d": "monetary",
    "average_order_value_180d": "aov",
    "product_diversity_180d": "product_diversity",
    "invoice_count_30d": "count_30d",
    "invoice_count_90d": "count_90d",
    "repurchase_30d": "repurchase",
}


def load_modeling_snapshots(path: Path) -> pd.DataFrame:
    """Read canonical snapshots and adapt names for the existing ML prototype."""
    data = pd.read_csv(
        path,
        dtype={"customer_id": "string", "split": "string", "use_for_modeling": "string"},
    )
    required = set(SNAPSHOT_TO_MODEL_COLUMNS) | {
        "split", "use_for_modeling", "recency_days"
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Snapshot file is missing required columns: {missing}")
    if data.duplicated(["snapshot_date", "customer_id"]).any():
        raise ValueError("Customer snapshots must be unique by date and customer ID")

    flags = data["use_for_modeling"].str.lower().map({"true": True, "false": False})
    if flags.isna().any():
        raise ValueError("use_for_modeling must contain only true or false")
    data["use_for_modeling"] = flags.astype(bool)
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="raise")
    if not data["repurchase_30d"].isin([0, 1]).all():
        raise ValueError("repurchase_30d must be binary")
    return data.rename(columns=SNAPSHOT_TO_MODEL_COLUMNS)


def partition_modeling_snapshots(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Honor the snapshot pipeline's eligibility flag and chronological split."""
    eligible = snapshots.loc[snapshots["use_for_modeling"]].copy()
    partitions = tuple(
        eligible.loc[eligible["split"].eq(name)].copy()
        for name in ("train", "validation", "test")
    )
    if any(part.empty for part in partitions):
        raise ValueError("Train, validation, and test partitions must all be present")
    train, validation, test = partitions
    if not (
        train["cutoff_date"].max() < validation["cutoff_date"].min()
        and validation["cutoff_date"].max() < test["cutoff_date"].min()
    ):
        raise ValueError("Snapshot partitions must be in chronological order")
    return train, validation, test


@dataclass(frozen=True)
class PipelineConfig:
    observation_days: int = 180
    label_days: int = 30
    train_fraction: float = 2 / 3
    validation_fraction: float = 1 / 6


def generate_monthly_cutoffs(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    config: PipelineConfig,
) -> pd.DatetimeIndex:
    """Generate first-of-month cutoffs with complete feature and label windows."""
    earliest = data_start.normalize() + pd.Timedelta(days=config.observation_days)
    earliest = earliest if earliest.day == 1 else earliest + pd.offsets.MonthBegin(1)
    coverage_end_exclusive = data_end.normalize() + pd.Timedelta(days=1)
    latest = coverage_end_exclusive - pd.Timedelta(days=config.label_days)
    cutoffs = pd.date_range(earliest, latest, freq="MS")
    if cutoffs.empty:
        raise ValueError("The source date range is too short for the configured windows")
    return cutoffs


def make_cutoff_schedule(
    cutoffs: Iterable[pd.Timestamp], config: PipelineConfig
) -> pd.DataFrame:
    """Create chronological train/validation/test blocks and purge leakage gaps."""
    cutoff_index = pd.DatetimeIndex(cutoffs).sort_values().unique()
    n_cutoffs = len(cutoff_index)
    if n_cutoffs < 3:
        raise ValueError("At least three cutoff dates are required for time splits")

    n_train = max(1, int(n_cutoffs * config.train_fraction))
    n_validation = max(1, int(n_cutoffs * config.validation_fraction))
    if n_train + n_validation >= n_cutoffs:
        n_train = max(1, n_cutoffs - 2)
        n_validation = 1

    split = pd.Series("test", index=range(n_cutoffs), dtype="string")
    split.iloc[:n_train] = "train"
    split.iloc[n_train : n_train + n_validation] = "validation"

    schedule = pd.DataFrame({"snapshot_date": cutoff_index, "split": split})
    schedule["feature_start"] = schedule["snapshot_date"] - pd.Timedelta(
        days=config.observation_days
    )
    schedule["feature_end_exclusive"] = schedule["snapshot_date"]
    schedule["label_start"] = schedule["snapshot_date"]
    schedule["label_end_exclusive"] = schedule["snapshot_date"] + pd.Timedelta(
        days=config.label_days
    )
    schedule["use_for_modeling"] = True

    validation_start = schedule.loc[schedule["split"].eq("validation"), "snapshot_date"].min()
    test_start = schedule.loc[schedule["split"].eq("test"), "snapshot_date"].min()

    train_overlap = schedule["split"].eq("train") & schedule[
        "label_end_exclusive"
    ].gt(validation_start)
    validation_overlap = schedule["split"].eq("validation") & schedule[
        "label_end_exclusive"
    ].gt(test_start)
    schedule.loc[train_overlap, "split"] = "purged_before_validation"
    schedule.loc[validation_overlap, "split"] = "purged_before_test"
    schedule.loc[train_overlap | validation_overlap, "use_for_modeling"] = False
    return schedule


def build_customer_snapshots(
    clean: pd.DataFrame,
    schedule: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Build RFM-style features and a future 30-day repurchase label."""
    parts: list[pd.DataFrame] = []
    clean = clean.sort_values("invoice_date", kind="stable")

    for row in schedule.itertuples(index=False):
        cutoff = pd.Timestamp(row.snapshot_date)
        feature_start = pd.Timestamp(row.feature_start)
        label_end = pd.Timestamp(row.label_end_exclusive)

        past = clean.loc[clean["invoice_date"].lt(cutoff)]
        observed = past.loc[past["invoice_date"].ge(feature_start)]
        if observed.empty:
            continue

        invoice_level = (
            observed.groupby(["customer_id", "invoice"], as_index=False, observed=True)
            .agg(
                invoice_date=("invoice_date", "max"),
                order_value=("line_amount", "sum"),
            )
        )
        customer = observed.groupby("customer_id", observed=True).agg(
            last_purchase_datetime=("invoice_date", "max"),
            monetary_value_180d=("line_amount", "sum"),
            product_diversity_180d=("stock_code", "nunique"),
        )
        invoice_features = invoice_level.groupby("customer_id", observed=True).agg(
            invoice_frequency_180d=("invoice", "size"),
            average_order_value_180d=("order_value", "mean"),
        )
        recent_30 = (
            invoice_level.loc[
                invoice_level["invoice_date"].ge(cutoff - pd.Timedelta(days=30))
            ]
            .groupby("customer_id", observed=True)["invoice"]
            .size()
            .rename("invoice_count_30d")
        )
        recent_90 = (
            invoice_level.loc[
                invoice_level["invoice_date"].ge(cutoff - pd.Timedelta(days=90))
            ]
            .groupby("customer_id", observed=True)["invoice"]
            .size()
            .rename("invoice_count_90d")
        )
        first_purchase = (
            past.loc[past["customer_id"].isin(customer.index)]
            .groupby("customer_id", observed=True)["invoice_date"]
            .min()
            .rename("first_purchase_datetime")
        )

        customer = customer.join(invoice_features).join(recent_30).join(recent_90)
        customer = customer.join(first_purchase)
        customer[["invoice_count_30d", "invoice_count_90d"]] = customer[
            ["invoice_count_30d", "invoice_count_90d"]
        ].fillna(0)
        customer["recency_days"] = (
            cutoff.normalize() - customer["last_purchase_datetime"].dt.normalize()
        ).dt.days
        customer["customer_history_days"] = (
            cutoff.normalize() - customer["first_purchase_datetime"].dt.normalize()
        ).dt.days

        future_buyers = clean.loc[
            clean["invoice_date"].ge(cutoff)
            & clean["invoice_date"].lt(label_end),
            "customer_id",
        ].unique()
        customer["repurchase_30d"] = customer.index.isin(future_buyers).astype("int8")
        customer["snapshot_date"] = cutoff
        customer["split"] = row.split
        customer["use_for_modeling"] = bool(row.use_for_modeling)
        customer = customer.reset_index()
        parts.append(customer)

    snapshots = pd.concat(parts, ignore_index=True)
    integer_columns = [
        "invoice_frequency_180d",
        "product_diversity_180d",
        "invoice_count_30d",
        "invoice_count_90d",
        "recency_days",
        "customer_history_days",
        "repurchase_30d",
    ]
    snapshots[integer_columns] = snapshots[integer_columns].astype("int64")
    snapshots["monetary_value_180d"] = snapshots["monetary_value_180d"].round(2)
    snapshots["average_order_value_180d"] = snapshots[
        "average_order_value_180d"
    ].round(2)
    return snapshots[
        [
            "snapshot_date",
            "customer_id",
            "split",
            "use_for_modeling",
            "recency_days",
            "invoice_frequency_180d",
            "monetary_value_180d",
            "average_order_value_180d",
            "product_diversity_180d",
            "invoice_count_30d",
            "invoice_count_90d",
            "customer_history_days",
            "repurchase_30d",
            "last_purchase_datetime",
            "first_purchase_datetime",
        ]
    ].sort_values(["snapshot_date", "customer_id"], kind="stable")


def validate_pipeline(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    excluded: pd.DataFrame,
    audit: pd.DataFrame,
    schedule: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Return explicit checks for row reconciliation, windows, and leakage."""
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    add(
        "row_reconciliation",
        len(raw) == len(clean) + len(excluded),
        f"raw={len(raw)}, clean={len(clean)}, excluded={len(excluded)}",
    )
    add(
        "audit_primary_reconciliation",
        int(audit["primary_rows"].sum()) == len(raw),
        f"audit_primary={int(audit['primary_rows'].sum())}, raw={len(raw)}",
    )
    add(
        "clean_unique_rows",
        not clean.duplicated(subset=CANONICAL_COLUMNS).any(),
        "No canonical duplicate remains in clean transactions",
    )
    add(
        "clean_customer_ids_present",
        clean["customer_id"].notna().all(),
        "Every clean row has a Customer ID",
    )
    add(
        "clean_positive_values",
        clean["quantity"].gt(0).all() and clean["unit_price"].gt(0).all(),
        "Every clean row has positive quantity and unit price",
    )
    add(
        "snapshot_keys_unique",
        not snapshots.duplicated(["snapshot_date", "customer_id"]).any(),
        "Each customer appears at most once per cutoff",
    )
    add(
        "snapshot_observation_eligibility",
        snapshots["invoice_frequency_180d"].ge(1).all()
        and snapshots["recency_days"].between(1, config.observation_days).all(),
        "All snapshots have a qualifying purchase in [t-180d, t)",
    )
    add(
        "binary_label",
        snapshots["repurchase_30d"].isin([0, 1]).all(),
        "All repurchase labels are 0 or 1",
    )
    add(
        "complete_label_windows",
        schedule["label_end_exclusive"].max()
        <= raw["invoice_date"].max().normalize() + pd.Timedelta(days=1),
        "Every scheduled label window ends within source coverage",
    )

    model_schedule = schedule.loc[schedule["use_for_modeling"]]
    validation_start = model_schedule.loc[
        model_schedule["split"].eq("validation"), "snapshot_date"
    ].min()
    test_start = model_schedule.loc[
        model_schedule["split"].eq("test"), "snapshot_date"
    ].min()
    train_ok = model_schedule.loc[
        model_schedule["split"].eq("train"), "label_end_exclusive"
    ].le(validation_start).all()
    validation_ok = model_schedule.loc[
        model_schedule["split"].eq("validation"), "label_end_exclusive"
    ].le(test_start).all()
    add(
        "split_label_windows_do_not_cross_boundaries",
        train_ok and validation_ok,
        f"validation_start={validation_start.date()}, test_start={test_start.date()}",
    )
    return pd.DataFrame(checks)
