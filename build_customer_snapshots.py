"""CLI and artifact writer for the canonical customer snapshot pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data_cleaning import clean_transactions, load_transactions
from src.features import (
    PipelineConfig,
    build_customer_snapshots,
    generate_monthly_cutoffs,
    make_cutoff_schedule,
    validate_pipeline,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")


def run_pipeline(
    input_paths: Sequence[Path], output_dir: Path, config: PipelineConfig
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_transactions(input_paths)
    clean, excluded, audit = clean_transactions(raw)
    cutoffs = generate_monthly_cutoffs(
        raw["invoice_date"].min(), raw["invoice_date"].max(), config
    )
    schedule = make_cutoff_schedule(cutoffs, config)
    snapshots = build_customer_snapshots(clean, schedule, config)
    validation = validate_pipeline(
        raw, clean, excluded, audit, schedule, snapshots, config
    )

    _write_csv(clean, output_dir / "clean_transactions.csv")
    _write_csv(excluded, output_dir / "excluded_transactions.csv")
    _write_csv(audit, output_dir / "cleaning_audit.csv")
    _write_csv(schedule, output_dir / "cutoff_schedule.csv")
    _write_csv(snapshots, output_dir / "customer_snapshots.csv")
    _write_csv(validation, output_dir / "validation_report.csv")

    manifest: dict[str, object] = {
        "config": {
            "observation_days": config.observation_days,
            "label_days": config.label_days,
            "cutoff_frequency": "month_start",
            "feature_interval": "[t-180d, t)",
            "label_interval": "[t, t+30d)",
        },
        "sources": [
            {
                "file": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "counts": {
            "raw_rows": len(raw),
            "clean_rows": len(clean),
            "excluded_rows": len(excluded),
            "snapshot_rows": len(snapshots),
            "cutoff_dates": len(schedule),
        },
        "date_range": {
            "source_min": raw["invoice_date"].min().isoformat(),
            "source_max": raw["invoice_date"].max().isoformat(),
            "first_cutoff": schedule["snapshot_date"].min().date().isoformat(),
            "last_cutoff": schedule["snapshot_date"].max().date().isoformat(),
        },
        "validation_passed": bool(validation["passed"].all()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Online Retail II data and build monthly customer snapshots."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Source CSV files in order")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/customer_snapshot_pipeline"),
        help="Directory for generated CSV and JSON outputs",
    )
    parser.add_argument("--observation-days", type=int, default=180)
    parser.add_argument("--label-days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        observation_days=args.observation_days,
        label_days=args.label_days,
    )
    manifest = run_pipeline(args.inputs, args.output_dir, config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
