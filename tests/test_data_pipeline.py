import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.cleaning import CANONICAL_COLUMNS, clean_transactions
from src.data.snapshots import (
    PipelineConfig,
    build_customer_snapshots,
    load_modeling_snapshots,
    make_cutoff_schedule,
    partition_modeling_snapshots,
)
from src.spending.labels import build_labels


def make_rows(rows):
    frame = pd.DataFrame(rows)
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["source_file"] = "test.csv"
    frame["source_row"] = range(2, len(frame) + 2)
    return frame


class CleaningTests(unittest.TestCase):
    def test_overlapping_source_files_deduplicate_exact_lines_only(self):
        row = {
            "invoice": "536365", "stock_code": "85123A",
            "description": "Product", "quantity": 6,
            "invoice_date": "2010-12-01 08:26",
            "unit_price": 2.55, "customer_id": "17850",
            "country": "United Kingdom",
        }
        frame = make_rows([row, row.copy(), {**row, "stock_code": "71053"}])
        frame["source_file"] = ["2009-2010.csv", "2010-2011.csv",
                                "2010-2011.csv"]
        clean, excluded, audit = clean_transactions(frame)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean.stock_code.tolist(), ["71053", "85123A"])
        self.assertEqual(excluded.cleaning_status.tolist(), ["exact_duplicate"])
        self.assertEqual(excluded.source_file.tolist(), ["2010-2011.csv"])
        self.assertEqual(int(audit.loc[audit.rule.eq("exact_duplicate"),
                                       "primary_rows"].iloc[0]), 1)

    def test_cleaning_rules_are_audited_and_reconciled(self):
        valid = {
            "invoice": "100",
            "stock_code": "12345",
            "description": "Product",
            "quantity": 2,
            "invoice_date": "2021-01-01 10:00",
            "unit_price": 3.0,
            "customer_id": "1",
            "country": "UK",
        }
        rows = [
            valid,
            valid.copy(),
            {**valid, "invoice": "101", "customer_id": pd.NA},
            {**valid, "invoice": "C102", "quantity": -1},
            {**valid, "invoice": "103", "unit_price": 0},
            {**valid, "invoice": "104", "stock_code": "POST"},
        ]
        clean, excluded, audit = clean_transactions(make_rows(rows))

        self.assertEqual(len(clean), 1)
        self.assertEqual(len(excluded), 5)
        self.assertEqual(int(audit["primary_rows"].sum()), len(rows))
        primary = dict(zip(audit["rule"], audit["primary_rows"]))
        self.assertEqual(primary["exact_duplicate"], 1)
        self.assertEqual(primary["missing_customer_id"], 1)
        self.assertEqual(primary["cancelled_invoice"], 1)
        self.assertEqual(primary["nonpositive_unit_price"], 1)
        self.assertEqual(primary["non_product_entry"], 1)


class SnapshotTests(unittest.TestCase):
    def test_modeling_loader_uses_canonical_snapshot_columns(self):
        row = {
            "snapshot_date": "2021-07-01",
            "customer_id": "00123",
            "split": "validation",
            "use_for_modeling": "TRUE",
            "recency_days": 4,
            "invoice_frequency_180d": 3,
            "monetary_value_180d": 42.5,
            "average_order_value_180d": 14.17,
            "product_diversity_180d": 2,
            "invoice_count_30d": 1,
            "invoice_count_90d": 2,
            "repurchase_30d": 1,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            loaded = load_modeling_snapshots(path)

        self.assertEqual(loaded.loc[0, "CustomerID"], "00123")
        self.assertEqual(loaded.loc[0, "frequency"], 3)
        self.assertEqual(loaded.loc[0, "repurchase"], 1)
        self.assertEqual(loaded.loc[0, "split"], "validation")
        self.assertTrue(bool(loaded.loc[0, "use_for_modeling"]))
        self.assertEqual(loaded.loc[0, "cutoff_date"], pd.Timestamp("2021-07-01"))

    def test_modeling_partitions_exclude_purged_snapshots(self):
        snapshots = pd.DataFrame({
            "cutoff_date": pd.to_datetime([
                "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"
            ]),
            "split": ["train", "purged_before_validation", "validation", "test"],
            "use_for_modeling": [True, False, True, True],
        })
        train, validation, test = partition_modeling_snapshots(snapshots)
        self.assertEqual([len(train), len(validation), len(test)], [1, 1, 1])
        self.assertEqual(train.iloc[0]["cutoff_date"], pd.Timestamp("2021-01-01"))

    def test_feature_and_label_windows_respect_cutoff(self):
        cutoff = pd.Timestamp("2021-07-01")
        config = PipelineConfig()
        rows = [
            {
                "invoice": "A",
                "stock_code": "10001",
                "description": "One",
                "quantity": 1,
                "invoice_date": cutoff - pd.Timedelta(days=180),
                "unit_price": 10.0,
                "customer_id": "1",
                "country": "UK",
                "line_amount": 10.0,
            },
            {
                "invoice": "B",
                "stock_code": "10002",
                "description": "Two",
                "quantity": 2,
                "invoice_date": cutoff - pd.Timedelta(hours=1),
                "unit_price": 5.0,
                "customer_id": "1",
                "country": "UK",
                "line_amount": 10.0,
            },
            {
                "invoice": "C",
                "stock_code": "10003",
                "description": "Future",
                "quantity": 1,
                "invoice_date": cutoff,
                "unit_price": 99.0,
                "customer_id": "1",
                "country": "UK",
                "line_amount": 99.0,
            },
            {
                "invoice": "D",
                "stock_code": "10004",
                "description": "Too old",
                "quantity": 1,
                "invoice_date": cutoff - pd.Timedelta(days=180, seconds=1),
                "unit_price": 7.0,
                "customer_id": "2",
                "country": "UK",
                "line_amount": 7.0,
            },
        ]
        clean = make_rows(rows)
        clean["line_amount"] = [10.0, 10.0, 99.0, 7.0]
        schedule = make_cutoff_schedule(
            [cutoff, cutoff + pd.offsets.MonthBegin(1), cutoff + pd.offsets.MonthBegin(2)],
            config,
        ).iloc[[0]]
        schedule.loc[:, "split"] = "train"
        schedule.loc[:, "use_for_modeling"] = True
        snapshots = build_customer_snapshots(clean, schedule, config)

        self.assertEqual(list(snapshots["customer_id"]), ["1"])
        row = snapshots.iloc[0]
        self.assertEqual(row["invoice_frequency_180d"], 2)
        self.assertEqual(row["monetary_value_180d"], 20.0)
        self.assertEqual(row["product_diversity_180d"], 2)
        self.assertEqual(row["recency_days"], 1)
        self.assertEqual(row["repurchase_30d"], 1)

    def test_split_schedule_purges_cross_boundary_labels(self):
        config = PipelineConfig(label_days=30, train_fraction=1 / 3, validation_fraction=1 / 3)
        cutoffs = pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"])
        schedule = make_cutoff_schedule(cutoffs, config)

        february = schedule.loc[schedule["snapshot_date"].eq(pd.Timestamp("2021-02-01"))].iloc[0]
        self.assertEqual(february["split"], "purged_before_test")
        self.assertFalse(bool(february["use_for_modeling"]))


class SpendLabelTests(unittest.TestCase):
    def test_spend_label_uses_exact_future_window_and_existing_cohort(self):
        snapshots = pd.DataFrame({
            'snapshot_date': ['2021-01-01', '2021-01-01'],
            'customer_id': ['1', '2'], 'split': ['train', 'train'],
            'use_for_modeling': [True, True], 'repurchase_30d': [1, 0],
        })
        schedule = pd.DataFrame({
            'snapshot_date': ['2021-01-01'],
            'label_end_exclusive': ['2021-01-31'],
        })
        clean = pd.DataFrame({
            'customer_id': ['1', '1', '1', '3'],
            'invoice_date': ['2020-12-31 23:59', '2021-01-01 00:00',
                             '2021-01-31 00:00', '2021-01-02 00:00'],
            'line_amount': [7.0, 12.35, 99.0, 50.0],
        })
        labels, audit = build_labels(snapshots, clean, schedule)
        self.assertEqual(labels.spend_30d.tolist(), [12.35, 0.0])
        self.assertEqual(audit.loc[0, 'cohort_future_spend'], 12.35)
        self.assertEqual(audit.loc[0, 'all_future_spend'], 62.35)
        self.assertTrue(bool(audit.loc[0, 'reconciled']))

    def test_spend_label_rejects_mismatch_with_repurchase(self):
        snapshots = pd.DataFrame({
            'snapshot_date': ['2021-01-01'], 'customer_id': ['1'],
            'split': ['train'], 'use_for_modeling': [True],
            'repurchase_30d': [0],
        })
        schedule = pd.DataFrame({
            'snapshot_date': ['2021-01-01'],
            'label_end_exclusive': ['2021-01-31'],
        })
        clean = pd.DataFrame({
            'customer_id': ['1'], 'invoice_date': ['2021-01-02'],
            'line_amount': [1.0],
        })
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            build_labels(snapshots, clean, schedule)


if __name__ == "__main__":
    unittest.main()
