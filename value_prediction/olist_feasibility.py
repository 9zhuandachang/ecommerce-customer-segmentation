"""Audit Olist's precommitted cohort before any model scoring."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features import PipelineConfig, build_customer_snapshots
from value_prediction.labels import build_labels, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/olist_feasibility'
SOURCE_URL = 'https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce'
SOURCE_SHA256 = '967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784'
SOURCE = Path(tempfile.gettempdir()) / 'ecommerce_spend_external/olist_official.zip'
FIRST_CUTOFF = pd.Timestamp('2017-04-01')
LAST_CUTOFF = pd.Timestamp('2018-08-01')
TEST_DATES = pd.date_range('2018-06-01', '2018-08-01', freq='MS')


def source_path() -> Path:
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        temporary = SOURCE.with_suffix('.download')
        try:
            with requests.get(SOURCE_URL, timeout=90, stream=True) as response:
                response.raise_for_status()
                with temporary.open('wb') as dest:
                    for block in response.iter_content(65536):
                        dest.write(block)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('Olist source hash mismatch')
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('Olist source hash mismatch')
    return SOURCE


def adapt_sources(orders: pd.DataFrame, customers: pd.DataFrame,
                  items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use order placement and item price, never future fulfillment state."""
    if (orders.order_id.duplicated().any()
            or customers.customer_id.duplicated().any()
            or items.duplicated(['order_id', 'order_item_id']).any()):
        raise ValueError('Olist primary key collision')
    required_orders = {'order_id', 'customer_id', 'order_status',
                       'order_purchase_timestamp'}
    required_customers = {'customer_id', 'customer_unique_id'}
    required_items = {'order_id', 'order_item_id', 'product_id', 'price'}
    for table, required in [(orders, required_orders), (customers, required_customers),
                            (items, required_items)]:
        if not required.issubset(table.columns):
            raise ValueError(f'Missing Olist fields: {sorted(required - set(table.columns))}')
    if (orders[list(required_orders)].isna().any().any()
            or customers[list(required_customers)].isna().any().any()
            or items[list(required_items)].isna().any().any()):
        raise ValueError('Olist required fields contain missing values')
    order_customer = orders.merge(customers, on='customer_id', how='left',
                                  validate='many_to_one', indicator=True)
    if not order_customer._merge.eq('both').all():
        raise ValueError('Olist order lacks stable customer ID')
    joined = items.merge(order_customer.drop(columns='_merge'), on='order_id',
                         how='left', validate='many_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Olist item lacks order/customer')
    joined = joined.drop(columns='_merge')
    joined['invoice_date'] = pd.to_datetime(
        joined.order_purchase_timestamp, errors='raise')
    amount = pd.to_numeric(joined.price, errors='raise').to_numpy(dtype=float)
    cents = np.rint(amount * 100)
    valid = np.isfinite(amount) & (amount > 0) & np.isclose(
        amount * 100, cents, rtol=0, atol=1e-6)
    clean = joined.loc[valid].copy()
    clean['line_amount'] = cents[valid].astype('int64') / 100
    clean = clean.rename(columns={
        'customer_unique_id': 'customer_id_canonical',
        'order_id': 'invoice', 'product_id': 'stock_code',
    })
    clean['customer_id'] = clean.customer_id_canonical.astype('string')
    if clean.invoice.groupby(clean.customer_id).nunique().sum() != clean.invoice.nunique():
        raise ValueError('Olist order is associated with multiple stable customers')
    result = clean[['customer_id', 'invoice', 'stock_code', 'invoice_date',
                    'line_amount']]
    if result[['customer_id', 'invoice', 'stock_code', 'invoice_date']].isna().any().any():
        raise ValueError('Olist adapted transaction field is missing')
    status_counts = orders.order_status.value_counts(dropna=False).to_dict()
    return result, {
        'source_orders': len(orders), 'source_customers': len(customers),
        'source_items': len(items), 'eligible_items': len(result),
        'excluded_invalid_price_items': int((~valid).sum()),
        'eligible_orders': int(result.invoice.nunique()),
        'eligible_stable_customers': int(result.customer_id.nunique()),
        'status_order_counts': {str(k): int(v) for k, v in status_counts.items()},
        'source_valid_item_amount_cents': int(cents[valid].sum()),
        'adapted_amount_cents': int(np.rint(result.line_amount.sum() * 100)),
    }


def read_source(path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        expected = {'olist_orders_dataset.csv', 'olist_customers_dataset.csv',
                    'olist_order_items_dataset.csv'}
        if not expected.issubset(names):
            raise ValueError('Olist source tables are incomplete')
        with archive.open('olist_orders_dataset.csv') as member:
            orders = pd.read_csv(member, usecols=[
                'order_id', 'customer_id', 'order_status',
                'order_purchase_timestamp'], dtype='string')
        with archive.open('olist_customers_dataset.csv') as member:
            customers = pd.read_csv(member, usecols=[
                'customer_id', 'customer_unique_id'], dtype='string')
        with archive.open('olist_order_items_dataset.csv') as member:
            items = pd.read_csv(member, usecols=[
                'order_id', 'order_item_id', 'product_id', 'price'],
                dtype={'order_id': 'string', 'order_item_id': 'Int64',
                       'product_id': 'string'})
    return adapt_sources(orders, customers, items)


def build_schedule(clean: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(FIRST_CUTOFF, LAST_CUTOFF, freq='MS')
    if (clean.invoice_date.min().normalize() > FIRST_CUTOFF - pd.Timedelta(days=180)
            or clean.invoice_date.max().normalize()
            < LAST_CUTOFF + pd.Timedelta(days=30) - pd.Timedelta(days=1)):
        raise ValueError('Olist source does not cover precommitted complete windows')
    schedule = pd.DataFrame({'snapshot_date': dates})
    schedule['split'] = np.select(
        [schedule.snapshot_date.ge(pd.Timestamp('2018-06-01')),
         schedule.snapshot_date.ge(pd.Timestamp('2018-03-01'))],
        ['locked_test', 'development'], default='history')
    schedule['feature_start'] = schedule.snapshot_date - pd.Timedelta(days=180)
    schedule['feature_end_exclusive'] = schedule.snapshot_date
    schedule['label_start'] = schedule.snapshot_date
    schedule['label_end_exclusive'] = schedule.snapshot_date + pd.Timedelta(days=30)
    schedule['use_for_modeling'] = True
    return schedule


def main() -> None:
    source = source_path()
    clean, source_audit = read_source(source)
    if source_audit['source_valid_item_amount_cents'] != source_audit['adapted_amount_cents']:
        raise AssertionError('Olist positive item amount does not reconcile')
    schedule = build_schedule(clean)
    snapshot = build_customer_snapshots(clean, schedule, PipelineConfig())
    labels, label_audit = build_labels(snapshot, clean, schedule)
    if len(labels) != len(snapshot) or not label_audit.reconciled.all():
        raise AssertionError('Olist labels do not reconcile')
    feasibility = label_audit[['snapshot_date', 'label_end_exclusive',
                               'cohort_snapshots', 'positive_snapshots',
                               'cohort_future_spend', 'reconciled']].copy()
    feasibility['split'] = feasibility.snapshot_date.map(
        schedule.set_index('snapshot_date').split)
    feasibility['eligible_for_fixed_test'] = (
        feasibility.cohort_snapshots.ge(1000)
        & feasibility.positive_snapshots.ge(100))
    test = feasibility.loc[feasibility.snapshot_date.isin(TEST_DATES)]
    if len(test) != 3:
        raise AssertionError('Olist locked test does not have three months')
    passed = bool(test.eligible_for_fixed_test.all())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    feasibility.to_csv(OUTPUT / 'cutoff_feasibility.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    manifest = {
        'status': 'feasibility_only_no_model_scoring',
        'source_url': SOURCE_URL, 'source_sha256': sha256(source),
        'protocol_sha256': sha256(ROOT / 'value_prediction/olist_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'source_audit': source_audit,
        'date_min': clean.invoice_date.min().isoformat(),
        'date_max': clean.invoice_date.max().isoformat(),
        'snapshot_count': len(snapshot),
        'all_labels_reconciled': bool(label_audit.reconciled.all()),
        'locked_test_feasible': passed,
        'model_scores_computed': False,
        'raw_or_customer_predictions_saved_in_repository': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Olist cross-merchant fixed window: source and sample size audit', '',
        'The anonymous orders officially disclosed by Olist are only used for the feasibility check of this pre-fixed window.'
        'The amount is the total price of the ordered goods, excluding shipping costs; it is not filtered according to the final performance status to avoid introducing post-order information into pre-cutoff characteristics.',
        f'Source order {source_audit["source_orders"]:,}, product line {source_audit["source_items"]:,};'
        f'Qualified orders {source_audit["eligible_orders"]:,}, stable customers {source_audit["eligible_stable_customers"]:,};'
        f'Customer monthly snapshot {len(snapshot):,}. The product amount and future tags are verified and approved.', '',
        '| Locking cutoff | Number of snapshots | Future buyers | Whether the monthly 1,000 / 100 threshold is reached |',
        '| --- | ---: | ---: | --- |',
    ]
    for row in test.itertuples(index=False):
        lines.append(f'| {row.snapshot_date:%Y-%m-%d} | '
                     f'{row.cohort_snapshots:,} | {row.positive_snapshots:,} | '
                     f'{"Yes" if row.eligible_for_fixed_test else "No"} |')
    lines += ['', f'Three-month feasibility: **{"satisfy" if passed else "dissatisfied"}**.'
              'According to ex-ante rules, if the threshold is lower than the threshold in any month, the candidate model acceptance indicators will not be calculated.',
              'The original files are only in the system temporary directory; the source hash, status distribution and monthly audit are in the same directory.'
              'This cross-merchant source also does not replace the original UK merchant\'s independent acceptance of the new month.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Olist locked test feasible={passed}; output={OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
