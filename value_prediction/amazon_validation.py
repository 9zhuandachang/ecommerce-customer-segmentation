"""Locked cross-platform check on the public Amazon purchase histories.

The raw Kaggle mirror stays in the OS temporary directory. Only aggregate
metrics, source hashes, and cutoff audits are written to the repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.features import PipelineConfig, build_customer_snapshots, generate_monthly_cutoffs
from value_prediction.conditional_amount import (
    conditional_metrics, fit_predict as predict_conditional,
)
from value_prediction.experiments import TARGET, evaluate
from value_prediction.external_validation import score_method
from value_prediction.labels import build_labels, sha256
from value_prediction.optimization import fit_rows_for_date, score_with_group_bias
from value_prediction.temporal_features import build_temporal_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/amazon_validation'
PROTOCOL = ROOT / 'value_prediction/amazon_protocol.md'
SOURCE_URL = ('https://www.kaggle.com/api/v1/datasets/download/'
              'dharshinisraghunath/harvard-ecommerce-dataset-for-big-data-analysis')
SOURCE = Path(tempfile.gettempdir()) / 'ecommerce_spend_external/amazon_purchases.zip'
SOURCE_SHA256 = '9ca53a81e59b1d3bad9946b57fd60b067135d0adb217127efa23a900a0f1a441'
CSV_NAME = 'amazon-purchases.csv'
SOURCE_COLUMNS = [
    'Survey ResponseID', 'Order Date', 'Purchase Price Per Unit',
    'Quantity', 'ASIN/ISBN (Product Code)',
]
METHODS = ('historic_180d', 'recent_90d', 'extended_tweedie')
CONDITIONAL_METHODS = (
    'historic_monthly_rate', 'average_prior_invoice', 'conditional_l1_median',
)
END_EXCLUSIVE = pd.Timestamp('2022-10-01')
TEST_DATES = pd.date_range('2022-07-01', '2022-09-01', freq='MS')
VALIDATION_DATES = pd.date_range('2022-04-01', '2022-06-01', freq='MS')


def source_path() -> Path:
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SOURCE.with_suffix('.download')
        try:
            with urlopen(SOURCE_URL, timeout=180) as response, temporary.open('wb') as dest:
                shutil.copyfileobj(response, dest)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('Amazon mirror source hash mismatch')
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('Amazon mirror source hash mismatch')
    return SOURCE


def archive_member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def adapt_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(SOURCE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing Amazon source fields: {sorted(missing)}')
    dates = pd.to_datetime(raw['Order Date'], errors='coerce')
    price = pd.to_numeric(raw['Purchase Price Per Unit'], errors='coerce')
    quantity = pd.to_numeric(raw['Quantity'], errors='coerce')
    amount_cents = np.rint(price * quantity * 100)
    valid = raw[SOURCE_COLUMNS].notna().all(axis=1) & dates.notna()
    valid &= price.gt(0) & quantity.gt(0) & np.isfinite(price) & np.isfinite(quantity)
    valid &= amount_cents.gt(0) & np.isfinite(amount_cents)
    customer_id = raw.loc[valid, 'Survey ResponseID'].astype('string')
    # The release has no order number. A source row is the published purchase unit.
    source_row = pd.Series(raw.index, index=raw.index).loc[valid].astype('string')
    clean = pd.DataFrame({
        'customer_id': customer_id,
        'invoice': customer_id + '/' + source_row,
        'stock_code': raw.loc[valid, 'ASIN/ISBN (Product Code)'].astype('string'),
        'invoice_date': dates.loc[valid],
        'line_amount': amount_cents.loc[valid].astype('int64') / 100,
    }).reset_index(drop=True)
    if clean.empty or clean.isna().any().any():
        raise ValueError('Amazon clean transactions are empty or incomplete')
    return clean, {
        'source_rows': int(len(raw)), 'eligible_rows': int(len(clean)),
        'excluded_rows': int((~valid).sum()),
        'missing_required_rows': int(raw[SOURCE_COLUMNS].isna().any(axis=1).sum()),
        'invalid_date_rows': int(dates.isna().sum()),
        'nonpositive_price_rows': int(price.le(0).sum()),
        'nonpositive_quantity_rows': int(quantity.le(0).sum()),
        'eligible_amount_cents': int(amount_cents.loc[valid].sum()),
        'eligible_customers': int(clean.customer_id.nunique()),
        'source_rows_at_or_after_locked_label_end': int(
            dates.ge(END_EXCLUSIVE).sum()),
        'source_rows_after_2023_03_31': int(
            dates.gt(pd.Timestamp('2023-03-31')).sum()),
        'pseudo_invoice_is_source_row': True,
    }


def build_schedule(clean: pd.DataFrame) -> pd.DataFrame:
    if clean.invoice_date.min() > pd.Timestamp('2018-01-31'):
        raise ValueError('Amazon mirror does not cover the declared 2018 start')
    if clean.invoice_date.max() < END_EXCLUSIVE:
        raise ValueError('Amazon mirror does not cover all locked labels')
    config = PipelineConfig(observation_days=180, label_days=30)
    dates = generate_monthly_cutoffs(clean.invoice_date.min(),
                                     END_EXCLUSIVE, config)
    dates = dates[dates <= TEST_DATES.max()]
    if (len(dates) < 9 or list(dates[-3:]) != list(TEST_DATES)
            or list(dates[-6:-3]) != list(VALIDATION_DATES)):
        raise ValueError('Amazon cutoff calendar differs from locked protocol')
    schedule = pd.DataFrame({'snapshot_date': dates})
    schedule['split'] = (['train'] * (len(dates) - 6)
                         + ['validation'] * 3 + ['test'] * 3)
    schedule['feature_start'] = schedule.snapshot_date - pd.Timedelta(days=180)
    schedule['feature_end_exclusive'] = schedule.snapshot_date
    schedule['label_start'] = schedule.snapshot_date
    schedule['label_end_exclusive'] = schedule.snapshot_date + pd.Timedelta(days=30)
    schedule['use_for_modeling'] = True
    if schedule.label_end_exclusive.max() > END_EXCLUSIVE:
        raise ValueError('Locked Amazon label window extends beyond coverage')
    return schedule


def main() -> None:
    source = source_path()
    with zipfile.ZipFile(source) as archive:
        if CSV_NAME not in archive.namelist():
            raise ValueError(f'Missing Amazon purchases file: {archive.namelist()}')
        csv_hash = archive_member_sha256(archive, CSV_NAME)
        with archive.open(CSV_NAME) as stream:
            raw = pd.read_csv(
                stream, usecols=SOURCE_COLUMNS,
                dtype={'Survey ResponseID': 'string',
                       'ASIN/ISBN (Product Code)': 'string'},
                low_memory=False)
    clean, audit = adapt_transactions(raw)
    del raw
    schedule = build_schedule(clean)
    print(f'Amazon clean rows={len(clean)} cutoffs={len(schedule)}', flush=True)
    snapshots = build_customer_snapshots(clean, schedule, PipelineConfig())
    print(f'Amazon snapshots={len(snapshots)}', flush=True)
    labels, label_audit = build_labels(snapshots, clean, schedule)
    temporal = build_temporal_features(snapshots, clean, schedule)
    data = snapshots.merge(labels[['snapshot_date', 'customer_id', TARGET]],
                           on=['snapshot_date', 'customer_id'], validate='one_to_one')
    data = data.merge(temporal, on=['snapshot_date', 'customer_id'],
                      validate='one_to_one')
    if len(data) != len(snapshots) or not label_audit.reconciled.all():
        raise ValueError('Amazon snapshot or spend reconciliation failed')
    feasibility_only = bool(any(
        int(label_audit.loc[label_audit.snapshot_date.eq(date),
                            'positive_snapshots'].iloc[0]) < 100
        or int(snapshots.snapshot_date.eq(date).sum()) < 1000
        for date in TEST_DATES))

    monthly, test_scores, conditional_monthly, conditional_test = [], [], [], []
    for row in schedule.loc[schedule.split.isin(['validation', 'test'])].itertuples():
        date = row.snapshot_date
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No Amazon customer cohort at {date.date()}')
        for method in METHODS:
            scores = score_method(method, fit, holdout)
            if not np.isfinite(scores).all() or (scores < 0).any():
                raise ValueError(f'Invalid Amazon forecast: {method}')
            metric = score_with_group_bias(holdout, scores)
            monthly.append({
                'snapshot_date': date.strftime('%Y-%m-%d'), 'split': row.split,
                'method': method, 'fit_rows': len(fit),
                'latest_fit_label_end': (
                    fit.snapshot_date.max() + pd.Timedelta(days=30)
                ).strftime('%Y-%m-%d'),
                'zero_spend_share': float(holdout[TARGET].eq(0).mean()),
                **metric,
            })
            if row.split == 'test':
                part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
                part['method'] = method
                part['score'] = scores
                test_scores.append(part)
        for method in CONDITIONAL_METHODS:
            amount = predict_conditional(method, fit, holdout)
            result = conditional_metrics(holdout[TARGET].to_numpy(dtype=float), amount)
            conditional_monthly.append({
                'snapshot_date': date.strftime('%Y-%m-%d'), 'split': row.split,
                'method': method, **result,
            })
            if row.split == 'test':
                part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
                part['method'] = method
                part['predicted_if_purchase'] = amount
                conditional_test.append(part)
        print(f'completed Amazon cutoff {date.date()} ({row.split})', flush=True)

    monthly = pd.DataFrame(monthly)
    scores = pd.concat(test_scores, ignore_index=True)
    pooled = pd.DataFrame([
        {'method': method, 'split': 'test_pooled',
         **score_with_group_bias(part, part.score.to_numpy(dtype=float))}
        for method in METHODS
        for part in [scores.loc[scores.method.eq(method)].copy()]
    ])
    conditional_monthly = pd.DataFrame(conditional_monthly)
    conditional_scores = pd.concat(conditional_test, ignore_index=True)
    conditional_pooled = pd.DataFrame([
        {'method': method, 'split': 'test_pooled',
         **conditional_metrics(part[TARGET].to_numpy(dtype=float),
                               part.predicted_if_purchase.to_numpy(dtype=float))}
        for method in CONDITIONAL_METHODS
        for part in [conditional_scores.loc[conditional_scores.method.eq(method)]]
    ])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, table in [
        ('monthly_metrics.csv', monthly), ('pooled_test_metrics.csv', pooled),
        ('conditional_monthly.csv', conditional_monthly),
        ('conditional_pooled_test.csv', conditional_pooled),
    ]:
        table.to_csv(OUTPUT / filename, index=False, encoding='utf-8-sig')
    manifest = {
        'status': ('cross_platform_feasibility_only' if feasibility_only
                   else 'locked_cross_platform_test_not_original_shop_acceptance'),
        'publisher': 'Open e-commerce 1.0, Harvard Dataverse',
        'mirror_url': SOURCE_URL,
        'source_zip_sha256': sha256(source), 'source_csv_sha256': csv_hash,
        'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__)),
        'source_audit': audit,
        'source_date_min': clean.invoice_date.min().isoformat(),
        'source_date_max': clean.invoice_date.max().isoformat(),
        'snapshot_count': len(snapshots),
        'label_reconciled': bool(label_audit.reconciled.all()),
        'minimum_test_cohort_met': not feasibility_only,
        'schedule': [
            {'cutoff': row.snapshot_date.strftime('%Y-%m-%d'), 'split': row.split,
             'snapshot_count': int(snapshots.snapshot_date.eq(row.snapshot_date).sum()),
             'positive_count': int(label_audit.loc[
                 label_audit.snapshot_date.eq(row.snapshot_date),
                 'positive_snapshots'].iloc[0])}
            for row in schedule.itertuples(index=False)
        ],
        'raw_transactions_saved_in_repository': False,
        'customer_predictions_saved_in_repository': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Amazon purchase history: fixed cross-platform timing verification', '',
        'The source is [Open e-commerce 1.0 ](https://doi.org/10.7910/DVN/YGLYDY),'
        'Read from the public Kaggle mirror; execution details are written to `amazon_protocol.md` first.',
        f'Source file line {audit["source_rows"]:,}, qualifying purchase record'
        f'{audit["eligible_rows"]:,} row, customer {audit["eligible_customers"]:,} name;'
        'Raw row and customer level forecasts are not written to the warehouse.',
        'This issuance document does not have an order ID, and the purchase record frequency is not equal to the original merchant invoice frequency.'
        'Participants volunteered to share history, and the sample selection differed from the original project.',
        f'There are rows {audit["source_rows_after_2023_03_31"]} in the image with dates later than'
        '2023 - 03 - 31; the last tag of the locked model ends at 2022 - 10 - 01,'
        'These late dates are not entered into inputs or labels.',
        f'The lower limit of the test month sample is: {"Feasibility analysis only" if feasibility_only else "Meet plan requirements"}.',
        '', '| Cutoff | Partition | Method | MAE | Buyers MAE | Top 20 Coverage | Total Deviation | Top 20 Population Deviation |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in monthly.itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.split} | {row.method} | '
                     f'{row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | '
                     f'{row.top20_group_bias:+.2%} |')
    lines += ['', 'The last three test months combined:', '',
              '| Method | MAE | Buyers MAE | Top 20 Coverage | Total Bias | Top 20 Group Bias |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in pooled.itertuples(index=False):
        lines.append(f'| {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | '
                     f'{row.top20_group_bias:+.2%} |')
    lines += ['', 'If the purchase is indeed made 30 days in the future, the buyer error of the conditional amount model.'
              'The historical monthly amount rate is a stronger comparison added after the results,'
              'Avoid mistaking the line-by-line purchase average price for a competitive monthly amount baseline:', '',
              '| Method | Buyer MAE | Buyer Amount Deviation |',
              '| --- | ---: | ---: |']
    for row in conditional_pooled.itertuples(index=False):
        lines.append(f'| {row.method} | {row.buyer_mae:.2f} | '
                     f'{row.buyer_amount_bias:+.2%} |')
    lines += ['', 'This is evidence of the robustness of another sample of platform voluntary participants.'
              'It does not replace the original UK merchant\'s acceptance in the new month, nor does it prove marketing benefits.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote Amazon cross-platform report to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
