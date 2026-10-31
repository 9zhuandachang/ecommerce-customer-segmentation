"""Locked, aggregate-only cross-shop check on publisher's jewelry transactions.

Run ``python -m archive.spending.jewelry_validation``. Raw rows stay in the
operating-system temporary directory. The protocol predates score inspection.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.data.snapshots import PipelineConfig, build_customer_snapshots, generate_monthly_cutoffs
from archive.spending.experiments import TARGET, evaluate
from archive.spending.external_validation import score_method
from src.spending.labels import build_labels, sha256
from archive.spending.optimization import fit_rows_for_date
from src.spending.features import build_temporal_features

from src.paths import ROOT
OUTPUT = ROOT / 'archive/artifacts/spending/jewelry_validation'
PROTOCOL = ROOT / 'archive/docs/spending/jewelry_protocol.md'
SOURCE_URL = ('https://www.kaggle.com/api/v1/datasets/download/'
              'mkechinov/ecommerce-purchase-history-from-jewelry-store')
SOURCE = Path(tempfile.gettempdir()) / 'ecommerce_spend_external/jewelry_transactions.zip'
SOURCE_SHA256 = '43c1a4e0b2afed6873f5efe31c8327e3cdfed91dd357b9508ae2ae312c6f5079'
SOURCE_COLUMNS = ['event_time', 'order_id', 'product_id', 'quantity',
                  'category_id', 'category_code', 'brand', 'price', 'user_id',
                  'gender', 'color', 'metal', 'gem']
ID_COLUMNS = {'order_id': 'string', 'product_id': 'string', 'user_id': 'string'}
METHODS = ('historic_180d', 'recent_90d', 'extended_tweedie')


def source_path() -> Path:
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SOURCE.with_suffix('.download')
        try:
            with urlopen(SOURCE_URL, timeout=90) as response, temporary.open('wb') as dest:
                shutil.copyfileobj(response, dest)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('Jewelry source hash mismatch')
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('Jewelry source hash mismatch')
    return SOURCE


def adapt_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the predeclared field mapping and positive-sale eligibility rule."""
    required = ['event_time', 'order_id', 'product_id', 'quantity', 'price', 'user_id']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing jewelry columns: {sorted(missing)}')
    date = pd.to_datetime(raw.event_time, errors='coerce', utc=True)
    price = pd.to_numeric(raw.price, errors='coerce')
    quantity = pd.to_numeric(raw.quantity, errors='coerce')
    valid = raw[required].notna().all(axis=1) & date.notna()
    valid &= price.gt(0) & quantity.gt(0)
    valid &= np.isfinite(price) & np.isfinite(quantity)
    amount_cents = np.rint(price * quantity * 100)
    valid &= amount_cents.gt(0) & np.isfinite(amount_cents)
    clean = pd.DataFrame({
        'customer_id': raw.loc[valid, 'user_id'].astype('string'),
        'invoice': (raw.loc[valid, 'user_id'].astype('string') + '/' +
                    raw.loc[valid, 'order_id'].astype('string')),
        'stock_code': raw.loc[valid, 'product_id'].astype('string'),
        'invoice_date': date.loc[valid].dt.tz_localize(None),
        'line_amount': amount_cents.loc[valid].astype('int64') / 100,
    }).reset_index(drop=True)
    if clean.empty or clean.isna().any().any():
        raise ValueError('Empty or missing canonical jewelry data')
    return clean, {
        'source_rows': len(raw), 'eligible_rows': len(clean),
        'excluded_rows': int((~valid).sum()),
        'missing_required_rows': int(raw[required].isna().any(axis=1).sum()),
        'nonpositive_price_rows': int(price.le(0).sum()),
        'nonpositive_quantity_rows': int(quantity.le(0).sum()),
        'eligible_amount_cents': int(amount_cents.loc[valid].sum()),
        'eligible_customers': int(clean.customer_id.nunique()),
        'eligible_orders': int(clean.invoice.nunique()),
    }


def build_schedule(clean: pd.DataFrame) -> pd.DataFrame:
    config = PipelineConfig(observation_days=180, label_days=30)
    dates = generate_monthly_cutoffs(clean.invoice_date.min(),
                                     clean.invoice_date.max(), config)
    if len(dates) < 9:
        raise ValueError('Fewer than nine complete jewelry cutoffs')
    schedule = pd.DataFrame({'snapshot_date': dates})
    schedule['split'] = (['train'] * (len(dates) - 6)
                         + ['validation'] * 3 + ['test'] * 3)
    schedule['feature_start'] = schedule.snapshot_date - pd.Timedelta(days=180)
    schedule['feature_end_exclusive'] = schedule.snapshot_date
    schedule['label_start'] = schedule.snapshot_date
    schedule['label_end_exclusive'] = schedule.snapshot_date + pd.Timedelta(days=30)
    schedule['use_for_modeling'] = True
    return schedule


def main() -> None:
    source = source_path()
    with zipfile.ZipFile(source) as archive:
        if archive.namelist() != ['jewelry.csv']:
            raise ValueError(f'Unexpected jewelry zip members: {archive.namelist()}')
        with archive.open('jewelry.csv') as stream:
            raw = pd.read_csv(stream, header=None, names=SOURCE_COLUMNS,
                              dtype=ID_COLUMNS, skip_blank_lines=True,
                              low_memory=False)
    clean, source_audit = adapt_transactions(raw)
    schedule = build_schedule(clean)
    snapshots = build_customer_snapshots(clean, schedule, PipelineConfig())
    labels, label_audit = build_labels(snapshots, clean, schedule)
    temporal = build_temporal_features(snapshots, clean, schedule)
    data = snapshots.merge(labels[['snapshot_date', 'customer_id', TARGET]],
                           on=['snapshot_date', 'customer_id'], validate='one_to_one')
    data = data.merge(temporal, on=['snapshot_date', 'customer_id'],
                      validate='one_to_one')
    if len(data) != len(snapshots) or not label_audit.reconciled.all():
        raise ValueError('Jewelry snapshot or amount reconciliation failed')
    positive_test = label_audit.loc[
        label_audit.snapshot_date.isin(schedule.loc[schedule.split.eq('test'),
                                                   'snapshot_date']),
        'positive_snapshots']
    feasibility_only = bool((positive_test < 100).any())

    metrics, pooled_rows = [], []
    for row in schedule.loc[schedule.split.isin(['validation', 'test'])].itertuples():
        date = row.snapshot_date
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No jewelry snapshots at {date.date()}')
        for method in METHODS:
            score = score_method(method, fit, holdout)
            if not np.isfinite(score).all() or (score < 0).any():
                raise ValueError(f'Invalid jewelry score: {method}')
            result = evaluate(holdout, score)
            metrics.append({
                'snapshot_date': date.strftime('%Y-%m-%d'), 'split': row.split,
                'method': method, 'fit_count': len(fit),
                'latest_fit_label_end': (
                    fit.snapshot_date.max() + pd.Timedelta(days=30)
                ).strftime('%Y-%m-%d'),
                'zero_spend_share': float(holdout[TARGET].eq(0).mean()),
                **result,
            })
            if row.split == 'test':
                part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
                part['score'] = score
                part['method'] = method
                pooled_rows.append(part)
        print(f'completed jewelry cutoff {date.date()} ({row.split})', flush=True)

    monthly = pd.DataFrame(metrics)
    heldout = pd.concat(pooled_rows, ignore_index=True)
    pooled = pd.DataFrame([
        {'method': method, 'split': 'test_pooled',
         **evaluate(part, part.score.to_numpy(dtype=float))}
        for method in METHODS
        for part in [heldout.loc[heldout.method.eq(method)].copy()]
    ])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    pooled.to_csv(OUTPUT / 'pooled_test_metrics.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': ('feasibility_only_due_to_small_test_buyer_cohort' if feasibility_only
                   else 'locked_cross_shop_ecommerce_test_not_original_shop_acceptance'),
        'source_url': SOURCE_URL, 'source_sha256': sha256(source),
        'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__)),
        'source_audit': source_audit,
        'date_min': clean.invoice_date.min().isoformat(),
        'date_max': clean.invoice_date.max().isoformat(),
        'snapshot_count': len(snapshots),
        'label_reconciled': bool(label_audit.reconciled.all()),
        'test_buyer_minimum_100_met': not feasibility_only,
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
        '# Independent e-commerce jewelry transaction timing inspection', '',
        'Fixed scheme: `jewelry_protocol.md`. Source: [REES 46 data catalog]'
        '( https://rees46.com/en/datasets ) published and linked to [Kaggle Jewelry Transaction Data]'
        '(https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-jewelry-store)。',
        f'The actual file covers {clean.invoice_date.min().date()} to {clean.invoice_date.max().date()};'
        f'There are {source_audit["source_rows"]:,} product lines in'
        f'{source_audit["eligible_rows"]:,} strips meet the predetermined cleaning conditions.'
        'The description of approximately two years and 13 in the catalog description is inconsistent with the actual content of the file. The following shall be subject to file audit.',
        'Use the original 8 historical features plus 12 time series features and fixed Tweedie parameters; the original rows are not written to the warehouse.',
        f'Applicability: {"Feasibility analysis only (There are not enough purchasing customers in the test month 100)" if feasibility_only else "The number of purchasing customers in the test month reaches the lower limit of the plan"}.',
        '', '| Cutoff | Partition | Method | MAE | Buyer MAE | Top 20 Coverage | Total Deviation |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in monthly.itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.split} | {row.method} | '
                     f'{row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | {row.aggregate_bias:+.2%} |')
    lines += ['', 'Combined in the last three months (Top 20 selected each month):', '',
              '| Method | MAE | Buyer MAE | Top 20 Coverage | Total Deviation |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in pooled.itertuples(index=False):
        lines.append(f'| {row.method} | {row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | {row.aggregate_bias:+.2%} |')
    base = pooled.set_index('method').loc['historic_180d']
    model = pooled.set_index('method').loc['extended_tweedie']
    lines += [
        '', '##Explanation of results', '',
        f'There are {int(model.snapshot_count):,} customer monthly snapshots during the test period, among which'
        f'{int(model.buyer_count):,} items will be purchased in the next 30 days ('
        f'{model.buyer_count / model.snapshot_count:.2%}）。',
        f'The overall model MAE decreases relative to the historical amount baseline'
        f'{1 - model.mae / base.mae:.2%}; Top 20 coverage improvement'
        f'{(model.top20_spend_capture - base.top20_spend_capture) * 100:.2f} percentage points.'
        'This is a comparison of the same customers sorted by different ratings.',
        f'Buyer MAE only lowers {1 - model.positive_mae / base.positive_mae:.2%};'
        f'The total deviation of the three-month combined forecast is {model.aggregate_bias:+.2%},'
        'The direction of the single-month deviation is also unstable. This set of results cannot be regarded as the amount calibration has been accurate.',
        '', 'This data comes from another merchant and can only provide evidence of robustness across merchants; the original project still does not have independent acceptance data for the new month.',
        '',
    ]
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote jewelry aggregate report to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
