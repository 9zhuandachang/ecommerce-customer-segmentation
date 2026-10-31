"""Locked cross-retailer check using the public Complete Journey transactions.

Run ``python -m value_prediction.external_validation``. Source rows stay in
the operating-system temporary directory; only aggregate results are saved.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import PipelineConfig, build_customer_snapshots, generate_monthly_cutoffs
from stage4.kmeans import FEATURES as BASE_FEATURES
from value_prediction.experiments import TARGET, evaluate
from value_prediction.labels import build_labels, sha256
from value_prediction.optimization import fit_rows_for_date
from value_prediction.temporal_features import FEATURES as EXTRA_FEATURES, build_temporal_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/external_validation'
SOURCE_URL = ('https://raw.githubusercontent.com/bradleyboehmke/'
              'completejourney/master/data/transactions.rds')
SOURCE_SHA256 = '1fa0700033f1e5d9bb6b09e2be063d8d68474d346e95c50f2833e09d083e0007'
SOURCE = Path(tempfile.gettempdir()) / 'ecommerce_spend_external/transactions.rds'


def source_path() -> Path:
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SOURCE.with_suffix('.download')
        try:
            with urlopen(SOURCE_URL, timeout=60) as response, temporary.open('wb') as dest:
                shutil.copyfileobj(response, dest)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('External source hash mismatch')
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('External source hash mismatch')
    return SOURCE


def adapt_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use the cleaning rule locked before downloading external labels."""
    required = ['household_id', 'basket_id', 'product_id',
                'transaction_timestamp', 'sales_value', 'quantity']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing external fields: {sorted(missing)}')
    valid = raw[required].notna().all(axis=1)
    valid &= pd.to_numeric(raw.sales_value, errors='coerce').gt(0)
    valid &= pd.to_numeric(raw.quantity, errors='coerce').gt(0)
    clean = raw.loc[valid, required].copy()
    clean['invoice_date'] = pd.to_datetime(clean.transaction_timestamp, errors='raise')
    clean['customer_id'] = clean.household_id.astype('string')
    clean['invoice'] = (clean.customer_id + '/' + clean.basket_id.astype('string'))
    clean['stock_code'] = clean.product_id.astype('string')
    clean['line_amount'] = pd.to_numeric(clean.sales_value, errors='raise')
    cents = np.rint(clean.line_amount.to_numpy(dtype=float) * 100)
    if not np.allclose(clean.line_amount.to_numpy(dtype=float) * 100,
                       cents, rtol=0, atol=1e-6):
        raise ValueError('External positive amounts are not exact cents')
    if clean[['invoice_date', 'customer_id', 'invoice', 'stock_code']].isna().any().any():
        raise ValueError('Missing adapted external fields')
    return clean[['customer_id', 'invoice', 'stock_code', 'invoice_date',
                  'line_amount']], {
                      'source_rows': len(raw), 'eligible_rows': len(clean),
                      'excluded_rows': int((~valid).sum()),
                  }


def build_schedule(clean: pd.DataFrame) -> pd.DataFrame:
    config = PipelineConfig(observation_days=180, label_days=30)
    dates = generate_monthly_cutoffs(clean.invoice_date.min(),
                                     clean.invoice_date.max(), config)
    if len(dates) < 5:
        raise ValueError('External source has fewer than five complete cutoffs')
    schedule = pd.DataFrame({'snapshot_date': dates})
    schedule['split'] = ['train'] * (len(dates) - 3) + ['validation', 'test', 'test']
    schedule['feature_start'] = schedule.snapshot_date - pd.Timedelta(days=180)
    schedule['feature_end_exclusive'] = schedule.snapshot_date
    schedule['label_start'] = schedule.snapshot_date
    schedule['label_end_exclusive'] = schedule.snapshot_date + pd.Timedelta(days=30)
    schedule['use_for_modeling'] = True
    return schedule


def score_method(method: str, fit: pd.DataFrame,
                 holdout: pd.DataFrame) -> np.ndarray:
    if method == 'historic_180d':
        return holdout.monetary_value_180d.to_numpy(dtype=float) / 6
    if method == 'recent_90d':
        return holdout.monetary_value_90d.to_numpy(dtype=float) / 3
    if method != 'extended_tweedie':
        raise ValueError(f'Unknown external method: {method}')
    model = lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        n_estimators=120, learning_rate=.05, num_leaves=15,
        random_state=42, n_jobs=1, verbose=-1)
    model.fit(fit[BASE_FEATURES + EXTRA_FEATURES], fit[TARGET])
    return np.maximum(model.predict(holdout[BASE_FEATURES + EXTRA_FEATURES]), 0)


def main() -> None:
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError('Install optional dependency: python -m pip install pyreadr') from exc
    source = source_path()
    raw = pyreadr.read_r(str(source))[None]
    clean, audit = adapt_transactions(raw)
    schedule = build_schedule(clean)
    snapshot = build_customer_snapshots(clean, schedule, PipelineConfig())
    labels, label_audit = build_labels(snapshot, clean, schedule)
    temporal = build_temporal_features(snapshot, clean, schedule)
    data = snapshot.merge(labels[['snapshot_date', 'customer_id', TARGET]],
                          on=['snapshot_date', 'customer_id'], validate='one_to_one')
    data = data.merge(temporal, on=['snapshot_date', 'customer_id'],
                      validate='one_to_one')
    if len(data) != len(snapshot) or not label_audit.reconciled.all():
        raise ValueError('External snapshot reconciliation failed')

    rows = []
    for date in schedule.loc[schedule.split.isin(['validation', 'test']),
                             'snapshot_date']:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        for method in ('historic_180d', 'recent_90d', 'extended_tweedie'):
            score = score_method(method, fit, holdout)
            if not np.isfinite(score).all() or (score < 0).any():
                raise ValueError(f'Invalid external score: {method}')
            metric = evaluate(holdout, score)
            rows.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                         'split': holdout.split.iloc[0], 'method': method,
                         'fit_rows': len(fit),
                         'latest_fit_label_end': (
                             fit.snapshot_date.max() + pd.Timedelta(days=30)
                         ).strftime('%Y-%m-%d'),
                         **metric})
        print(f'completed external cutoff {date.date()}', flush=True)
    metrics = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'cross_retailer_validation_not_original_ecommerce_acceptance',
        'source_url': SOURCE_URL, 'source_sha256': sha256(source),
        'protocol_sha256': sha256(ROOT / 'value_prediction/external_validation_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'source_audit': audit,
        'date_min': clean.invoice_date.min().isoformat(),
        'date_max': clean.invoice_date.max().isoformat(),
        'eligible_households': clean.customer_id.nunique(),
        'snapshot_count': len(snapshot),
        'schedule': [
            {'cutoff': row.snapshot_date.strftime('%Y-%m-%d'), 'split': row.split,
             'snapshot_count': int(snapshot.snapshot_date.eq(row.snapshot_date).sum())}
            for row in schedule.itertuples(index=False)
        ],
        'label_reconciled': bool(label_audit.reconciled.all()),
        'raw_transactions_saved_in_repository': False,
        'customer_predictions_saved_in_repository': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Independent Grocery Retail Data Inspection', '',
        'This time, the previously locked `external_validation_protocol.md` will be strictly implemented.',
        'This is data from another retailer, and the unit of amount is different from the original item; the results are only used to test the robustness of the method.',
        'Raw rows and customer-level predictions are not saved; native temporary cache with source hash see `manifest.json`.', '',
        '| Cutoff | Partition | Method | MAE | Buyer MAE | Top 20 Coverage | Total Deviation |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.split} | {row.method} | '
                     f'{row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | {row.aggregate_bias:+.2%} |')
    lines += ['', 'External inspection is not a substitute for subsequent independent acceptance of the original online retail item in new months.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote external aggregate metrics to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
