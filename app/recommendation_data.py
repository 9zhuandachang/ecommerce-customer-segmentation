"""Verify and expose safe next-basket recommendation lists to the dashboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'recommendation/outputs/v1'
MANIFEST = OUTPUT / 'manifest.json'
LOCK = OUTPUT / 'locked_selection.json'
RECOMMENDATIONS = OUTPUT / 'recommendations.csv'
EXPORT_COLUMNS = ['snapshot_date', 'customer_id', 'rank', 'stock_code', 'description',
                  'purchased_in_past_180d', 'recommendation_method', 'reason']
NAMES = {'popular': 'Popular products', 'personal_repeat': 'Personal repeat purchase', 'itemknn': 'ItemKNN',
         'implicit_als': 'Implicit ALS', 'ease': 'EASE'}


def file_hash(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def export_recommendations(rows: pd.DataFrame) -> pd.DataFrame:
    if not set(EXPORT_COLUMNS).issubset(rows.columns):
        raise ValueError('Recommendation list is missing export fields')
    return rows[EXPORT_COLUMNS].sort_values(['snapshot_date', 'customer_id', 'rank'], kind='stable').copy()


def validate_recommendations(rows: pd.DataFrame, method: str) -> None:
    keys = ['snapshot_date', 'customer_id']
    if (set(rows.columns) != set(EXPORT_COLUMNS) or rows.empty or rows.isna().any().any()
            or rows.duplicated(keys + ['rank']).any() or rows.duplicated(keys + ['stock_code']).any()
            or not rows.recommendation_method.eq(method).all()
            or not rows['rank'].between(1, 10).all()):
        raise ValueError('Invalid or unsafe recommendation rows')
    if not rows.groupby(keys).size().eq(10).all():
        raise ValueError('Expected ten distinct products for each customer')
    if not rows.purchased_in_past_180d.isin([True, False]).all():
        raise ValueError('Invalid prior-purchase flag')


def load_recommendations() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if (manifest.get('status') != 'completed_next_basket_comparison'
            or not manifest.get('test_evaluated_after_selection_lock')
            or lock['test_labels_used_for_selection'] or lock['global_test_model_updates']
            or lock['selected_method'] != manifest['selected_method']):
        raise ValueError('Recommendation comparison or selection is inconsistent')
    for name, expected in manifest['output_hashes'].items():
        path = (OUTPUT / name).resolve()
        if path.parent != OUTPUT.resolve() or file_hash(path) != expected:
            raise ValueError(f'Recommendation artifact changed: {name}')
    for name, expected in manifest['code_hashes'].items():
        path = (ROOT / 'recommendation' / name).resolve()
        if path.parent != (ROOT / 'recommendation').resolve() or file_hash(path) != expected:
            raise ValueError(f'Recommendation implementation changed: {name}')
    if lock['model_sha256'] != manifest['output_hashes']['selected_model.joblib']:
        raise ValueError('Selected recommendation model does not match its lock')
    sources = {'transactions': ROOT / 'data/processed/customer_snapshot_pipeline/clean_transactions.csv',
               'snapshots': ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv',
               'schedule': ROOT / 'data/processed/customer_snapshot_pipeline/cutoff_schedule.csv'}
    for name, path in sources.items():
        if file_hash(path) != manifest['source_hashes'][name] or lock['source_hashes'][name] != manifest['source_hashes'][name]:
            raise ValueError(f'Recommendation data source changed: {name}')
    rows = pd.read_csv(RECOMMENDATIONS, dtype={'customer_id': 'string', 'stock_code': 'string'})
    monthly = pd.read_csv(OUTPUT / 'monthly_comparison.csv')
    comparison = pd.read_csv(OUTPUT / 'comparison.csv')
    validate_recommendations(rows, lock['selected_method'])
    if set(rows.snapshot_date) != set(monthly.snapshot_date):
        raise ValueError('Recommendation months do not match the metrics')
    for phase in ('validation', 'test'):
        catalog = pd.read_csv(OUTPUT / f'{phase}_catalog.csv', dtype={'stock_code': 'string'}).set_index('stock_code')
        dates = set(monthly.loc[monthly.phase.eq(phase), 'snapshot_date'])
        part = rows.loc[rows.snapshot_date.isin(dates)]
        if (not part.stock_code.isin(catalog.index).all()
                or not part.description.eq(part.stock_code.map(catalog.description)).all()):
            raise ValueError('Recommended products/names do not match the frozen known catalog')
        metrics = monthly.loc[monthly.phase.eq(phase) & monthly.method.eq(lock['selected_method'])]
        for metric in metrics.itertuples():
            if rows.loc[rows.snapshot_date.eq(metric.snapshot_date), 'customer_id'].nunique() != metric.customers:
                raise ValueError('Recommendation customer count differs from evaluation')
    return rows, monthly, lock, comparison


def load_comparison_list(date: str, customer: str, method: str) -> pd.DataFrame:
    """Read only precomputed recommendation IDs, never target or hit columns."""
    rows, monthly, lock, _ = load_recommendations()
    if method not in NAMES or date not in set(monthly.snapshot_date):
        raise ValueError('Unsupported recommendation method/date')
    if method == lock['selected_method']:
        selected = rows.loc[rows.snapshot_date.eq(date) & rows.customer_id.eq(customer)]
        if len(selected) != 10:
            raise ValueError('Unsupported customer for this snapshot')
        return export_recommendations(selected)
    phase = monthly.loc[monthly.snapshot_date.eq(date), 'phase'].iloc[0]
    predictions = pd.read_csv(OUTPUT / f'{phase}_customer_metrics.csv',
                              usecols=['snapshot_date', 'customer_id', 'method', 'recommended_items'],
                              dtype={'customer_id': 'string'})
    match = predictions.loc[predictions.snapshot_date.eq(date) & predictions.customer_id.eq(customer)
                            & predictions.method.eq(method)]
    if len(match) != 1:
        raise ValueError('Missing comparison recommendation')
    codes = json.loads(match.recommended_items.iloc[0])
    if len(codes) != 10 or len(set(codes)) != 10:
        raise ValueError('Invalid comparison product list')
    catalog = pd.read_csv(OUTPUT / f'{phase}_catalog.csv', dtype={'stock_code': 'string'}).set_index('stock_code')
    if not set(codes).issubset(catalog.index):
        raise ValueError('Comparison contains a product outside the frozen catalog')
    tx = pd.read_csv(ROOT / 'data/processed/customer_snapshot_pipeline/clean_transactions.csv',
                     usecols=['invoice_date', 'customer_id', 'stock_code'],
                     dtype={'customer_id': 'string', 'stock_code': 'string'}, parse_dates=['invoice_date'])
    cutoff = pd.Timestamp(date)
    prior = tx.loc[tx.customer_id.eq(customer) & tx.invoice_date.lt(cutoff)
                   & tx.invoice_date.ge(cutoff - pd.Timedelta(days=180))]
    bought = set(prior.stock_code)
    no_history = not bool(bought & set(catalog.index))
    out = pd.DataFrame([dict(snapshot_date=date, customer_id=customer, rank=rank,
                             stock_code=code, description=catalog.loc[code, 'description'],
                             purchased_in_past_180d=code in bought, recommendation_method=method,
                             reason=('No personalized signal available, please fill the position according to popular products' if no_history
                                     else f'{NAMES[method]} Sorting results of comparison plans; ties are ranked according to popularity'))
                        for rank, code in enumerate(codes, 1)])
    validate_recommendations(out, method)
    return export_recommendations(out)
