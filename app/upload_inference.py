"""Feature-only inference from uploaded purchase history and an analysis date.

No saved customer predictions or future labels are loaded by this module.
"""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.business_views import display_customer_groups
from src.recommendation.data import make_context
from src.recommendation.features import FEATURE_SETS, item_context, pair_features
from src.recommendation.models import predict_item_scores
from src.recommendation.algorithms import topk
from src.recommendation.profiles import purchase_profiles
from src.data.cleaning import NON_PRODUCT_CODES
from src.data.validation import FEATURES as BASE_FEATURES
from src.repurchase.comparison import add_features
from src.integration.models import score
from src.spending.comparison import make_priority, predict, rule_score
from src.spending.features import build_temporal_features

from src.paths import ROOT
UI_TEXT_VERSION = 'en-1'
DEMO = ROOT / 'data/demo/2011-12-01'
SNAPSHOT_COLUMNS = ['snapshot_date', 'customer_id'] + BASE_FEATURES
TRANSACTION_COLUMNS = ['customer_id', 'invoice', 'invoice_date', 'stock_code',
                       'description', 'quantity', 'unit_price', 'line_amount']
MAX_BYTES = 150 * 1024 * 1024
MIN_ANALYSIS_DATE = pd.Timestamp('2011-09-01')
MAX_ANALYSIS_DATE = pd.Timestamp('2011-12-01')
# Current business default, selected for the project. Frozen experiment decisions
# remain in their original manifests and selection files.
RECOMMENDATION_METHOD = 'D_classifier'


class UploadError(ValueError):
    """An actionable input error that may be shown directly to the user."""


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _read_csv(content: bytes, columns: list[str], label: str) -> pd.DataFrame:
    if not content or len(content) > MAX_BYTES:
        raise UploadError(f'{label} must not be empty or larger than 150 MB.')
    try:
        data = pd.read_csv(BytesIO(content), dtype='string', encoding='utf-8-sig')
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise UploadError(f'{label} could not be read. Use a UTF-8 encoded CSV file.') from error
    missing, extra = sorted(set(columns) - set(data)), sorted(set(data) - set(columns))
    if missing:
        raise UploadError(f'{label} is missing fields: {", ".join(missing)}. Please refer to the example file.')
    if extra:
        raise UploadError(f'{label} contains unsupported fields: {", ".join(extra)}. Keep only the template fields and do not upload prediction labels.')
    if data.empty or len(data) > 2_000_000 or data.isna().any().any():
        raise UploadError(f'{label} must contain 1 to 2,000,000 rows, with all fields completed.')
    for column in columns:
        data[column] = data[column].str.strip()
        if data[column].eq('').any():
            raise UploadError(f'{label} contains empty values in {column}.')
    return data[columns]


def _dates(data: pd.DataFrame, column: str, label: str) -> None:
    try:
        data[column] = pd.to_datetime(data[column], errors='raise', format='mixed')
        if data[column].dt.tz is not None or data[column].isna().any():
            raise ValueError('Expected local datetimes')
    except (ValueError, TypeError, AttributeError) as error:
        raise UploadError(f'{label} has invalid dates in {column}. Use dates or times without a time zone.') from error


def _numbers(data: pd.DataFrame, columns: list[str], label: str) -> None:
    try:
        for column in columns:
            data[column] = pd.to_numeric(data[column], errors='raise').astype('float64')
        if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite number')
    except (ValueError, TypeError) as error:
        raise UploadError(f'Numeric fields in {label} must contain finite numbers.') from error


def snapshots_from_history(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """The canonical eight features, without constructing a future-answer column."""
    cutoff = pd.Timestamp(cutoff)
    past = history.loc[history.invoice_date.lt(cutoff)]
    observed = past.loc[past.invoice_date.ge(cutoff - pd.Timedelta(days=180))]
    if observed.empty:
        raise UploadError('There are no eligible purchases in the 180 days before the analysis date.')
    invoices = observed.groupby(['customer_id', 'invoice'], observed=True, as_index=False).agg(
        invoice_date=('invoice_date', 'max'), order_value=('line_amount', 'sum'))
    customers = observed.groupby('customer_id', observed=True).agg(
        last_purchase_datetime=('invoice_date', 'max'), monetary_value_180d=('line_amount', 'sum'),
        product_diversity_180d=('stock_code', 'nunique'))
    customers = customers.join(invoices.groupby('customer_id', observed=True).agg(
        invoice_frequency_180d=('invoice', 'size'), average_order_value_180d=('order_value', 'mean')))
    first = past.groupby('customer_id', observed=True).invoice_date.min()
    customers['customer_history_days'] = (cutoff.normalize() - customers.index.to_series().map(first).dt.normalize()).dt.days
    customers['recency_days'] = (cutoff.normalize() - customers.last_purchase_datetime.dt.normalize()).dt.days
    for days in (30, 90):
        counts = invoices.loc[invoices.invoice_date.ge(cutoff - pd.Timedelta(days=days))].groupby('customer_id').size()
        customers[f'invoice_count_{days}d'] = customers.index.to_series().map(counts).fillna(0).astype(int)
    customers['snapshot_date'] = cutoff
    customers[['monetary_value_180d', 'average_order_value_180d']] = customers[
        ['monetary_value_180d', 'average_order_value_180d']].round(2)
    return customers.reset_index()[SNAPSHOT_COLUMNS].sort_values('customer_id', kind='stable').reset_index(drop=True)


def validate_upload(history_bytes: bytes, analysis_date) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Validate one history file and build the eligible cohort strictly before cutoff."""
    try:
        cutoff = pd.Timestamp(analysis_date)
        if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
            raise ValueError('Expected a local calendar date')
    except (ValueError, TypeError) as error:
        raise UploadError('The analysis date must be a valid calendar date without a time of day or time zone.') from error
    if not MIN_ANALYSIS_DATE <= cutoff <= MAX_ANALYSIS_DATE:
        raise UploadError('The analysis date must be from 2011-09-01 to 2011-12-01, inclusive.')
    history = _read_csv(history_bytes, TRANSACTION_COLUMNS, 'Purchase history')
    _dates(history, 'invoice_date', 'Purchase history')
    future = history.invoice_date.ge(cutoff)
    ignored = int(future.sum())
    history = history.loc[~future].copy()
    if history.empty:
        raise UploadError('No purchases precede the analysis date. Check the date or upload earlier purchase history.')
    _numbers(history, ['quantity', 'unit_price', 'line_amount'], 'Purchase history')
    if (history[['quantity', 'unit_price', 'line_amount']] <= 0).any().any():
        raise UploadError('Upload cleaned purchase history with quantity, unit price and amount greater than zero.')
    if not np.allclose(history.line_amount, (history.quantity * history.unit_price).round(2), rtol=0, atol=1e-6):
        raise UploadError('In purchase history, line_amount must equal quantity multiplied by unit_price, rounded to two decimal places.')
    cents = history.line_amount.to_numpy(dtype=float) * 100
    if not np.allclose(cents, np.rint(cents), rtol=0, atol=1e-6):
        raise UploadError('Purchase-history amounts must have no more than two decimal places.')
    code = history.stock_code.str.upper()
    if (history.invoice.str.upper().str.startswith('C').any() or code.isin(NON_PRODUCT_CODES).any()
            or code.str.startswith(('GIFT_', 'TEST')).any()):
        raise UploadError('Remove cancelled orders, charges, test entries, gift vouchers and other non-product records.')
    if history.duplicated(TRANSACTION_COLUMNS).any():
        raise UploadError('Purchase history contains duplicate rows. Remove duplicates first.')
    if history.groupby('invoice').customer_id.nunique().gt(1).any():
        raise UploadError('An invoice ID is associated with different customers. Check the invoice IDs.')
    snapshots = snapshots_from_history(history, cutoff)
    return snapshots, history, ignored


def load_models() -> dict:
    """Load only trusted local model packages and the frozen product catalog."""
    segment_dir = ROOT / 'artifacts/integration/current'
    lock = json.loads((segment_dir / 'locked_config.json').read_text(encoding='utf-8'))
    model_path = segment_dir / 'model/integrated_bundle.joblib'
    if digest(model_path) != lock['output_hashes']['model']:
        raise ValueError('Customer model integrity check failed')
    customer = joblib.load(model_path)
    value_dir = ROOT / 'artifacts/spending/current'
    value_lock = json.loads((value_dir / 'locked_selection.json').read_text(encoding='utf-8'))
    value_path = value_dir / 'ranking_bundle.joblib'
    if digest(value_path) != value_lock['model_sha256']:
        raise ValueError('Value model integrity check failed')
    value = joblib.load(value_path)
    if value['selected_method'] != value_lock['selected_method'] or value_lock['selected_method'] != 'lgbm_two_stage':
        raise ValueError('Unexpected selected value model')
    item_dir = ROOT / 'artifacts/recommendation/current'
    item_lock = json.loads((item_dir / 'locked_selection.json').read_text(encoding='utf-8'))
    manifest = json.loads((item_dir / 'manifest.json').read_text(encoding='utf-8'))
    catalog_path = item_dir / 'test_catalog.csv'
    item_path = item_dir / 'model_bundle.joblib'
    if (item_lock['selected_method'] != manifest['selected_method']
            or item_lock['best_new'] != RECOMMENDATION_METHOD
            or digest(catalog_path) != manifest['output_hashes']['test_catalog.csv']
            or digest(item_path) != item_lock['model_sha256']
            or item_lock['model_sha256'] != manifest['output_hashes']['model_bundle.joblib']):
        raise ValueError('Unexpected recommendation configuration or catalog')
    catalog = pd.read_csv(catalog_path, dtype={'stock_code': 'string'})
    item_bundle = joblib.load(item_path)
    item_model = item_bundle['test']['models'][RECOMMENDATION_METHOD]
    ease = item_bundle['test']['ease']
    if (item_model['group'] != 'D' or item_model['spec']['family'] != 'classifier'
            or item_model['features'] != FEATURE_SETS['D']
            or pd.Timestamp(item_model['fit_cutoff']) > MIN_ANALYSIS_DATE
            or pd.Timestamp(ease['fit_cutoff']) > MIN_ANALYSIS_DATE
            or list(catalog.stock_code) != list(ease['catalog'].stock_code)):
        raise ValueError('Recommendation model features or frozen catalog differ')
    return dict(customer=customer, value=value['review'][value_lock['selected_method']],
                value_method=value_lock['selected_method'], baseline=value['baseline_method'],
                catalog=catalog, recommendation_method=RECOMMENDATION_METHOD,
                recommendation_model=item_model, recommendation_ease=ease,
                recommendation_version=item_lock['model_sha256'])


def analyze_upload(history_bytes: bytes, analysis_date, models: dict | None = None) -> dict:
    snapshots, history, ignored = validate_upload(history_bytes, analysis_date)
    models = load_models() if models is None else models
    cutoff = snapshots.snapshot_date.iloc[0]
    catalog = models['catalog']
    if pd.to_datetime(catalog.catalog_cutoff).max() > cutoff:
        raise UploadError('The analysis date precedes the frozen product catalog date.')
    invoices = history.groupby(['customer_id', 'invoice'], observed=True, as_index=False).invoice_date.max()
    features = add_features(snapshots, invoices)
    scored = score(models['customer'], features)
    customers = features[SNAPSHOT_COLUMNS].copy()
    customers['cluster'] = scored.cluster.to_numpy()
    customers['repurchase_score'] = scored.score.to_numpy()
    customers = display_customer_groups(customers).rename(columns={
        'customer_id': 'CustomerID', 'invoice_frequency_180d': 'frequency',
        'monetary_value_180d': 'monetary', 'average_order_value_180d': 'aov',
        'product_diversity_180d': 'product_diversity'})
    customers['snapshot_date'] = str(cutoff.date())
    schedule = pd.DataFrame({'snapshot_date': [cutoff], 'feature_start': [cutoff - pd.Timedelta(days=180)]})
    temporal = build_temporal_features(snapshots, history, schedule)
    value_inputs = snapshots.merge(temporal, on=['snapshot_date', 'customer_id'], validate='one_to_one')
    priority = make_priority(value_inputs, predict(models['value'], value_inputs),
                             rule_score(models['baseline'], value_inputs), models['value_method'])
    priority['snapshot_date'] = priority.snapshot_date.dt.strftime('%Y-%m-%d')
    events = history.drop_duplicates(['customer_id', 'invoice', 'invoice_date', 'stock_code'], keep='last')
    context = make_context(events, snapshots, cutoff, catalog)
    profiles = purchase_profiles(events, context['customers'], cutoff)
    profiles['snapshot_date'] = profiles.snapshot_date.dt.strftime('%Y-%m-%d')
    ages = events.loc[events.invoice_date.ge(cutoff - pd.Timedelta(days=180))].groupby(
        ['customer_id', 'stock_code'], observed=True).invoice_date.max()
    recent = events.loc[events.invoice_date.ge(cutoff - pd.Timedelta(days=180))
                        & events.customer_id.isin(snapshots.customer_id)]
    unknown = set(recent.stock_code) - set(catalog.stock_code)
    return dict(date=str(cutoff.date()), customers=customers, priority=priority,
                profiles=profiles, context=context, item_last_purchase=ages,
                catalog=catalog, recommendation_method=models['recommendation_method'],
                recommendation_history=events[['customer_id', 'invoice', 'invoice_date', 'stock_code', 'description']],
                recommendation_model=models['recommendation_model'],
                recommendation_ease=models['recommendation_ease'],
                recommendation_version=models['recommendation_version'],
                generated_snapshots=snapshots,
                history_sha256=hashlib.sha256(history_bytes).hexdigest(),
                ignored_future_rows=ignored,
                history_rows=len(history), history_start=str(history.invoice_date.min()),
                history_end=str(history.invoice_date.max()), unknown_products=len(unknown))


def recommend_customer(workspace: dict, customer: str) -> pd.DataFrame:
    """Score this customer's full frozen catalog using the saved D classifier."""
    if workspace.get('recommendation_method') != RECOMMENDATION_METHOD:
        raise UploadError('The recommendation model has been updated. Please upload purchase history and generate the analysis again.')
    context = workspace['context']
    positions = np.flatnonzero(context['customers'] == customer)
    if len(positions) != 1:
        raise UploadError('No purchase history was found for this customer.')
    cutoff = pd.Timestamp(workspace['date'])
    snapshots = workspace['generated_snapshots'].loc[workspace['generated_snapshots'].customer_id.eq(customer)]
    item_inputs = item_context(workspace['recommendation_history'], snapshots, cutoff, workspace['recommendation_ease'])
    columns = np.arange(len(workspace['catalog']))
    features = pair_features(item_inputs, np.zeros(len(columns), dtype=int), columns)
    scores = predict_item_scores(workspace['recommendation_model'], features)
    indices = topk(scores[None, :])[0]
    counts = item_inputs['counts_dense'][0]
    rows = []
    for rank, item in enumerate(indices, 1):
        product = workspace['catalog'].iloc[item]
        count = int(counts[item])
        last = workspace['item_last_purchase'].get((customer, product.stock_code), pd.NaT)
        rows.append(dict(rank=rank, stock_code=str(product.stock_code), description=product.description,
                         purchased_in_past_180d=count > 0, purchase_count_180d=count,
                         days_since_item_purchase=(cutoff - last).total_seconds() / 86400 if count else np.nan,
                         recommendation_method=workspace['recommendation_method'], ranking_score=float(scores[item])))
    return pd.DataFrame(rows)
