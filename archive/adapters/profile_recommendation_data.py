"""Safe purchase-habit profiles and locked hybrid product lists for the dashboard."""

from __future__ import annotations

import json

import pandas as pd

from archive.recommendation.hybrid import NAMES
from src.recommendation.profiles import FEATURES
from pipelines.recommendation_profiles import OUTPUT, check_completed

MANIFEST = OUTPUT / 'manifest.json'
PROFILE_COLUMNS = ['snapshot_date', 'customer_id'] + FEATURES + [
    'history_status', 'habit_group', 'predicted_repeat_share', 'propensity_method']
LIST_COLUMNS = ['snapshot_date', 'customer_id', 'method', 'recommended_items', 'prior_purchase_flags', 'repeat_weight']
EXPORT_COLUMNS = ['snapshot_date', 'customer_id', 'rank', 'stock_code', 'description',
                  'purchased_in_past_180d', 'recommendation_method', 'recommendation_route', 'reason',
                  'repeat_weight', 'historical_repeat_share', 'recent_repeat_share', 'repeat_share_change',
                  'top3_concentration', 'order_count', 'comparable_order_count', 'distinct_products',
                  'history_status', 'predicted_repeat_share', 'propensity_method']


def load_profile_package() -> dict:
    manifest = check_completed()
    lock = json.loads((OUTPUT / 'locked_selection.json').read_text(encoding='utf-8'))
    if (lock['test_labels_used_for_selection'] or lock['global_test_model_updates']
            or manifest['selected_method'] != lock['selected_method']
            or not manifest['review_after_selection_lock']
            or lock['model_sha256'] != manifest['output_hashes']['model_bundle.joblib']
            or lock['plan_sha256'] != manifest['output_hashes']['locked_plan.json']):
        raise ValueError('Profile selection/model lock is inconsistent')
    profiles = pd.read_csv(OUTPUT / 'profiles.csv', dtype={'customer_id': 'string'})
    profiles['snapshot_date'] = pd.to_datetime(profiles.snapshot_date, errors='raise').dt.strftime('%Y-%m-%d')
    if set(profiles) != set(PROFILE_COLUMNS) or profiles.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Unsafe or duplicated purchase profiles')
    if not profiles.predicted_repeat_share.between(0, 1).all():
        raise ValueError('Invalid conditional repeat-share predictions')
    if not profiles.historical_repeat_share.dropna().between(0, 1).all():
        raise ValueError('Invalid historic repeat shares')
    lists = pd.concat([pd.read_csv(OUTPUT / f'{phase}_lists.csv', dtype={'customer_id': 'string'})
                       for phase in ('validation', 'test')], ignore_index=True)
    if (set(lists) != set(LIST_COLUMNS) or lists.duplicated(['snapshot_date', 'customer_id', 'method']).any()
            or set(lists.method) != set(NAMES)):
        raise ValueError('Unsafe or incomplete recommendation comparison lists')
    keys = ['snapshot_date', 'customer_id']
    expected = set(map(tuple, profiles[keys].to_numpy()))
    for method, part in lists.groupby('method'):
        if set(map(tuple, part[keys].to_numpy())) != expected:
            raise ValueError(f'Missing profile recommendation customers: {method}')
        weights = part.repeat_weight
        if method.startswith(('fixed_', 'adaptive_')):
            if not weights.between(0, 1).all():
                raise ValueError('Invalid hybrid weights')
        elif weights.notna().any():
            raise ValueError('Pure method unexpectedly has hybrid weights')
    catalogs = {phase: pd.read_csv(OUTPUT / f'{phase}_catalog.csv', dtype={'stock_code': 'string'}).set_index('stock_code')
                for phase in ('validation', 'test')}
    return dict(profiles=profiles, lists=lists, lock=lock, catalogs=catalogs,
                monthly=pd.read_csv(OUTPUT / 'monthly_comparison.csv'),
                comparison=pd.read_csv(OUTPUT / 'comparison.csv'))


def profile_recommendation_list(package: dict, date: str, customer: str, method: str) -> pd.DataFrame:
    if method not in NAMES:
        raise ValueError('Unsupported recommendation method')
    profiles, lists = package['profiles'], package['lists']
    profile = profiles.loc[profiles.snapshot_date.eq(date) & profiles.customer_id.eq(customer)]
    match = lists.loc[lists.snapshot_date.eq(date) & lists.customer_id.eq(customer) & lists.method.eq(method)]
    if len(profile) != 1 or len(match) != 1:
        raise ValueError('Unsupported recommendation date/customer')
    profile, match = profile.iloc[0], match.iloc[0]
    codes = json.loads(match.recommended_items)
    flags = json.loads(match.prior_purchase_flags)
    if (len(codes) != 10 or len(set(codes)) != 10 or len(flags) != 10
            or not all(isinstance(flag, bool) for flag in flags)):
        raise ValueError('Expected ten unique safe products and prior-purchase flags')
    phase = 'validation' if date < '2011-09-01' else 'test'
    catalog = package['catalogs'][phase]
    if not set(codes).issubset(catalog.index):
        raise ValueError('Product is outside the frozen catalog')
    rows = []
    for rank, (code, bought) in enumerate(zip(codes, flags), 1):
        if method.startswith(('fixed_', 'adaptive_')):
            reason = 'Enter the replenishment path according to the number of personal purchases, and then sort by mixed points' if bought else 'Score according to the algorithm to enter the discovery path; with the same score, according to the popularity'
            if match.repeat_weight in (0., 1.):
                reason += '; Zero points and same time points will be supplemented according to popularity'
        elif method == 'personal_repeat':
            reason = 'Sort by individual purchases' if bought else 'There are insufficient 10 pieces of old products, please fill them in according to their popularity.'
        elif method == 'popular':
            reason = 'Sort by popularity during training period'
        else:
            reason = f'{NAMES[method]} is rated based on past purchasing relationships; same score based on popularity'
        rows.append(dict(snapshot_date=date, customer_id=customer, rank=rank, stock_code=code,
                         description=catalog.loc[code, 'description'], purchased_in_past_180d=bought,
                         recommendation_method=method, recommendation_route='Bought product candidates' if bought else 'Unbought product candidates',
                         reason=reason, repeat_weight=match.repeat_weight,
                         **{column: profile[column] for column in EXPORT_COLUMNS
                            if column in profile.index and column not in ('snapshot_date', 'customer_id')}))
    return export_profile_recommendations(pd.DataFrame(rows))


def export_profile_recommendations(rows: pd.DataFrame) -> pd.DataFrame:
    if not set(EXPORT_COLUMNS).issubset(rows):
        raise ValueError('Missing safe profile recommendation export fields')
    out = rows[EXPORT_COLUMNS].copy()
    if (len(out) != 10 or out.stock_code.nunique() != 10 or sorted(out['rank']) != list(range(1, 11))
            or out.customer_id.nunique() != 1 or out.snapshot_date.nunique() != 1):
        raise ValueError('Expected one customer and ten distinct recommendation products')
    return out.sort_values('rank', kind='stable')
