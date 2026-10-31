"""Verified, future-answer-free item scoring lists and customer purchase habits."""

from __future__ import annotations

from src.paths import ROOT

import json

import numpy as np
import pandas as pd

from src.recommendation.profiles import FEATURES
from src.recommendation.data import ROOT, sha256
from src.recommendation.models import NAMES, MODEL_NAMES
from src.recommendation.features import GROUP_NAMES
from pipelines.recommendation import OUTPUT, check_completed

MANIFEST = OUTPUT / 'manifest.json'
PROFILE_COLUMNS = ['snapshot_date', 'customer_id'] + FEATURES + ['history_status', 'habit_group']
LIST_COLUMNS = ['snapshot_date', 'customer_id', 'method', 'recommended_items',
                'purchase_counts_180d', 'days_since_item_purchase', 'ranking_scores']
EXPORT_COLUMNS = ['snapshot_date', 'customer_id', 'rank', 'stock_code', 'description',
                  'recommendation_method', 'feature_group', 'model_family', 'ranking_score',
                  'purchased_in_past_180d', 'purchase_count_180d', 'days_since_item_purchase',
                  'historical_repeat_share', 'recent_repeat_share', 'repeat_share_change', 'top3_concentration',
                  'order_count', 'comparable_order_count', 'history_status', 'reason']


def load_item_package() -> dict:
    manifest = check_completed()
    lock = json.loads((OUTPUT / 'locked_selection.json').read_text(encoding='utf-8'))
    if (manifest['selected_method'] != lock['selected_method'] or lock['test_labels_used_for_selection']
            or lock['global_test_model_updates'] or not manifest['review_after_selection_lock']
            or lock['model_sha256'] != manifest['output_hashes']['model_bundle.joblib']
            or lock['plan_sha256'] != manifest['output_hashes']['locked_plan.json']):
        raise ValueError('Item model and selection locks do not agree')
    profiles = pd.read_csv(OUTPUT / 'profiles.csv', dtype={'customer_id': 'string'})
    profiles.snapshot_date = pd.to_datetime(profiles.snapshot_date, errors='raise').dt.strftime('%Y-%m-%d')
    if set(profiles) != set(PROFILE_COLUMNS) or profiles.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Unsafe or duplicated item profiles')
    lists = pd.concat([pd.read_csv(OUTPUT / f'{phase}_lists.csv', dtype={'customer_id': 'string'})
                       for phase in ('validation', 'test')], ignore_index=True)
    if (set(lists) != set(LIST_COLUMNS) or set(lists.method) != set(NAMES)
            or lists.duplicated(['snapshot_date', 'customer_id', 'method']).any()):
        raise ValueError('Unsafe or incomplete item comparison lists')
    expected = set(map(tuple, profiles[['snapshot_date', 'customer_id']].to_numpy()))
    for method, part in lists.groupby('method'):
        if set(map(tuple, part[['snapshot_date', 'customer_id']].to_numpy())) != expected:
            raise ValueError(f'Incomplete item recommendation customers: {method}')
    catalogs = {phase: pd.read_csv(OUTPUT / f'{phase}_catalog.csv', dtype={'stock_code': 'string'}).set_index('stock_code')
                for phase in ('validation', 'test')}
    diagnostics = None
    audit_path = ROOT / 'artifacts/recommendation/audit'
    if (audit_path / 'manifest.json').exists():
        audit = json.loads((audit_path / 'manifest.json').read_text(encoding='utf-8'))
        if (audit['source_manifest_sha256'] != sha256(MANIFEST)
                or audit['code_sha256'] != sha256(ROOT / 'pipelines/audit_recommendation.py')):
            raise ValueError('Item audit source/code does not match')
        for name, digest in audit['output_hashes'].items():
            path = (audit_path / name).resolve()
            if path.parent != audit_path.resolve() or sha256(path) != digest:
                raise ValueError('Item audit output changed')
        diagnostics = json.loads((audit_path / 'report.json').read_text(encoding='utf-8'))
    return dict(lock=lock, profiles=profiles, lists=lists, catalogs=catalogs,
                monthly=pd.read_csv(OUTPUT / 'monthly_comparison.csv'),
                comparison=pd.read_csv(OUTPUT / 'comparison.csv'), ablations=pd.read_csv(OUTPUT / 'validation_ablations.csv'),
                diagnostics=diagnostics)


def export_item_recommendations(rows: pd.DataFrame) -> pd.DataFrame:
    if not set(EXPORT_COLUMNS).issubset(rows):
        raise ValueError('Missing safe item export fields')
    out = rows[EXPORT_COLUMNS].copy()
    if (len(out) != 10 or out.stock_code.nunique() != 10 or sorted(out['rank']) != list(range(1, 11))
            or out.customer_id.nunique() != 1 or out.snapshot_date.nunique() != 1
            or not np.isfinite(out.ranking_score).all() or not out.purchase_count_180d.ge(0).all()):
        raise ValueError('Expected one customer and ten safe distinct products')
    return out.sort_values('rank', kind='stable')


def item_recommendation_list(package: dict, date: str, customer: str, method: str) -> pd.DataFrame:
    if method not in NAMES:
        raise ValueError('Unsupported item recommendation method')
    profiles, lists = package['profiles'], package['lists']
    profile = profiles.loc[profiles.snapshot_date.eq(date) & profiles.customer_id.eq(customer)]
    match = lists.loc[lists.snapshot_date.eq(date) & lists.customer_id.eq(customer) & lists.method.eq(method)]
    if len(profile) != 1 or len(match) != 1:
        raise ValueError('Unsupported item recommendation date/customer')
    profile, match = profile.iloc[0], match.iloc[0]
    codes, counts, ages, scores = [json.loads(match[name]) for name in
                                  ['recommended_items', 'purchase_counts_180d', 'days_since_item_purchase', 'ranking_scores']]
    if (any(len(x) != 10 for x in (codes, counts, ages, scores)) or len(set(codes)) != 10
            or not np.isfinite(counts + ages + scores).all() or min(counts) < 0 or min(ages) < 0):
        raise ValueError('Invalid item recommendation evidence')
    phase = 'validation' if date < '2011-09-01' else 'test'
    catalog = package['catalogs'][phase]
    if not set(codes).issubset(catalog.index):
        raise ValueError('Recommended item is outside frozen catalog')
    if method[0] in GROUP_NAMES:
        group, family = method.split('_', 1)
        group_name, family_name = GROUP_NAMES[group], MODEL_NAMES[family]
        reason = f'The historical information of {group_name} has been uniformly scored by {family_name}; the number and time of purchases are for verification'
    else:
        group_name, family_name = 'Comparison with old plan', NAMES[method]
        reason = 'Recommendations of original frozen solutions, sorted by their rules or models'
    rows = []
    for rank, (code, count, age, score) in enumerate(zip(codes, counts, ages, scores), 1):
        rows.append(dict(snapshot_date=date, customer_id=customer, rank=rank, stock_code=code,
                         description=catalog.loc[code, 'description'], recommendation_method=method,
                         feature_group=group_name, model_family=family_name, ranking_score=score,
                         purchased_in_past_180d=count > 0, purchase_count_180d=count,
                         days_since_item_purchase=age if count > 0 else np.nan, reason=reason,
                         **{column: profile[column] for column in EXPORT_COLUMNS if column in profile.index
                            and column not in ('snapshot_date', 'customer_id')}))
    return export_item_recommendations(pd.DataFrame(rows))
