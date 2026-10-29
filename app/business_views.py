"""Business-facing tables: preserve frozen scores and expose only useful known fields."""

from __future__ import annotations

import pandas as pd

UI_TEXT_VERSION = 'en-1'

PAGES = {
    'segments': ('Customer segments', ':material/grid_view:'),
    'repurchase': ('Repurchase priorities', ':material/repeat:'),
    'value': ('High-value customers', ':material/trending_up:'),
    'recommendations': ('Product recommendations', ':material/shopping_bag:'),
}
# Frozen model cluster IDs stay unchanged; these are the business display names.
GROUP_NAMES = {
    1: 'Group 1 | Long inactive',
    2: 'Group 2 | Recent frequent buyers',
    0: 'Group 3 | Recent low order value',
    3: 'Group 4 | Recent high order value',
}
CUSTOMER_COLUMNS = {
    'rank': 'Priority rank', 'CustomerID': 'Customer ID', 'segment_name': 'Customer segment',
    'repurchase_score': 'Repurchase score', 'recency_days': 'Days since last purchase',
    'frequency': 'Orders (180d)', 'monetary': 'Spend in last 180 days (£)', 'aov': 'Average order value (£)',
}
VALUE_COLUMNS = {
    'priority_rank': 'Potential rank', 'customer_id': 'Customer ID',
    'recency_days': 'Days since last purchase', 'monetary_value_180d': 'Spend in last 180 days (£)',
}
PRODUCT_COLUMNS = {
    'rank': 'Recommendation rank', 'stock_code': 'Product code', 'description': 'Product name',
    'purchase_count_180d': 'Purchases (180d)',
    'days_since_item_purchase': 'Days since last product purchase',
}


def display_customer_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename groups for the UI without changing model IDs or customer membership."""
    names = frame.cluster.map(GROUP_NAMES)
    if names.isna().any():
        raise ValueError('Unknown customer cluster')
    return frame.assign(segment_name=names)


def customer_filter(frame: pd.DataFrame, segment='All segments', customer='') -> pd.DataFrame:
    out = frame
    if segment != 'All segments':
        out = out.loc[out.segment_name.eq(segment)]
    if customer.strip():
        column = 'CustomerID' if 'CustomerID' in out else 'customer_id'
        out = out.loc[out[column].astype(str).str.contains(customer.strip(), regex=False, na=False)]
    return out.copy()


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby('segment_name', sort=True).agg(
        customers=('CustomerID', 'size'), recency=('recency_days', 'median'),
        orders=('frequency', 'median'), spending=('monetary', 'median'), aov=('aov', 'median'),
    ).reset_index()
    grouped['share'] = grouped.customers / max(len(frame), 1)
    return grouped


def customer_table(frame: pd.DataFrame, *, include_score=True) -> pd.DataFrame:
    columns = CUSTOMER_COLUMNS.copy()
    if not include_score:
        columns.pop('rank')
        columns.pop('repurchase_score')
    return frame[[name for name in columns if name in frame]].rename(columns=columns).reset_index(drop=True)


def value_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(VALUE_COLUMNS)].rename(columns=VALUE_COLUMNS).reset_index(drop=True)


def product_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[list(PRODUCT_COLUMNS)].rename(columns=PRODUCT_COLUMNS).reset_index(drop=True)


def profile_details(profile) -> pd.DataFrame:
    """Explain available habit features beyond the summary cards."""
    def formatted(field, template, scale=1):
        value = profile[field]
        return None if pd.isna(value) else template.format(value * scale)

    rows = [
        ('Top-3 product concentration', formatted('top3_concentration', '{:.1%}'),
         'Share of product purchase records accounted for by the three most frequently purchased products.'),
        ('Recent repeat-product share', formatted('recent_repeat_share', '{:.1%}'),
         'Average share of previously purchased products in the latest two comparable orders, or the one available order.'),
        ('Change in repeat-product share', formatted('repeat_share_change', '{:+.1f} pp', 100),
         'Recent two-order average minus the earlier comparable-order average, in percentage points. Requires at least five purchases.'),
        ('Average basket size (products)', formatted('mean_basket_size', '{:.1f} products'),
         'Average number of distinct products per purchase.'),
        ('Days since last purchase', formatted('recency_days', '{:.1f} days'),
         'Days from the latest purchase to the analysis date.'),
        ('Purchase history span', formatted('history_span_days', '{:.1f} days'),
         'Days between the first and latest purchases within the last 180 days.'),
        ('Comparable orders', formatted('comparable_order_count', '{:.0f} orders'),
         'The first purchase establishes history; later orders allow repeat-product comparisons.'),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Value', 'Meaning']).dropna(subset=['Value']).set_index('Metric')


def business_export(shown: pd.DataFrame, date: str, customer: str | None = None, profile=None) -> pd.DataFrame:
    """Export exactly the visible business columns with their date/customer context."""
    allowed = set(CUSTOMER_COLUMNS.values()) | set(VALUE_COLUMNS.values()) | set(PRODUCT_COLUMNS.values())
    out = shown[[name for name in shown if name in allowed]].copy()
    out.insert(0, 'Analysis date', date)
    if customer is not None and 'Customer ID' not in out:
        out.insert(1, 'Customer ID', customer)
    if profile is not None:
        for field, name in {'order_count': 'Customer purchases (180d)', 'distinct_products': 'Customer distinct products',
                            'historical_repeat_share': 'Customer historical repeat-product share',
                            'top3_concentration': 'Customer top-3 product concentration'}.items():
            out[name] = profile[field]
    return out.reset_index(drop=True)
