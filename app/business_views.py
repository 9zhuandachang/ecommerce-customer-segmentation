"""Business-facing tables: preserve frozen scores and expose only useful known fields."""

from __future__ import annotations

import pandas as pd

PAGES = {
    'segments': ('客户分群', ':material/grid_view:'),
    'repurchase': ('复购优先名单', ':material/repeat:'),
    'value': ('高消费潜力', ':material/trending_up:'),
    'recommendations': ('画像与商品推荐', ':material/shopping_bag:'),
}
# Frozen model cluster IDs stay unchanged; these are the business display names.
GROUP_NAMES = {
    1: '群 1｜较久未购',
    2: '群 2｜近期高频',
    0: '群 3｜近期低客单',
    3: '群 4｜近期高客单',
}
CUSTOMER_COLUMNS = {
    'rank': '优先名次', 'CustomerID': '客户 ID', 'segment_name': '客户群体',
    'repurchase_score': '复购评分', 'recency_days': '距上次购买天数',
    'frequency': '近180天订单数', 'monetary': '近180天消费额 (£)', 'aov': '平均客单价 (£)',
}
VALUE_COLUMNS = {
    'priority_rank': '潜力排名', 'customer_id': '客户 ID',
    'recency_days': '距上次购买天数', 'monetary_value_180d': '近180天消费额 (£)',
}
PRODUCT_COLUMNS = {
    'rank': '推荐名次', 'stock_code': '商品编码', 'description': '商品名称',
    'purchase_count_180d': '近180天购买次数',
    'days_since_item_purchase': '距该商品上次购买天数',
}


def display_customer_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename groups for the UI without changing model IDs or customer membership."""
    names = frame.cluster.map(GROUP_NAMES)
    if names.isna().any():
        raise ValueError('Unknown customer cluster')
    return frame.assign(segment_name=names)


def customer_filter(frame: pd.DataFrame, segment='全部群体', customer='') -> pd.DataFrame:
    out = frame
    if segment != '全部群体':
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
        ('常购商品集中度', formatted('top3_concentration', '{:.1%}'),
         '购买次数最多的3种商品，占全部商品购买记录的比例。'),
        ('近期订单老商品平均占比', formatted('recent_repeat_share', '{:.1%}'),
         '最近最多2笔可比较订单中，老商品占比的平均值。'),
        ('老商品占比变化', formatted('repeat_share_change', '{:+.1f}个百分点', 100),
         '最近2笔与此前可比较订单的平均占比之差；至少有5次购买时计算。'),
        ('平均每次购买商品种数', formatted('mean_basket_size', '{:.1f}种'),
         '每次购买中不同商品种数的平均值。'),
        ('距上次购买天数', formatted('recency_days', '{:.1f}天'),
         '最近一次购买距分析日期的天数。'),
        ('本期购买跨度', formatted('history_span_days', '{:.1f}天'),
         '近180天内，首次与最近一次购买相隔的天数。'),
        ('可比较订单数', formatted('comparable_order_count', '{:.0f}笔'),
         '首次购买建立历史，后续订单用于比较老商品占比。'),
    ]
    return pd.DataFrame(rows, columns=['画像指标', '当前值', '含义']).dropna(subset=['当前值']).set_index('画像指标')


def business_export(shown: pd.DataFrame, date: str, customer: str | None = None, profile=None) -> pd.DataFrame:
    """Export exactly the visible business columns with their date/customer context."""
    allowed = set(CUSTOMER_COLUMNS.values()) | set(VALUE_COLUMNS.values()) | set(PRODUCT_COLUMNS.values())
    out = shown[[name for name in shown if name in allowed]].copy()
    out.insert(0, '分析日期', date)
    if customer is not None and '客户 ID' not in out:
        out.insert(1, '客户 ID', customer)
    if profile is not None:
        for field, name in {'order_count': '客户近180天购买次数', 'distinct_products': '客户购买商品种数',
                            'historical_repeat_share': '客户历史重复购买比例',
                            'top3_concentration': '客户常购商品集中度'}.items():
            out[name] = profile[field]
    return out.reset_index(drop=True)
