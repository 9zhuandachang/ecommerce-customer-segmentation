"""Four-task customer workspace. Research reports remain outside the business UI."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.stage6_data import review_list
from app.upload_inference import (DEMO, UploadError, analyze_upload, load_models,
                                  recommend_customer, MIN_ANALYSIS_DATE, MAX_ANALYSIS_DATE, RECOMMENDATION_METHOD)
from app.business_views import (PAGES, customer_filter, group_summary, customer_table,
                                value_table, product_table, business_export)

st.set_page_config(page_title='客户洞察', page_icon=':material/analytics:', layout='wide',
                   initial_sidebar_state='expanded')
st.markdown('<style>' + Path(__file__).with_name('dashboard.css').read_text(encoding='utf-8') + '</style>',
            unsafe_allow_html=True)
LOGGER = logging.getLogger(__name__)
BLUES = ['#2458cd', '#5788e8', '#8fb2f4', '#bfd3fa']


@st.cache_resource(show_spinner=False)
def frozen_models(recommendation_method):
    # The active method also keys the cache after switching the business default.
    return load_models()


def open_page(name):
    current = st.session_state.get('workspace_page')
    if name == 'upload' and current in PAGES:
        st.session_state['last_analysis_page'] = current
    st.session_state['workspace_page'] = name


def header(page, date, customer_count):
    title, _ = PAGES[page]
    descriptions = {
        'segments': '了解不同客户的购买习惯与消费特征。',
        'repurchase': '按未来30天的复购倾向，整理值得优先查看的客户。',
        'value': '查看客户消费潜力排名，确定关注顺序。',
        'recommendations': '查看客户购买习惯与下一单的10件候选商品。',
    }
    title_column, date_column = st.columns([3, 1], vertical_alignment='center')
    with title_column:
        st.markdown('<div class="page-eyebrow">CUSTOMER WORKSPACE</div>', unsafe_allow_html=True)
        st.title(title, anchor=False)
        st.caption(descriptions[page])
    with date_column:
        st.caption(f'分析日期 · {date}  \n本批客户 · {customer_count:,}位')
        if st.button('更换数据', key='change_data', icon=':material/upload_file:'):
            open_page('upload')
            st.rerun()


def upload_workspace():
    st.markdown('<div class="page-eyebrow">CUSTOMER WORKSPACE</div>', unsafe_allow_html=True)
    st.title('数据导入', anchor=False)
    st.caption('上传购买明细，选择分析日期，系统自动生成客户快照和四项分析。')
    current = st.session_state.get('uploaded_workspace')
    if current is not None:
        st.caption(f'当前分析：{current["date"]} · {len(current["customers"]):,}位客户')
        if current.get('recommendation_method') != RECOMMENDATION_METHOD:
            st.info('推荐方案已更新，请重新上传购买明细并生成分析。')
    with st.container(border=True, key='card_upload'):
        left, right = st.columns(2)
        with left:
            history = st.file_uploader('购买明细 CSV', type=['csv'], key='upload_history',
                                       help='提供完整可见购买历史，系统只使用分析日期之前的记录。')
        with right:
            previous_date = (st.session_state.get('uploaded_workspace') or {}).get('date', '2011-12-01')
            default_date = min(max(pd.Timestamp(previous_date), MIN_ANALYSIS_DATE), MAX_ANALYSIS_DATE)
            if 'upload_date' in st.session_state and not (
                    MIN_ANALYSIS_DATE <= pd.Timestamp(st.session_state['upload_date']) <= MAX_ANALYSIS_DATE):
                del st.session_state['upload_date']
            analysis_date = st.date_input('分析日期', value=default_date.date(),
                                          min_value=MIN_ANALYSIS_DATE.date(), max_value=MAX_ANALYSIS_DATE.date(),
                                          key='upload_date', format='YYYY-MM-DD',
                                          help='可选择2011-09-01至2011-12-01之间任意一天。当天及之后的交易会自动排除。')
        if st.button('生成分析', key='run_upload', type='primary', icon=':material/analytics:'):
            if history is None:
                st.error('请先上传购买明细CSV文件。')
            else:
                try:
                    with st.spinner('正在核对数据并生成分析…'):
                        workspace = analyze_upload(history.getvalue(), analysis_date, frozen_models(RECOMMENDATION_METHOD))
                    st.session_state['uploaded_workspace'] = workspace
                    open_page('segments')
                    for key in ('segment_group', 'repurchase_group', 'segment_search', 'repurchase_search',
                                'value_search', 'recommendation_customer'):
                        st.session_state.pop(key, None)
                    st.rerun()
                except UploadError as error:
                    st.error(str(error))
                except (ValueError, KeyError, OSError, AssertionError):
                    LOGGER.exception('Could not analyze uploaded customer data')
                    st.error('分析暂时无法完成，请联系项目维护者检查模型文件。')
        if current is not None and current.get('recommendation_method') == RECOMMENDATION_METHOD:
            if st.button('返回当前分析', key='cancel_upload'):
                previous = st.session_state.get('last_analysis_page', 'segments')
                open_page(previous if previous in PAGES else 'segments')
                st.rerun()
    demo = DEMO / 'december_upload_demo.zip'
    if demo.exists():
        st.download_button('下载12月真实演示文件', demo.read_bytes(), file_name=demo.name,
                           mime='application/zip', icon=':material/download:')
        st.caption('解压后只上传purchase_history.csv，分析日期选择2011-12-01。')


def row_limit(key):
    return st.selectbox('显示客户数', [25, 50, 100, 200, 500], index=1, key=key)


def download(shown, date, name, customer=None, profile=None):
    out = business_export(shown, date, customer, profile)
    st.download_button('导出当前名单', data=out.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'{name}_{date}.csv', mime='text/csv', icon=':material/download:')


def money_columns():
    return {
        '近180天消费额 (£)': st.column_config.NumberColumn(format='£%.2f'),
        '平均客单价 (£)': st.column_config.NumberColumn(format='£%.2f'),
    }


def style_chart(figure, group_names):
    figure.update_layout(height=290, margin=dict(l=4, r=20, t=10, b=12), showlegend=False,
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(family='Segoe UI, Microsoft YaHei, sans-serif', size=12, color='#64748b'))
    figure.update_xaxes(showgrid=True, gridcolor='#eef3fa', zeroline=False)
    figure.update_yaxes(showgrid=False, zeroline=False, autorange='reversed',
                       categoryorder='array', categoryarray=list(group_names))
    return figure


st.session_state.setdefault('workspace_page', 'upload')
if st.session_state.pop('show_upload', False):
    open_page('upload')
current_workspace = st.session_state.get('uploaded_workspace')
workspace_ready = current_workspace is not None and current_workspace.get('recommendation_method') == RECOMMENDATION_METHOD
if st.session_state.workspace_page not in {'upload', *PAGES} or not workspace_ready:
    st.session_state.workspace_page = 'upload'
with st.sidebar:
    st.markdown('''<div class="workspace-brand"><div class="brand-mark" aria-hidden="true">
        <i></i><i></i><i></i><i></i></div><div><div class="brand-title">客户洞察</div>
        <div class="brand-subtitle">CUSTOMER ANALYTICS</div></div></div>''', unsafe_allow_html=True)
    st.button('数据导入', key='nav_upload', icon=':material/upload_file:', width='stretch',
              type='primary' if st.session_state.workspace_page == 'upload' else 'secondary',
              on_click=open_page, args=('upload',))
    st.markdown('<div class="nav-label">客户分析</div>', unsafe_allow_html=True)
    for name, (label, icon) in PAGES.items():
        st.button(label, key=f'nav_{name}', icon=icon, width='stretch',
                  type='primary' if st.session_state.workspace_page == name else 'secondary',
                  disabled=not workspace_ready,
                  on_click=open_page, args=(name,))
    st.markdown('<div class="sidebar-bottom">从购买记录，到客户洞察。</div>', unsafe_allow_html=True)

if st.session_state.workspace_page == 'upload':
    upload_workspace()
    st.stop()

workspace = st.session_state['uploaded_workspace']
page = st.session_state.workspace_page
date = workspace['date']
header(page, date, len(workspace['customers']))
if workspace.get('ignored_future_rows', 0):
    st.caption(f'已排除分析日当天及之后的{workspace["ignored_future_rows"]:,}条交易，仅使用此前记录。')
month = workspace['customers'] if page in ('segments', 'repurchase') else (
    workspace['priority'] if page == 'value' else workspace['profiles'])

if page == 'segments':
    summary = group_summary(month)
    palette = dict(zip(summary.segment_name, BLUES))
    left, right = st.columns(2)
    with left, st.container(border=True, key='card_segment_sizes'):
        st.subheader('客户群体分布', anchor=False)
        figure = px.bar(summary, x='customers', y='segment_name', color='segment_name', orientation='h',
                         color_discrete_map=palette,
                         labels={'customers': '客户数', 'segment_name': ''}, text='customers')
        figure.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(style_chart(figure, summary.segment_name), width='stretch', config={'displayModeBar': False})
    with right, st.container(border=True, key='card_segment_spending'):
        st.subheader('各群近180天累计消费', anchor=False)
        figure = px.bar(summary, x='spending', y='segment_name', color='segment_name', orientation='h',
                         color_discrete_map=palette,
                         labels={'spending': '近180天累计消费中位数 (£)', 'segment_name': ''})
        figure.update_traces(hovertemplate='%{y}<br>消费中位数 £%{x:,.2f}<extra></extra>')
        st.plotly_chart(style_chart(figure, summary.segment_name), width='stretch', config={'displayModeBar': False})
    with st.container(border=True, key='card_segment_summary'):
        st.subheader('群体特征', anchor=False)
        st.caption(f'整体参照：本批客户近180天消费中位数 £{month.monetary.median():,.2f}')
        table = summary.rename(columns={'segment_name': '客户群体', 'customers': '客户数', 'share': '占比',
                                         'recency': '距上次购买中位天数', 'orders': '近180天订单中位数',
                                         'spending': '近180天消费中位数 (£)', 'aov': '客单价中位数 (£)'})
        st.dataframe(table[['客户群体', '客户数', '占比', '距上次购买中位天数', '近180天订单中位数',
                            '近180天消费中位数 (£)', '客单价中位数 (£)']], hide_index=True, width='stretch',
                     column_config={'占比': st.column_config.NumberColumn(format='percent'),
                                    '近180天消费中位数 (£)': st.column_config.NumberColumn(format='£%.2f'),
                                    '客单价中位数 (£)': st.column_config.NumberColumn(format='£%.2f')})
    with st.container(border=True, key='card_segment_customers'):
        st.subheader('查看群内客户', anchor=False)
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            segment = st.selectbox('客户群体', ['全部群体'] + sorted(month.segment_name.unique()), key='segment_group')
        with f2:
            query = st.text_input('客户 ID', placeholder='输入客户 ID 筛选', key='segment_search')
        with f3:
            limit = row_limit('segment_limit')
        filtered = customer_filter(month, segment, query)
        shown = customer_table(filtered.sort_values(['recency_days', 'CustomerID']).head(limit), include_score=False)
        action_left, action_right = st.columns([3, 1])
        with action_right:
            download(shown, date, 'customer_segments')
        st.caption(f'共匹配 {len(filtered):,} 位客户，当前显示 {len(shown):,} 位')
        if shown.empty:
            st.info('没有匹配的客户，请调整筛选。')
        st.dataframe(shown, hide_index=True, width='stretch', height=380, column_config=money_columns())

elif page == 'repurchase':
    with st.container(border=True, key='card_repurchase_filters'):
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            segment = st.selectbox('客户群体', ['全部群体'] + sorted(month.segment_name.unique()), key='repurchase_group')
        with f2:
            query = st.text_input('客户 ID', placeholder='输入客户 ID 筛选', key='repurchase_search')
        with f3:
            limit = row_limit('repurchase_limit')
    filtered = customer_filter(month, segment, query)
    ranked = review_list(filtered, '全部群体', limit)
    with st.container(border=True, key='card_repurchase_customers'):
        shown = customer_table(ranked)
        title_column, action_column = st.columns([3, 1], vertical_alignment='center')
        with title_column:
            st.subheader('复购优先名单', anchor=False)
        with action_column:
            download(shown, date, 'repurchase_priority')
        st.caption(f'共匹配 {len(filtered):,} 位客户，当前显示 {len(shown):,} 位')
        st.caption('评分越高，查看优先级越高；评分不等于准确的复购概率。')
        if shown.empty:
            st.info('没有匹配的客户，请调整筛选。')
        config = money_columns()
        config['复购评分'] = st.column_config.ProgressColumn(min_value=0., max_value=1., format='%.3f')
        st.dataframe(shown, hide_index=True, width='stretch', height=450, column_config=config)

elif page == 'value':
    with st.container(border=True, key='card_value_filters'):
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            scope = st.selectbox('客户范围', ['前20%优先客户', '全部客户'], key='value_scope')
        with f2:
            query = st.text_input('客户 ID', placeholder='输入客户 ID 筛选', key='value_search')
        with f3:
            limit = row_limit('value_limit')
    priority_count = int(month.priority_band.eq('Top 20%').sum())
    scoped = month.loc[month.priority_band.eq('Top 20%')] if scope == '前20%优先客户' else month
    filtered = customer_filter(scoped, customer=query).sort_values('priority_rank')
    with st.container(border=True, key='card_value_customers'):
        shown = value_table(filtered.head(limit))
        title_column, action_column = st.columns([3, 1], vertical_alignment='center')
        with title_column:
            st.subheader('消费潜力名单', anchor=False)
            st.caption(f'前20%消费潜力客户 · **{priority_count:,}位**')
        with action_column:
            download(shown, date, 'spend_potential')
        st.caption(f'共匹配 {len(filtered):,} 位客户，当前显示 {len(shown):,} 位')
        st.caption('名次表示本批上传客户的相对消费潜力，不是预计消费金额。')
        if shown.empty:
            st.info('没有匹配的客户，请调整筛选。')
        st.dataframe(shown, hide_index=True, width='stretch', height=450, column_config=money_columns())

else:
    with st.container(border=True, key='card_recommendation_customer'):
        ids = sorted(month.customer_id.unique())
        if st.session_state.get('recommendation_customer') not in ids:
            st.session_state['recommendation_customer'] = '12347' if '12347' in ids else ids[0]
        customer = st.selectbox('客户 ID', ids, key='recommendation_customer')
    profile = month.loc[month.customer_id.eq(customer)].iloc[0]
    a, b, c = st.columns(3)
    a.metric('近180天购买次数', str(int(profile.order_count)))
    b.metric('购买商品种数', str(int(profile.distinct_products)))
    c.metric('历史订单老商品平均占比', '记录不足' if pd.isna(profile.historical_repeat_share) else f'{profile.historical_repeat_share:.1%}',
             help='在近180天内，第一笔订单建立购买历史；对后续每单计算以前买过的商品占比，再取平均。')
    if profile.history_status != '可比较':
        st.caption('可用购买记录较少，购买习惯仅供参考。')
    with st.expander('画像详情'):
        concentration = '记录不足' if pd.isna(profile.top3_concentration) else f'{profile.top3_concentration:.1%}'
        st.caption(f'常购商品集中度：**{concentration}**')
        st.caption('购买次数最多的3种商品，占近180天全部商品购买记录的比例。')
    if workspace['unknown_products']:
        st.caption('部分历史商品不在当前商品库中，推荐名单仅覆盖已知商品。')
    rows = recommend_customer(workspace, customer)
    shown = product_table(rows)
    with st.container(border=True, key='card_recommendation_products'):
        title_column, action_column = st.columns([3, 1], vertical_alignment='center')
        with title_column:
            st.subheader('为这位客户推荐的10件商品', anchor=False)
        with action_column:
            download(shown, date, f'product_recommendations_{customer}', customer, profile)
        st.dataframe(shown, hide_index=True, width='stretch', height=450,
                     column_config={'近180天购买次数': st.column_config.NumberColumn(format='%d'),
                                    '距该商品上次购买天数': st.column_config.NumberColumn(format='%.1f')})
