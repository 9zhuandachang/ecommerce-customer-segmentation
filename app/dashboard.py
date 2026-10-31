"""Four-task customer workspace. Research reports remain outside the business UI."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# A running Streamlit session may retain imported modules after the page script
# changes. Refresh older text modules once without clearing uploaded data.
for module_name in ('app.business_views', 'app.upload_inference'):
    text_module = importlib.import_module(module_name)
    if getattr(text_module, 'UI_TEXT_VERSION', None) != 'en-1':
        importlib.reload(text_module)

from app.stage6_data import review_list
from app.upload_inference import (UploadError, analyze_upload, load_models,
                                  recommend_customer, MIN_ANALYSIS_DATE, MAX_ANALYSIS_DATE, RECOMMENDATION_METHOD)
from app.business_views import (PAGES, customer_filter, group_summary, customer_table,
                                value_table, product_table, profile_details, business_export,
                                GROUP_NAMES, display_customer_groups)

st.set_page_config(page_title='Customer Insights', page_icon=':material/analytics:', layout='wide',
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
        'segments': 'Explore purchase habits and spending patterns across customer segments.',
        'repurchase': 'Prioritize customers by their likelihood of purchasing again within 30 days.',
        'value': 'Review customer spending potential and decide whom to prioritize.',
        'recommendations': 'Review purchase habits and 10 product recommendations for the next order.',
    }
    title_column, date_column = st.columns([3, 1], vertical_alignment='center')
    with title_column:
        st.markdown('<div class="page-eyebrow">CUSTOMER WORKSPACE</div>', unsafe_allow_html=True)
        st.title(title, anchor=False)
        st.caption(descriptions[page])
    with date_column:
        st.caption(f'Analysis date · {date}  \nCustomers · {customer_count:,}')
        if st.button('Change data', key='change_data', icon=':material/upload_file:'):
            open_page('upload')
            st.rerun()


def upload_workspace():
    st.markdown('<div class="page-eyebrow">CUSTOMER WORKSPACE</div>', unsafe_allow_html=True)
    st.title('Data import', anchor=False)
    st.caption('Upload purchase history and select an analysis date to generate customer snapshots and all four analyses.')
    current = st.session_state.get('uploaded_workspace')
    if current is not None:
        st.caption(f'Current analysis: {current["date"]} · {len(current["customers"]):,} customers')
        if current.get('recommendation_method') != RECOMMENDATION_METHOD:
            st.info('The recommendation model has been updated. Please upload purchase history and generate the analysis again.')
    with st.container(border=True, key='card_upload'):
        left, right = st.columns(2)
        with left:
            history = st.file_uploader('Purchase history CSV', type=['csv'], key='upload_history',
                                       help='Provide the complete available purchase history. Only transactions before the analysis date are used.')
        with right:
            previous_date = (st.session_state.get('uploaded_workspace') or {}).get('date', '2011-12-01')
            default_date = min(max(pd.Timestamp(previous_date), MIN_ANALYSIS_DATE), MAX_ANALYSIS_DATE)
            if 'upload_date' in st.session_state and not (
                    MIN_ANALYSIS_DATE <= pd.Timestamp(st.session_state['upload_date']) <= MAX_ANALYSIS_DATE):
                del st.session_state['upload_date']
            analysis_date = st.date_input('Analysis date', value=default_date.date(),
                                          min_value=MIN_ANALYSIS_DATE.date(), max_value=MAX_ANALYSIS_DATE.date(),
                                          key='upload_date', format='YYYY-MM-DD',
                                          help='Choose any date from 2011-09-01 to 2011-12-01. Transactions on or after that date are excluded.')
        if st.button('Generate analysis', key='run_upload', type='primary', icon=':material/analytics:'):
            if history is None:
                st.error('Please upload a purchase-history CSV file first.')
            else:
                try:
                    with st.spinner('Validating data and generating analysis…'):
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
                    st.error('Analysis could not be completed. Please ask the project maintainer to check the model files.')
        if current is not None and current.get('recommendation_method') == RECOMMENDATION_METHOD:
            if st.button('Return to current analysis', key='cancel_upload'):
                previous = st.session_state.get('last_analysis_page', 'segments')
                open_page(previous if previous in PAGES else 'segments')
                st.rerun()


def row_limit(key):
    return st.selectbox('Customers to display', [25, 50, 100, 200, 500], index=1, key=key)


def download(shown, date, name, customer=None, profile=None):
    out = business_export(shown, date, customer, profile)
    st.download_button('Export current list', data=out.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'{name}_{date}.csv', mime='text/csv', icon=':material/download:')


def money_columns():
    return {
        'Spend in last 180 days (£)': st.column_config.NumberColumn(format='£%.2f'),
        'Average order value (£)': st.column_config.NumberColumn(format='£%.2f'),
    }


def style_chart(figure, group_names):
    figure.update_layout(height=290, margin=dict(l=4, r=20, t=10, b=12), showlegend=False,
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(family='Segoe UI, Microsoft YaHei, sans-serif', size=12, color='#64748b'))
    figure.update_xaxes(showgrid=True, gridcolor='#eef3fa', zeroline=False)
    figure.update_yaxes(showgrid=False, zeroline=False, autorange='reversed',
                       categoryorder='array', categoryarray=list(group_names))
    return figure


# Preserve selections when a previously opened Chinese-language session reloads.
legacy_choices = {
    '\u5168\u90e8\u7fa4\u4f53': 'All segments',
    '\u5ba2\u6237ID\u5347\u5e8f': 'Customer ID (ascending)',
    '\u8fd1180\u5929\u8d2d\u4e70\u6b21\u6570\u4ece\u9ad8\u5230\u4f4e': 'Purchases in last 180 days (descending)',
    '\u7fa4 1\uff5c\u8f83\u4e45\u672a\u8d2d': GROUP_NAMES[1],
    '\u7fa4 2\uff5c\u8fd1\u671f\u9ad8\u9891': GROUP_NAMES[2],
    '\u7fa4 3\uff5c\u8fd1\u671f\u4f4e\u5ba2\u5355': GROUP_NAMES[0],
    '\u7fa4 4\uff5c\u8fd1\u671f\u9ad8\u5ba2\u5355': GROUP_NAMES[3],
}
for selection_key in ('segment_group', 'repurchase_group', 'recommendation_customer_sort'):
    previous_choice = st.session_state.get(selection_key)
    if previous_choice in legacy_choices:
        st.session_state[selection_key] = legacy_choices[previous_choice]

st.session_state.setdefault('workspace_page', 'upload')
if st.session_state.pop('show_upload', False):
    open_page('upload')
current_workspace = st.session_state.get('uploaded_workspace')
workspace_ready = current_workspace is not None and current_workspace.get('recommendation_method') == RECOMMENDATION_METHOD
if st.session_state.workspace_page not in {'upload', *PAGES} or not workspace_ready:
    st.session_state.workspace_page = 'upload'
with st.sidebar:
    st.markdown('''<div class="workspace-brand"><div class="brand-mark" aria-hidden="true">
        <i></i><i></i><i></i><i></i></div><div><div class="brand-title">Customer Insights</div>
        <div class="brand-subtitle">CUSTOMER ANALYTICS</div></div></div>''', unsafe_allow_html=True)
    st.button('Data import', key='nav_upload', icon=':material/upload_file:', width='stretch',
              type='primary' if st.session_state.workspace_page == 'upload' else 'secondary',
              on_click=open_page, args=('upload',))
    st.markdown('<div class="nav-label">Customer analysis</div>', unsafe_allow_html=True)
    for name, (label, icon) in PAGES.items():
        st.button(label, key=f'nav_{name}', icon=icon, width='stretch',
                  type='primary' if st.session_state.workspace_page == name else 'secondary',
                  disabled=not workspace_ready,
                  on_click=open_page, args=(name,))
    st.markdown('<div class="sidebar-bottom">From purchase history to customer insights.</div>', unsafe_allow_html=True)

if st.session_state.workspace_page == 'upload':
    upload_workspace()
    st.stop()

workspace = st.session_state['uploaded_workspace']
page = st.session_state.workspace_page
date = workspace['date']
header(page, date, len(workspace['customers']))
if workspace.get('ignored_future_rows', 0):
    st.caption(f'Excluded {workspace["ignored_future_rows"]:,} transactions on or after the analysis date. Only earlier records are used.')
month = display_customer_groups(workspace['customers']) if page in ('segments', 'repurchase') else (
    workspace['priority'] if page == 'value' else workspace['profiles'])

if page == 'segments':
    summary = group_summary(month)
    palette = dict(zip(summary.segment_name, BLUES))
    left, right = st.columns(2)
    with left, st.container(border=True, key='card_segment_sizes'):
        st.subheader('Customer segment distribution', anchor=False)
        figure = px.bar(summary, x='customers', y='segment_name', color='segment_name', orientation='h',
                         color_discrete_map=palette,
                         labels={'customers': 'Customers', 'segment_name': ''}, text='customers')
        figure.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(style_chart(figure, summary.segment_name), width='stretch', config={'displayModeBar': False})
    with right, st.container(border=True, key='card_segment_spending'):
        st.subheader('180-day spend by segment', anchor=False)
        figure = px.bar(summary, x='spending', y='segment_name', color='segment_name', orientation='h',
                         color_discrete_map=palette,
                         labels={'spending': 'Median 180-day spend (£)', 'segment_name': ''})
        figure.update_traces(hovertemplate='%{y}<br>Median spend £%{x:,.2f}<extra></extra>')
        st.plotly_chart(style_chart(figure, summary.segment_name), width='stretch', config={'displayModeBar': False})
    with st.container(border=True, key='card_segment_summary'):
        st.subheader('Segment profiles', anchor=False)
        st.caption(f'Overall reference: median 180-day spend £{month.monetary.median():,.2f}')
        table = summary.rename(columns={'segment_name': 'Customer segment', 'customers': 'Customers', 'share': 'Share',
                                         'recency': 'Median days since last purchase', 'orders': 'Median orders (180d)',
                                         'spending': 'Median 180-day spend (£)', 'aov': 'Median order value (£)'})
        st.dataframe(table[['Customer segment', 'Customers', 'Share', 'Median days since last purchase', 'Median orders (180d)',
                            'Median 180-day spend (£)', 'Median order value (£)']], hide_index=True, width='stretch',
                     column_config={'Share': st.column_config.NumberColumn(format='percent'),
                                    'Median 180-day spend (£)': st.column_config.NumberColumn(format='£%.2f'),
                                    'Median order value (£)': st.column_config.NumberColumn(format='£%.2f')})
    with st.container(border=True, key='card_segment_customers'):
        st.subheader('Customers in this segment', anchor=False)
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            segment = st.selectbox('Customer segment', ['All segments'] + sorted(month.segment_name.unique()), key='segment_group')
        with f2:
            query = st.text_input('Customer ID', placeholder='Enter a customer ID to filter', key='segment_search')
        with f3:
            limit = row_limit('segment_limit')
        filtered = customer_filter(month, segment, query)
        shown = customer_table(filtered.sort_values(['recency_days', 'CustomerID']).head(limit), include_score=False)
        action_left, action_right = st.columns([3, 1])
        with action_right:
            download(shown, date, 'customer_segments')
        st.caption(f'Matched {len(filtered):,} customers · Showing {len(shown):,}')
        if shown.empty:
            st.info('No matching customers. Adjust the filters.')
        st.dataframe(shown, hide_index=True, width='stretch', height=380, column_config=money_columns())

elif page == 'repurchase':
    with st.container(border=True, key='card_repurchase_filters'):
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            segment = st.selectbox('Customer segment', ['All segments'] + sorted(month.segment_name.unique()), key='repurchase_group')
        with f2:
            query = st.text_input('Customer ID', placeholder='Enter a customer ID to filter', key='repurchase_search')
        with f3:
            limit = row_limit('repurchase_limit')
    filtered = customer_filter(month, segment, query)
    ranked = review_list(filtered, 'All segments', limit)
    with st.container(border=True, key='card_repurchase_customers'):
        shown = customer_table(ranked)
        title_column, action_column = st.columns([3, 1], vertical_alignment='center')
        with title_column:
            st.subheader('Repurchase priorities', anchor=False)
        with action_column:
            download(shown, date, 'repurchase_priority')
        st.caption(f'Matched {len(filtered):,} customers · Showing {len(shown):,}')
        st.caption('Higher scores indicate higher review priority.')
        if shown.empty:
            st.info('No matching customers. Adjust the filters.')
        config = money_columns()
        config['Repurchase score'] = st.column_config.ProgressColumn(min_value=0., max_value=1., format='%.3f')
        st.dataframe(shown, hide_index=True, width='stretch', height=450, column_config=config)

elif page == 'value':
    with st.container(border=True, key='card_value_filters'):
        f1, f2 = st.columns([3, 1])
        with f1:
            query = st.text_input('Customer ID', placeholder='Enter a customer ID to filter', key='value_search')
        with f2:
            limit = row_limit('value_limit')
    priority_count = int(month.priority_band.eq('Top 20%').sum())
    scoped = month.loc[month.priority_band.eq('Top 20%')]
    filtered = customer_filter(scoped, customer=query).sort_values('priority_rank')
    with st.container(border=True, key='card_value_customers'):
        shown = value_table(filtered.head(limit))
        title_column, action_column = st.columns([3, 1], vertical_alignment='center')
        with title_column:
            st.subheader('Spending potential ranking', anchor=False)
            st.caption(f'Top 20% by spending potential · **{priority_count:,} customers**')
        with action_column:
            download(shown, date, 'spend_potential')
        st.caption(f'Matched {len(filtered):,} customers · Showing {len(shown):,}')
        st.caption('Ranks show relative spending potential within the uploaded cohort, rather than predicted amounts.')
        if shown.empty:
            st.info('No matching customers. Adjust the filters.')
        st.dataframe(shown, hide_index=True, width='stretch', height=450, column_config=money_columns())

else:
    with st.container(border=True, key='card_recommendation_customer'):
        sort_column, customer_column = st.columns([1, 3])
        with sort_column:
            customer_sort = st.selectbox('Customer sort order', ['Customer ID (ascending)', 'Purchases in last 180 days (descending)'],
                                         key='recommendation_customer_sort')
        ordered = (month.sort_values(['order_count', 'customer_id'], ascending=[False, True], kind='stable')
                   if customer_sort == 'Purchases in last 180 days (descending)' else month.sort_values('customer_id', kind='stable'))
        ids = ordered.customer_id.tolist()
        if st.session_state.get('recommendation_customer') not in ids:
            st.session_state['recommendation_customer'] = '12347' if '12347' in ids else ids[0]
        with customer_column:
            customer = st.selectbox('Customer ID', ids, key='recommendation_customer')
    profile = month.loc[month.customer_id.eq(customer)].iloc[0]
    summary_columns = st.columns(3 if pd.notna(profile.historical_repeat_share) else 2)
    summary_columns[0].metric('Purchases (180d)', str(int(profile.order_count)))
    summary_columns[1].metric('Distinct products purchased', str(int(profile.distinct_products)))
    if pd.notna(profile.historical_repeat_share):
        summary_columns[2].metric('Historical repeat-product share', f'{profile.historical_repeat_share:.1%}',
                                 help='Within the last 180 days, the first order establishes history. Average the share of previously purchased products across later orders.')
    if profile.history_status != 'Comparable':
        st.caption('Purchase history is limited; interpret these habits with care.')
    with st.expander('Profile details'):
        st.table(profile_details(profile))
    rows = recommend_customer(workspace, customer)
    shown = product_table(rows)
    with st.container(border=True, key='card_recommendation_products'):
        title_column, action_column = st.columns([3, 1], vertical_alignment='center')
        with title_column:
            st.subheader('10 product recommendations for this customer', anchor=False)
        with action_column:
            download(shown, date, f'product_recommendations_{customer}', customer, profile)
        st.dataframe(shown, hide_index=True, width='stretch', height=450,
                     column_config={'Purchases (180d)': st.column_config.NumberColumn(format='%d'),
                                    'Days since last product purchase': st.column_config.NumberColumn(format='%.1f')})
