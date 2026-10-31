"""Regression checks for cluster comparison and formal validation experiments."""

from src.paths import ROOT

import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from streamlit.testing.v1 import AppTest

from src.segmentation.comparison import (behavior_features, candidates,
                                         customer_holdout, one_state_per_customer)
from archive.segmentation.stability import align_clusters
from archive.repurchase.extended import FEATURE_SETS, fold_features, make_fold
from archive.repurchase.experiments import WEIGHTS, choose_weight, top20_metrics
from src.repurchase.comparison import (FEATURE_SETS as REPURCHASE_FEATURE_SETS,
                                       add_features as add_repurchase_features,
                                       choose_winner as choose_repurchase_winner,
                                       forward_fold as repurchase_forward_fold,
                                       make_model as make_repurchase_model,
                                       paired_customer_bootstrap)
from archive.integration.finalize import choose_f1_threshold, select_model, top20_mask
from archive.integration.optimize import BASELINE, choose_candidate, forward_parts, passes, ranking_guard
from archive.adapters.stage6_data import ROOT, export_review, load_stage6_replay, review_list


class ClusterStabilityTests(unittest.TestCase):
    def test_alignment_ignores_cluster_number_permutation(self):
        reference = np.array([0, 0, 1, 1, 2, 2])
        renamed = np.array([2, 2, 0, 0, 1, 1])
        self.assertEqual(align_clusters(reference, renamed, 3), {2: 0, 0: 1, 1: 2})

    def test_new_comparison_fixes_three_algorithms_and_two_to_six_groups(self):
        specs = candidates()
        self.assertEqual(len(specs), 20)
        self.assertEqual({spec.algorithm for spec in specs},
                         {'kmeans', 'bisecting_kmeans', 'gmm'})
        self.assertEqual({spec.k for spec in specs}, set(range(2, 7)))

    def test_new_behavior_features_ignore_future_answers(self):
        frame = pd.DataFrame({
            'recency_days': [20], 'invoice_frequency_180d': [4],
            'average_order_value_180d': [120.], 'product_diversity_180d': [12],
            'invoice_count_90d': [3], 'repurchase_30d': [0], 'spend_30d': [0.],
        })
        before = behavior_features(frame)
        frame.loc[0, ['repurchase_30d', 'spend_30d']] = [1, 9999.]
        after = behavior_features(frame)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.iloc[0].products_per_invoice_180d, 3)
        self.assertEqual(before.iloc[0].recent90_invoice_share, .75)

    def test_new_comparison_holds_out_whole_customers(self):
        frame = pd.DataFrame({'customer_id': [str(i) for i in range(20) for _ in range(3)]})
        fit, hold = customer_holdout(frame)
        self.assertFalse(set(frame.iloc[fit].customer_id) & set(frame.iloc[hold].customer_id))
        self.assertEqual(len(set(frame.iloc[hold].customer_id)), 4)
        one_state = one_state_per_customer(frame.iloc[hold].reset_index(drop=True))
        self.assertEqual(len(one_state), 4)
        self.assertFalse(one_state.customer_id.duplicated().any())


class StageSixDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.upload_inference import DEMO, analyze_upload
        cls.history_bytes = (DEMO / 'purchase_history.csv').read_bytes()
        cls.workspace = analyze_upload(cls.history_bytes, '2011-12-01')

    def open_dashboard(self):
        dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40)
        dashboard.session_state['uploaded_workspace'] = self.workspace
        dashboard.session_state['workspace_page'] = 'segments'
        return dashboard.run()

    def test_dashboard_starts_with_upload_and_does_not_load_saved_customer_results(self):
        import streamlit as st
        import re
        with patch('streamlit.set_page_config', wraps=st.set_page_config) as config:
            dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40).run()
        self.assertEqual(config.call_args.kwargs['initial_sidebar_state'], 'expanded')
        css = (ROOT / 'app/dashboard.css').read_text(encoding='utf-8')
        for selector, declarations in re.findall(r'([^{}]+)\{([^{}]*)\}', css):
            if 'stToolbar' in selector or 'stExpandSidebarButton' in selector:
                self.assertNotRegex(declarations, r'(visibility\s*:\s*hidden|display\s*:\s*none)')
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.title[0].value, 'Data import')
        self.assertEqual(dashboard.session_state['workspace_page'], 'upload')
        self.assertEqual([button.label for button in dashboard.sidebar.button],
                         ['Data import', 'Customer segments', 'Repurchase priorities', 'High-value customers', 'Product recommendations'])
        self.assertTrue(all(button.disabled for button in dashboard.sidebar.button if button.key != 'nav_upload'))
        self.assertFalse(dashboard.dataframe)
        self.assertEqual(len(dashboard.get('file_uploader')), 1)
        self.assertEqual(dashboard.date_input[0].value, pd.Timestamp('2011-12-01').date())
        self.assertEqual(dashboard.date_input[0].min, pd.Timestamp('2011-09-01').date())
        self.assertEqual(dashboard.date_input[0].max, pd.Timestamp('2011-12-01').date())
        next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertTrue(dashboard.error)
        self.assertFalse(dashboard.exception)

    def test_upload_button_computes_december_then_replace_can_cancel_or_reject(self):
        dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40).run()
        with patch('streamlit.file_uploader', return_value=BytesIO(self.history_bytes)):
            next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')
        self.assertEqual(dashboard.session_state['uploaded_workspace']['recommendation_method'], 'D_classifier')
        self.assertTrue(any('Customers · 3,444' in caption.value for caption in dashboard.caption))
        next(button for button in dashboard.button if button.key == 'change_data').click().run()
        with patch('streamlit.file_uploader', return_value=BytesIO(b'bad,file\n1,2')):
            next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertTrue(dashboard.error)
        self.assertFalse(dashboard.dataframe)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')
        next(button for button in dashboard.button if button.key == 'cancel_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertTrue(any('Customers · 3,444' in caption.value for caption in dashboard.caption))

    def test_old_rule_session_requires_regeneration_before_opening_business_pages(self):
        dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40)
        dashboard.session_state['uploaded_workspace'] = dict(self.workspace, recommendation_method='personal_repeat')
        dashboard.session_state['workspace_page'] = 'recommendations'
        dashboard.run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.title[0].value, 'Data import')
        self.assertTrue(any('The recommendation model has been updated' in message.value for message in dashboard.info))
        self.assertTrue(all(button.disabled for button in dashboard.sidebar.button if button.key != 'nav_upload'))
        self.assertFalse(any(button.key == 'cancel_upload' for button in dashboard.button))
        self.assertEqual(dashboard.session_state['uploaded_workspace']['recommendation_method'], 'personal_repeat')
        with patch('streamlit.file_uploader', return_value=BytesIO(self.history_bytes)):
            next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['recommendation_method'], 'D_classifier')
        self.assertEqual(dashboard.session_state['workspace_page'], 'segments')
        self.assertFalse(any(button.disabled for button in dashboard.sidebar.button))

    def test_upload_date_controls_generated_cohort_and_four_pages(self):
        dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40).run()
        dashboard.date_input[0].set_value(pd.Timestamp('2011-11-15').date()).run()
        with patch('streamlit.file_uploader', return_value=BytesIO(self.history_bytes)):
            next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        workspace = dashboard.session_state['uploaded_workspace']
        self.assertEqual(workspace['date'], '2011-11-15')
        self.assertTrue(workspace['generated_snapshots'].snapshot_date.eq(pd.Timestamp('2011-11-15')).all())
        self.assertLess(pd.Timestamp(workspace['history_end']), pd.Timestamp('2011-11-15'))
        self.assertGreater(workspace['ignored_future_rows'], 0)
        for page in ('segments', 'repurchase', 'value', 'recommendations'):
            self.open_business_page(dashboard, page)
            self.assertFalse(dashboard.exception)
            self.assertFalse(dashboard.error)
            self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-11-15')

    def test_sidebar_import_preserves_analysis_and_can_replace_it(self):
        dashboard = self.open_dashboard()
        self.open_business_page(dashboard, 'repurchase')
        original = dashboard.dataframe[0].value.copy()
        self.open_business_page(dashboard, 'upload')
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.title[0].value, 'Data import')
        self.assertEqual(len(dashboard.get('file_uploader')), 1)
        self.assertFalse(dashboard.dataframe)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')
        self.assertFalse(any(button.disabled for button in dashboard.sidebar.button))
        next(button for button in dashboard.button if button.key == 'cancel_upload').click().run()
        self.assertEqual(dashboard.session_state['workspace_page'], 'repurchase')
        pd.testing.assert_frame_equal(original, dashboard.dataframe[0].value)
        self.open_business_page(dashboard, 'upload')
        dashboard.date_input[0].set_value(pd.Timestamp('2011-11-15').date()).run()
        with patch('streamlit.file_uploader', return_value=BytesIO(self.history_bytes)):
            next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(dashboard.session_state['workspace_page'], 'segments')
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-11-15')

    def test_business_group_numbers_preserve_membership_and_export_names(self):
        from app.business_views import (GROUP_NAMES, business_export, customer_filter,
                                        customer_table, display_customer_groups, group_summary)
        frame = pd.DataFrame({
            'CustomerID': ['low', 'inactive', 'frequent', 'large_order'],
            'cluster': [0, 1, 2, 3], 'segment_name': ['old'] * 4,
            'recency_days': [10, 130, 5, 20], 'frequency': [2, 1, 5, 2],
            'monetary': [200., 300., 1600., 600.], 'aov': [100., 300., 320., 300.],
        })
        before = frame.copy(deep=True)
        shown = display_customer_groups(frame)
        self.assertEqual(shown.segment_name.tolist(), [
            'Group 3 | Recent low order value', 'Group 1 | Long inactive', 'Group 2 | Recent frequent buyers', 'Group 4 | Recent high order value'])
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(shown.drop(columns='segment_name'), before.drop(columns='segment_name'))
        self.assertEqual(group_summary(shown).segment_name.tolist(), list(GROUP_NAMES.values()))
        for cluster, name in GROUP_NAMES.items():
            filtered = customer_filter(shown, name)
            self.assertEqual(filtered.cluster.tolist(), [cluster])
            exported = business_export(customer_table(filtered, include_score=False), '2011-09-01')
            self.assertEqual(exported['Customer segment'].tolist(), [name])
            self.assertEqual(exported['Customer ID'].tolist(), frame.loc[frame.cluster.eq(cluster), 'CustomerID'].tolist())

    def test_frozen_model_reproduces_saved_test_scores_without_exporting_labels(self):
        replay, metadata, comparison = load_stage6_replay()
        self.assertEqual(len(replay), 8875)
        self.assertEqual(metadata['test_dates'],
                         ['2011-09-01', '2011-10-01', '2011-11-01'])
        self.assertEqual(metadata['model_name'], 'XGBoost')
        self.assertEqual(len(metadata['cluster_names']), 4)
        self.assertEqual(set(comparison.model), {'XGBoost', 'LR'})
        self.assertNotIn('repurchase_30d', replay.columns)
        self.assertTrue(replay.CustomerID.notna().all())
        self.assertEqual(replay.segment_name.nunique(), 4)

    def test_review_filter_rank_and_csv_export_are_the_same_rows(self):
        frame = pd.DataFrame({
            'CustomerID': pd.Series(['b', 'a', 'c'], dtype='string'),
            'segment_name': ['A', 'A', 'B'],
            'repurchase_score': [.7, .7, .9],
            'recency_days': [2, 3, 4], 'frequency': [3, 2, 1],
            'monetary': [30., 20., 10.],
            'snapshot_date': ['2011-09-01'] * 3,
            'repurchase_30d': [1, 0, 1],
        })
        shown = review_list(frame, 'A', 2)
        self.assertEqual(shown.CustomerID.tolist(), ['a', 'b'])
        self.assertEqual(shown['rank'].tolist(), [1, 2])
        exported = export_review(shown, 'sha256:example')
        self.assertEqual(exported.CustomerID.tolist(), shown.CustomerID.tolist())
        self.assertEqual(exported['rank'].tolist(), shown['rank'].tolist())
        self.assertNotIn('repurchase_30d', exported.columns)
        self.assertTrue(exported.model_version.eq('sha256:example').all())

    @staticmethod
    def open_business_page(dashboard, key):
        next(button for button in dashboard.sidebar.button if button.key == 'nav_' + key).click().run()
        return dashboard

    def test_dashboard_has_four_rectangular_business_entries_without_research_pages(self):
        from app.business_views import PAGES
        dashboard = self.open_dashboard()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual([button.label for button in dashboard.sidebar.button], ['Data import'] + [value[0] for value in PAGES.values()])
        self.assertEqual(len(dashboard.sidebar.radio), 0)
        self.assertEqual(len(dashboard.radio), 0)
        for key in PAGES:
            self.open_business_page(dashboard, key)
            self.assertFalse(dashboard.exception)
            self.assertFalse(dashboard.error)
            text = ' '.join(str(element.value) for category in ('title', 'caption', 'subheader', 'info')
                            for element in getattr(dashboard, category))
            for redundant in ('sha256', 'XGBoost', 'LightGBM', 'NDCG', 'ROC-AUC', 'Brier', 'Validation period', 'Test-period review', 'model version'):
                self.assertNotIn(redundant, text)
            rendered = [text] + [element.value for element in dashboard.markdown]
            rendered += [button.label for button in dashboard.button]
            rendered += [widget.label + ' '.join(widget.options) for widget in dashboard.selectbox]
            rendered += [widget.label for widget in dashboard.text_input]
            rendered += [metric.label + metric.value for metric in dashboard.metric]
            rendered += [table.value.to_csv() for table in (*dashboard.dataframe, *dashboard.table)]
            for chart in dashboard.get('plotly_chart'):
                rendered.append(chart.proto.spec)
            self.assertNotRegex(' '.join(rendered), r'[\u4e00-\u9fff]')
            self.assertEqual(sum('Customers · 3,444' in caption.value for caption in dashboard.caption), 1)
            self.assertEqual([expander.label for expander in dashboard.expander],
                             ['Profile details'] if key == 'recommendations' else [])
            if key != 'recommendations':
                self.assertFalse(dashboard.metric)
            else:
                self.assertEqual([metric.label for metric in dashboard.metric],
                                 ['Purchases (180d)', 'Distinct products purchased', 'Historical repeat-product share'])
        self.assertEqual(dashboard.session_state['workspace_page'], 'recommendations')

    def test_english_labels_preserve_existing_session_data_and_selections(self):
        from app.business_views import GROUP_NAMES
        legacy_names = {1: '\u7fa4 1\uff5c\u8f83\u4e45\u672a\u8d2d', 2: '\u7fa4 2\uff5c\u8fd1\u671f\u9ad8\u9891',
                        0: '\u7fa4 3\uff5c\u8fd1\u671f\u4f4e\u5ba2\u5355', 3: '\u7fa4 4\uff5c\u8fd1\u671f\u9ad8\u5ba2\u5355'}
        legacy_customers = self.workspace['customers'].assign(
            segment_name=self.workspace['customers'].cluster.map(legacy_names))
        original = legacy_customers.copy(deep=True)
        dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40)
        dashboard.session_state['uploaded_workspace'] = dict(self.workspace, customers=legacy_customers)
        dashboard.session_state['workspace_page'] = 'segments'
        dashboard.session_state['segment_group'] = legacy_names[1]
        dashboard.session_state['repurchase_group'] = legacy_names[3]
        dashboard.session_state['recommendation_customer_sort'] = '\u8fd1180\u5929\u8d2d\u4e70\u6b21\u6570\u4ece\u9ad8\u5230\u4f4e'
        dashboard.session_state['recommendation_customer'] = '12348'
        dashboard.run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(next(widget for widget in dashboard.selectbox if widget.label == 'Customer segment').value,
                         GROUP_NAMES[1])
        self.assertEqual(dashboard.dataframe[1].value['Customer segment'].unique().tolist(), [GROUP_NAMES[1]])
        pd.testing.assert_frame_equal(dashboard.session_state['uploaded_workspace']['customers'], original)
        self.open_business_page(dashboard, 'repurchase')
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.dataframe[0].value['Customer segment'].unique().tolist(), [GROUP_NAMES[3]])
        self.open_business_page(dashboard, 'recommendations')
        self.assertFalse(dashboard.exception)
        self.assertEqual(next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID').value, '12348')
        self.assertEqual(next(widget for widget in dashboard.selectbox if widget.label == 'Customer sort order').value,
                         'Purchases in last 180 days (descending)')
        pd.testing.assert_frame_equal(dashboard.session_state['uploaded_workspace']['customers'], original)

    def test_cached_chinese_ui_modules_reload_without_losing_uploaded_analysis(self):
        import importlib
        from types import ModuleType
        package = importlib.import_module('app')
        views = importlib.import_module('app.business_views')
        inference = importlib.import_module('app.upload_inference')
        legacy_pages = {key: (f'Old page {number}', icon)
                        for number, (key, (_, icon)) in enumerate(views.PAGES.items())}
        legacy_groups = {cluster: f'Old group {cluster}' for cluster in views.GROUP_NAMES}
        orphaned = ModuleType('app.business_views')
        orphaned.UI_TEXT_VERSION = 'zh-legacy'
        orphaned.PAGES = legacy_pages
        original = self.workspace['customers'].copy(deep=True)
        with patch.object(package, 'business_views', orphaned), \
                patch.object(views, 'UI_TEXT_VERSION', 'zh-legacy'), \
                patch.object(views, 'PAGES', legacy_pages), \
                patch.object(views, 'GROUP_NAMES', legacy_groups), \
                patch.object(inference, 'UI_TEXT_VERSION', 'zh-legacy'):
            dashboard = self.open_dashboard()
            self.assertFalse(dashboard.exception)
            self.assertEqual(dashboard.title[0].value, 'Customer segments')
            self.assertEqual([button.label for button in dashboard.sidebar.button],
                             ['Data import', 'Customer segments', 'Repurchase priorities',
                              'High-value customers', 'Product recommendations'])
            self.assertNotRegex(dashboard.dataframe[0].value.to_csv(), r'[\u4e00-\u9fff]')
            self.assertEqual(views.UI_TEXT_VERSION, 'en-1')
            self.assertEqual(inference.UI_TEXT_VERSION, 'en-1')
            pd.testing.assert_frame_equal(dashboard.session_state['uploaded_workspace']['customers'], original)

    def test_dashboard_keeps_analysis_date_between_segments_and_repurchase(self):
        dashboard = self.open_dashboard()
        names = ['Group 1 | Long inactive', 'Group 2 | Recent frequent buyers', 'Group 3 | Recent low order value', 'Group 4 | Recent high order value']
        summary = dashboard.dataframe[0].value
        self.assertFalse(dashboard.metric)
        self.assertIn('Matched 3,444 customers · Showing 50', [caption.value for caption in dashboard.caption])
        self.assertTrue(any(f'Overall reference: median 180-day spend £{self.workspace["customers"].monetary.median():,.2f}' in caption.value
                            for caption in dashboard.caption))
        self.assertEqual(summary['Customer segment'].tolist(), names)
        self.assertEqual(summary['Customers'].tolist(), [590, 816, 704, 1334])
        self.assertEqual(next(widget for widget in dashboard.selectbox if widget.label == 'Customer segment').options,
                         ['All segments'] + names)
        for chart in dashboard.get('plotly_chart'):
            import json
            spec = json.loads(chart.proto.spec)
            self.assertEqual(spec['layout']['yaxis']['categoryarray'], names)
            self.assertEqual(spec['layout']['yaxis']['autorange'], 'reversed')
        self.open_business_page(dashboard, 'repurchase')
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')
        self.assertFalse(any(widget.label == 'Analysis date' for widget in dashboard.selectbox))
        shown = dashboard.dataframe[0].value
        self.assertEqual(len(shown), 50)
        self.assertEqual(shown['Priority rank'].tolist(), list(range(1, 51)))
        self.assertTrue(shown['Repurchase score'].is_monotonic_decreasing)
        group = next(widget for widget in dashboard.selectbox if widget.label == 'Customer segment')
        self.assertEqual(group.options, ['All segments'] + names)
        group.set_value(group.options[1]).run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.dataframe[0].value['Customer segment'].nunique(), 1)
        self.assertIn('Matched 590 customers · Showing 50', [caption.value for caption in dashboard.caption])
        next(widget for widget in dashboard.selectbox if widget.label == 'Customers to display').set_value(25).run()
        self.assertEqual(len(dashboard.dataframe[0].value), 25)
        self.assertIn('Matched 590 customers · Showing 25', [caption.value for caption in dashboard.caption])
        next(widget for widget in dashboard.text_input if widget.label == 'Customer ID').set_value('not-a-customer').run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertTrue(dashboard.dataframe[0].value.empty)
        self.assertIn('Matched 0 customers · Showing 0', [caption.value for caption in dashboard.caption])

    def test_dashboard_high_value_only_exposes_actionable_customer_columns(self):
        from app.business_views import VALUE_COLUMNS
        dashboard = self.open_dashboard()
        dashboard.session_state['value_scope'] = 'All customers'  # A previous session must also use the fixed Top20% scope.
        self.open_business_page(dashboard, 'value')
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        shown = dashboard.dataframe[0].value
        self.assertEqual(len(shown), 50)
        self.assertEqual(list(shown), list(VALUE_COLUMNS.values()))
        self.assertNotIn('Attention level', shown)
        self.assertEqual(shown['Potential rank'].tolist(), list(range(1, 51)))
        self.assertFalse(dashboard.metric)
        self.assertIn('Top 20% by spending potential · **689 customers**', [caption.value for caption in dashboard.caption])
        self.assertIn('Matched 689 customers · Showing 50', [caption.value for caption in dashboard.caption])
        self.assertEqual(len(dashboard.dataframe[0].value), 50)
        self.assertEqual([widget.label for widget in dashboard.selectbox], ['Customers to display'])
        priority = self.workspace['priority']
        last_priority = priority.loc[priority.priority_band.eq('Top 20%')].sort_values('priority_rank').iloc[-1]
        next(widget for widget in dashboard.text_input if widget.label == 'Customer ID').set_value(last_priority.customer_id).run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.dataframe[0].value['Potential rank'].tolist(), [int(last_priority.priority_rank)])
        outside_priority = priority.loc[priority.priority_band.ne('Top 20%')].iloc[0].customer_id
        next(widget for widget in dashboard.text_input if widget.label == 'Customer ID').set_value(outside_priority).run()
        self.assertFalse(dashboard.exception)
        self.assertTrue(dashboard.dataframe[0].value.empty)
        self.assertIn('Matched 0 customers · Showing 0', [caption.value for caption in dashboard.caption])
        self.assertIn('Top 20% by spending potential · **689 customers**', [caption.value for caption in dashboard.caption])
        self.open_business_page(dashboard, 'segments')
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')

    def test_dashboard_recommendations_show_d_classifier_and_purchase_habits(self):
        from app.upload_inference import recommend_customer
        dashboard = self.open_dashboard()
        self.open_business_page(dashboard, 'recommendations')
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        shown = dashboard.dataframe[0].value
        self.assertEqual(len(shown), 10)
        self.assertEqual(shown['Product code'].nunique(), 10)
        self.assertNotIn('Ranking score', shown)
        self.assertNotIn('Recommendation type', shown)
        self.assertEqual(len(dashboard.dataframe), 1)
        self.assertEqual([metric.label for metric in dashboard.metric],
                         ['Purchases (180d)', 'Distinct products purchased', 'Historical repeat-product share'])
        self.assertEqual([expander.label for expander in dashboard.expander], ['Profile details'])
        self.assertEqual({widget.label for widget in dashboard.selectbox}, {'Customer ID', 'Customer sort order'})
        customer = next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID').value
        expected = recommend_customer(self.workspace, customer)
        self.assertTrue(expected.recommendation_method.eq('D_classifier').all())
        self.assertEqual(shown['Product code'].tolist(), expected.stock_code.tolist())
        self.assertNotIn('Recommendation reason', shown)
        self.assertEqual(shown['Purchases (180d)'].tolist(), expected.purchase_count_180d.tolist())
        profile = self.workspace['profiles'].loc[self.workspace['profiles'].customer_id.eq(customer)].iloc[0]
        self.assertEqual([metric.value for metric in dashboard.metric],
                         [str(int(profile.order_count)), str(int(profile.distinct_products)),
                          f'{profile.historical_repeat_share:.1%}'])
        details = dashboard.table[0].value
        self.assertEqual(len(details), 6 + int(pd.notna(profile.repeat_share_change)))
        self.assertEqual(details.loc['Top-3 product concentration', 'Value'], f'{profile.top3_concentration:.1%}')
        self.assertEqual(details.loc['Recent repeat-product share', 'Value'], f'{profile.recent_repeat_share:.1%}')
        if pd.isna(profile.repeat_share_change):
            self.assertNotIn('Change in repeat-product share', details.index)
        else:
            self.assertEqual(details.loc['Change in repeat-product share', 'Value'], f'{100 * profile.repeat_share_change:+.1f} pp')
        self.assertEqual(details.loc['Average basket size (products)', 'Value'], f'{profile.mean_basket_size:.1f} products')
        self.assertEqual(details.loc['Days since last purchase', 'Value'], f'{profile.recency_days:.1f} days')
        self.assertEqual(details.loc['Purchase history span', 'Value'], f'{profile.history_span_days:.1f} days')
        self.assertEqual(details.loc['Comparable orders', 'Value'], f'{profile.comparable_order_count:.0f} orders')
        picker = next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID')
        picker.set_value(self.workspace['profiles'].customer_id.min()).run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(len(dashboard.dataframe[0].value), 10)
        self.assertNotIn('Recommendation reason', dashboard.dataframe[0].value)

    def test_recommendation_customer_sort_preserves_selection_and_short_history_details(self):
        dashboard = self.open_dashboard()
        self.open_business_page(dashboard, 'recommendations')
        original = dashboard.dataframe[0].value.copy()
        customer = next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID').value
        next(widget for widget in dashboard.selectbox if widget.label == 'Customer sort order').set_value('Purchases in last 180 days (descending)').run()
        self.assertFalse(dashboard.exception)
        picker = next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID')
        self.assertEqual(picker.value, customer)
        pd.testing.assert_frame_equal(original, dashboard.dataframe[0].value)
        ids = picker.options
        profiles = self.workspace['profiles'].set_index('customer_id')
        self.assertEqual(set(ids), set(profiles.index))
        counts = profiles.loc[ids, 'order_count']
        self.assertTrue(counts.is_monotonic_decreasing)
        for left, right in zip(ids, ids[1:]):
            if profiles.loc[left, 'order_count'] == profiles.loc[right, 'order_count']:
                self.assertLess(left, right)
        single = profiles.loc[profiles.order_count.eq(1)].index[0]
        picker.set_value(single).run()
        self.assertFalse(dashboard.exception)
        self.assertEqual([metric.label for metric in dashboard.metric], ['Purchases (180d)', 'Distinct products purchased'])
        self.assertEqual(dashboard.metric[0].value, '1')
        details = dashboard.table[0].value
        self.assertEqual(len(details), 5)
        self.assertNotIn('Recent repeat-product share', details.index)
        self.assertNotIn('Change in repeat-product share', details.index)
        self.assertNotIn('Insufficient history', details['Value'].tolist())
        self.assertEqual(details.loc['Comparable orders', 'Value'], '0 orders')
        self.assertEqual(details.loc['Purchase history span', 'Value'], '0.0 days')
        self.assertEqual(len(dashboard.dataframe[0].value), 10)
        self.assertGreater(self.workspace['unknown_products'], 0)
        self.assertFalse(any('The recommended list only covers known products' in caption.value for caption in dashboard.caption))
        two_orders = profiles.loc[profiles.order_count.eq(2) & profiles.historical_repeat_share.eq(0)].index[0]
        next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID').set_value(two_orders).run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.metric[2].value, '0.0%')
        details = dashboard.table[0].value
        self.assertEqual(len(details), 6)
        self.assertEqual(details.loc['Recent repeat-product share', 'Value'], '0.0%')
        self.assertNotIn('Change in repeat-product share', details.index)
        complete = profiles.loc[profiles.repeat_share_change.notna()].index[0]
        next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID').set_value(complete).run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(len(dashboard.metric), 3)
        self.assertEqual(len(dashboard.table[0].value), 7)
        next(widget for widget in dashboard.selectbox if widget.label == 'Customer sort order').set_value('Customer ID (ascending)').run()
        picker = next(widget for widget in dashboard.selectbox if widget.label == 'Customer ID')
        self.assertEqual(picker.value, complete)
        self.assertEqual(picker.options, sorted(ids))

    def test_business_exports_strip_research_metadata_and_future_answers(self):
        from app.business_views import customer_table, value_table, product_table, business_export
        customer = pd.DataFrame({'rank': [1], 'CustomerID': ['a'], 'segment_name': ['Group A'], 'repurchase_score': [.7],
                                 'recency_days': [5], 'frequency': [3], 'monetary': [120.], 'aov': [40.],
                                 'repurchase_30d': [1], 'model_version': ['SECRET']})
        out = business_export(customer_table(customer), '2011-09-01')
        self.assertEqual(out['Customer ID'].tolist(), ['a'])
        self.assertNotIn('model_version', out)
        self.assertNotIn('repurchase_30d', out)
        self.assertEqual(out['Analysis date'].tolist(), ['2011-09-01'])
        priority = pd.DataFrame({'priority_rank': [2], 'customer_id': ['a'], 'priority_band': ['Top 20%'],
                                 'recency_days': [5], 'monetary_value_180d': [120.], 'baseline_rank': [7],
                                 'ranking_method': ['SECRET'], 'spend_30d': [999]})
        value = business_export(value_table(priority), '2011-09-01')
        self.assertEqual(value['Potential rank'].tolist(), [2])
        self.assertFalse({'baseline_rank', 'ranking_method', 'spend_30d', 'priority_band', 'Attention level'} & set(value))
        self.assertNotIn('Attention level', business_export(value.assign(**{'Attention level': 'old field'}), '2011-09-01'))
        products = pd.DataFrame({'rank': [1], 'stock_code': ['sku'], 'description': ['cup'],
                                 'purchased_in_past_180d': [True], 'purchase_count_180d': [2],
                                 'days_since_item_purchase': [5.], 'recommendation_method': ['D_classifier'],
                                 'ranking_score': [10.], 'target_items': ['SECRET']})
        profile = dict(order_count=3, distinct_products=5, historical_repeat_share=.4, top3_concentration=.6)
        exported = business_export(product_table(products), '2011-09-01', 'a', profile)
        self.assertEqual(exported['Customer historical repeat-product share'].iloc[0], .4)
        self.assertEqual(exported['Customer ID'].iloc[0], 'a')
        self.assertEqual(exported['Purchases (180d)'].iloc[0], 2)
        self.assertFalse({'ranking_score', 'target_items', 'recommendation_method', 'Recommendation reason', 'Recommendation type',
                          'recommendation_kind'} & set(exported))
        stale = exported.assign(**{'Recommendation reason': 'old field', 'Recommendation type': 'buy again'})
        self.assertFalse({'Recommendation reason', 'Recommendation type'} & set(business_export(stale, '2011-09-01')))


class UploadedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.upload_inference import DEMO, load_models, snapshots_from_history
        cls.models = load_models()
        cls.history = pd.read_csv(DEMO / 'purchase_history.csv',
                                  dtype={'customer_id': 'string', 'invoice': 'string', 'stock_code': 'string'},
                                  parse_dates=['invoice_date'])
        cls.small = cls.history.loc[cls.history.customer_id.isin(['12347', '12348'])].copy()
        cls.snapshots = snapshots_from_history(cls.small, pd.Timestamp('2011-12-01'))

    @staticmethod
    def csv(frame):
        return frame.to_csv(index=False).encode('utf-8-sig')

    def test_live_default_loads_frozen_d_classifier_and_matching_ease_catalog(self):
        from src.recommendation.features import FEATURE_SETS
        model = self.models['recommendation_model']
        self.assertEqual(self.models['recommendation_method'], 'D_classifier')
        self.assertEqual(model['group'], 'D')
        self.assertEqual(model['spec']['family'], 'classifier')
        self.assertEqual(model['features'], FEATURE_SETS['D'])
        self.assertEqual(len(model['features']), 34)
        self.assertEqual(model['spec']['leaves'], 15)
        self.assertEqual(model['spec']['trees'], 120)
        self.assertLessEqual(pd.Timestamp(model['latest_label_end']), pd.Timestamp('2011-09-01'))
        self.assertEqual(self.models['catalog'].stock_code.tolist(),
                         self.models['recommendation_ease']['catalog'].stock_code.tolist())

    def test_live_default_rejects_changed_model_package(self):
        from app.upload_inference import digest, load_models
        item_path = ROOT / 'artifacts/recommendation/current/model_bundle.joblib'
        with patch('app.upload_inference.digest', side_effect=lambda path: 'changed' if path == item_path else digest(path)):
            with self.assertRaisesRegex(ValueError, 'recommendation configuration'):
                load_models()

    def test_every_calendar_day_in_supported_range_builds_a_past_only_snapshot(self):
        from app.upload_inference import MIN_ANALYSIS_DATE, MAX_ANALYSIS_DATE, UploadError, validate_upload
        content = self.csv(self.small)
        for cutoff in pd.date_range(MIN_ANALYSIS_DATE, MAX_ANALYSIS_DATE, freq='D'):
            with self.subTest(date=str(cutoff.date())):
                snapshots, history, _ = validate_upload(content, cutoff)
                self.assertTrue(snapshots.snapshot_date.eq(cutoff).all())
                self.assertTrue(history.invoice_date.lt(cutoff).all())
                self.assertFalse(snapshots.customer_id.duplicated().any())
        for cutoff in ('2011-08-31', '2011-12-02'):
            with self.assertRaisesRegex(UploadError, '2011-09-01'):
                validate_upload(content, cutoff)

    def test_upload_rejects_missing_fields_future_answers_duplicate_rows_and_bad_dates(self):
        from app.upload_inference import UploadError, validate_upload
        broken = [self.small.drop(columns='invoice_date'),
                  self.small.assign(repurchase_30d=0),
                  pd.concat([self.small, self.small]),
                  self.small.assign(invoice_date='invalid'),
                  self.small.assign(quantity=np.inf)]
        for history in broken:
            with self.subTest(columns=list(history), rows=len(history)):
                with self.assertRaises(UploadError):
                    validate_upload(self.csv(history), '2011-12-01')
        for content in (b'', b'\xff\xfe', b'a,b\n1,2'):
            with self.assertRaises(UploadError):
                validate_upload(content, '2011-12-01')
        for date in ('2011-12-02', '2011-11-15T12:00:00', 'invalid', None, '2011-12-01T00:00:00Z'):
            with self.assertRaises(UploadError):
                validate_upload(self.csv(self.small), date)

    def test_upload_rejects_amount_mismatch_and_empty_eligible_history(self):
        from app.upload_inference import UploadError, validate_upload
        bad_amount = self.small.copy()
        bad_amount.loc[bad_amount.index[0], 'line_amount'] += 1
        for history in (bad_amount, self.small.assign(quantity=-1), self.small.assign(invoice_date='2011-12-01')):
            with self.assertRaises(UploadError):
                validate_upload(self.csv(history), '2011-12-01')
        with self.assertRaises(UploadError):
            validate_upload(self.csv(self.small), '2013-12-01')

    def test_upload_scores_new_customer_ids_from_inputs_and_changes_when_history_changes(self):
        from app.upload_inference import analyze_upload, recommend_customer
        before = analyze_upload(self.csv(self.small), '2011-12-01', self.models)
        renamed = self.small.copy()
        renamed.customer_id = 'new-' + renamed.customer_id
        after = analyze_upload(self.csv(renamed), '2011-12-01', self.models)
        self.assertEqual(after['customers'].CustomerID.tolist(), ['new-12347', 'new-12348'])
        np.testing.assert_allclose(before['customers'].repurchase_score, after['customers'].repurchase_score, rtol=0, atol=0)
        pd.testing.assert_frame_equal(recommend_customer(before, '12347'), recommend_customer(after, 'new-12347'))
        changed = self.small.copy()
        changed.quantity *= 2
        changed.line_amount *= 2
        changed_result = analyze_upload(self.csv(changed), '2011-12-01', self.models)
        np.testing.assert_allclose(changed_result['customers'].monetary, before['customers'].monetary * 2)
        self.assertFalse({'repurchase_30d', 'spend_30d', 'target_items'} & set(after['customers']))

    def test_future_transactions_do_not_change_any_generated_features_scores_or_recommendations(self):
        from app.upload_inference import analyze_upload, recommend_customer
        before = analyze_upload(self.csv(self.small), '2011-12-01', self.models)
        future = self.small.iloc[:1].copy()
        future['invoice_date'] = pd.Timestamp('2011-12-01')
        future['line_amount'] = 999999.
        after = analyze_upload(self.csv(pd.concat([self.small, future])), '2011-12-01', self.models)
        self.assertEqual(after['ignored_future_rows'], 1)
        for field in ('generated_snapshots', 'customers', 'priority', 'profiles'):
            pd.testing.assert_frame_equal(before[field], after[field])
        pd.testing.assert_frame_equal(recommend_customer(before, '12347'), recommend_customer(after, '12347'))

    def test_feature_builder_excludes_cutoff_and_future_transactions(self):
        from app.upload_inference import snapshots_from_history
        future = self.small.iloc[:1].copy()
        future['invoice_date'] = pd.Timestamp('2011-12-01')
        future['line_amount'] = 999999.
        actual = snapshots_from_history(pd.concat([self.small, future]), pd.Timestamp('2011-12-01'))
        pd.testing.assert_frame_equal(actual, self.snapshots)

    def test_short_but_complete_customer_history_is_supported(self):
        from app.upload_inference import analyze_upload, snapshots_from_history
        history = self.small.loc[self.small.invoice_date.ge(self.small.invoice_date.max() - pd.Timedelta(days=15))].copy()
        snapshots = snapshots_from_history(history, pd.Timestamp('2011-12-01'))
        result = analyze_upload(self.csv(history), '2011-12-01', self.models)
        self.assertEqual(len(result['customers']), len(snapshots))

    def test_new_inference_reproduces_november_model_assignments_scores_and_value_ranks(self):
        import json
        from app.upload_inference import analyze_upload, recommend_customer
        actual = analyze_upload(self.csv(self.history), '2011-11-01', self.models)
        saved = pd.read_csv(ROOT / 'artifacts/integration/current/test_predictions.csv', dtype={'customer_id': 'string'})
        saved = saved.loc[saved.snapshot_date.eq('2011-11-01')].sort_values('customer_id')
        np.testing.assert_array_equal(actual['customers'].cluster, saved.cluster)
        np.testing.assert_allclose(actual['customers'].repurchase_score, saved.repurchase_score, rtol=0, atol=1e-12)
        ranks = pd.read_csv(ROOT / 'artifacts/spending/current/customer_priority.csv', dtype={'customer_id': 'string'})
        ranks = ranks.loc[ranks.snapshot_date.eq('2011-11-01')].sort_values('priority_rank')
        self.assertEqual(actual['priority'].customer_id.tolist(), ranks.customer_id.tolist())
        lists = pd.read_csv(ROOT / 'artifacts/recommendation/current/test_lists.csv', dtype={'customer_id': 'string'})
        saved_list = lists.loc[lists.snapshot_date.eq('2011-11-01') & lists.customer_id.eq('12347')
                               & lists.method.eq('D_classifier')].iloc[0]
        recommendations = recommend_customer(actual, '12347')
        self.assertEqual(recommendations.stock_code.tolist(), json.loads(saved_list.recommended_items))
        np.testing.assert_allclose(recommendations.ranking_score, json.loads(saved_list.ranking_scores), rtol=0, atol=1e-12)

    def test_uploaded_d_classifier_reproduces_september_and_october_full_catalog_lists(self):
        import json
        from app.upload_inference import analyze_upload, recommend_customer
        lists = pd.read_csv(ROOT / 'artifacts/recommendation/current/test_lists.csv', dtype={'customer_id': 'string'})
        content = self.csv(self.history)
        for date in ('2011-09-01', '2011-10-01'):
            actual = analyze_upload(content, date, self.models)
            for customer in ('12347', '14646'):
                with self.subTest(date=date, customer=customer):
                    expected = lists.loc[lists.snapshot_date.eq(date) & lists.customer_id.eq(customer)
                                         & lists.method.eq('D_classifier')].iloc[0]
                    recommendations = recommend_customer(actual, customer)
                    self.assertEqual(recommendations.stock_code.tolist(), json.loads(expected.recommended_items))
                    np.testing.assert_allclose(recommendations.ranking_score, json.loads(expected.ranking_scores),
                                               rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(recommendations.purchase_count_180d,
                                                  json.loads(expected.purchase_counts_180d))

    def test_recommendation_rejects_old_workspace_and_unknown_customer(self):
        from app.upload_inference import UploadError, analyze_upload, recommend_customer
        workspace = analyze_upload(self.csv(self.small), '2011-12-01', self.models)
        with self.assertRaisesRegex(UploadError, 'upload'):
            recommend_customer(dict(workspace, recommendation_method='personal_repeat'), '12347')
        with self.assertRaisesRegex(UploadError, 'No purchase history'):
            recommend_customer(workspace, 'missing')


class StageFiveTests(unittest.TestCase):
    def test_repurchase_rhythm_uses_only_prior_180_day_invoices(self):
        snapshots = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-03-01']),
            'customer_id': pd.Series(['a'], dtype='string'),
            'split': ['train'], 'invoice_frequency_180d': [2],
            'invoice_count_90d': [1], 'invoice_count_30d': [1],
            'customer_history_days': [200],
            'recency_days': [19], 'monetary_value_180d': [80.],
            'average_order_value_180d': [40.], 'product_diversity_180d': [5],
        })
        invoices = pd.DataFrame({
            'customer_id': pd.Series(['a'] * 4, dtype='string'),
            'invoice': ['old', 'first', 'second', 'future'],
            'invoice_date': pd.to_datetime([
                '2010-08-01', '2010-11-01', '2011-02-10', '2011-03-05']),
        })
        result = add_repurchase_features(snapshots, invoices).iloc[0]
        without_future = add_repurchase_features(snapshots, invoices.iloc[:3]).iloc[0]
        for feature in ('active_30d_periods_180d', 'last_invoice_gap_days_180d',
                        'has_two_invoices_180d'):
            self.assertEqual(result[feature], without_future[feature])
        self.assertEqual(result.active_30d_periods_180d, 2)
        self.assertEqual(result.has_two_invoices_180d, 1)
        self.assertAlmostEqual(result.last_invoice_gap_days_180d, 101)
        self.assertEqual(result.recent90_invoice_share, .5)

    def test_repurchase_forward_fold_purges_unfinished_labels(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-01-01', '2011-02-01', '2011-03-01']),
            'customer_id': ['a', 'b', 'c'],
        })
        fit, hold = repurchase_forward_fold(frame, pd.Timestamp('2011-03-01'))
        self.assertEqual(fit.customer_id.tolist(), ['a'])
        self.assertEqual(hold.customer_id.tolist(), ['c'])

    def test_repurchase_candidates_exclude_rfm_rule_and_old_cluster_distances(self):
        all_features = sum(REPURCHASE_FEATURE_SETS.values(), [])
        self.assertNotIn('repurchase_30d', all_features)
        self.assertFalse(any(name.startswith('cluster_distance') for name in all_features))
        self.assertEqual(len(REPURCHASE_FEATURE_SETS['base_8_plus_trend_and_rhythm']), 14)
        model = make_repurchase_model('SplineLR', 0.1, REPURCHASE_FEATURE_SETS['base_8'])
        self.assertIn('recency_spline',
                      [name for name, _, _ in model.named_steps['features'].transformers])

    def test_repurchase_selection_respects_forward_and_top20_guards(self):
        candidates = pd.DataFrame([
            {'feature_set': 'base_8', 'model': 'LR', 'ap': .50,
             'mean_fold_ap': .48, 'precision_at_20pct': .55, 'feature_count': 8},
            {'feature_set': 'base_8', 'model': 'XGBoost', 'ap': .60,
             'mean_fold_ap': .47, 'precision_at_20pct': .60, 'feature_count': 8},
            {'feature_set': 'base_8_plus_trend', 'model': 'LightGBM', 'ap': .56,
             'mean_fold_ap': .50, 'precision_at_20pct': .56, 'feature_count': 11},
        ])
        selected = choose_repurchase_winner(candidates)
        self.assertEqual(selected.loc[selected.selected, 'model'].tolist(), ['LightGBM'])
        self.assertFalse(selected.loc[selected.model.eq('XGBoost'), 'eligible'].iloc[0])

    def test_customer_bootstrap_keeps_matching_predictions_paired(self):
        rows = []
        for model in ('LR', 'XGBoost'):
            for customer, label, score in [('a', 1, .8), ('b', 0, .3),
                                           ('c', 1, .7), ('d', 0, .1)]:
                rows.append({'snapshot_date': pd.Timestamp('2011-06-01'),
                             'customer_id': customer, 'repurchase_30d': label,
                             'feature_set': 'base_8', 'model': model, 'score': score})
        comparison = pd.DataFrame([
            {'feature_set': 'base_8', 'model': 'XGBoost', 'ap': 1.0, 'selected': True},
            {'feature_set': 'base_8', 'model': 'LR', 'ap': 1.0, 'selected': False},
        ])
        result = paired_customer_bootstrap(pd.DataFrame(rows), comparison, draws=20)
        self.assertTrue(result.observed_ap_delta.eq(0).all())
        self.assertTrue(result.bootstrap_low_2_5.eq(0).all())
        self.assertTrue(result.bootstrap_high_97_5.eq(0).all())

    def test_rolling_fold_purges_training_labels_crossing_boundary(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-01-01', '2011-02-01',
                                             '2011-03-01', '2011-04-01']),
            'customer_id': ['a', 'b', 'c', 'd'],
        })
        fit, validation = make_fold(frame, pd.Timestamp('2011-03-01'))
        self.assertEqual(fit.customer_id.tolist(), ['a'])
        self.assertEqual(validation.customer_id.tolist(), ['c'])

    def test_behavior_set_uses_only_snapshot_features(self):
        row = {feature: 1.0 for feature in FEATURE_SETS['A_base_8']}
        row.update(invoice_frequency_180d=4, invoice_count_90d=2,
                   invoice_count_30d=1, customer_history_days=120,
                   repurchase_30d=0)
        fit = pd.DataFrame([row])
        validation = pd.DataFrame([{**row, 'repurchase_30d': 1}])
        trained, heldout, columns = fold_features(
            fit, validation, 'C_base_8_plus_4_behavior')
        self.assertEqual(len(columns), 12)
        self.assertTrue((trained[columns] == heldout[columns]).all().all())
        self.assertNotIn('repurchase_30d', columns)

    def test_rolling_cluster_distances_ignore_validation_extremes(self):
        rows = []
        for i in range(18):
            group = i // 6
            rows.append({
                'recency_days': 10 + group * 55 + i % 6,
                'invoice_frequency_180d': 1 + group * 3 + i % 2,
                'monetary_value_180d': 100 + group * 700 + i * 5,
                'average_order_value_180d': 100 + group * 80 + i,
                'product_diversity_180d': 5 + group * 20 + i % 3,
                'invoice_count_30d': group,
                'invoice_count_90d': 1 + group * 2,
                'customer_history_days': 30 + group * 100 + i,
            })
        fit = pd.DataFrame(rows)
        normal = fit.iloc[[0]].copy()
        extreme = normal.copy()
        extreme.loc[:, 'monetary_value_180d'] = 1_000_000
        first, _, _ = fold_features(fit, normal, 'B_base_8_plus_3_distances')
        second, _, _ = fold_features(fit, extreme, 'B_base_8_plus_3_distances')
        distance_columns = FEATURE_SETS['B_base_8_plus_3_distances'][-3:]
        np.testing.assert_allclose(first[distance_columns], second[distance_columns])

    def test_weight_selection_uses_only_validation_ap_and_prefers_half_on_tie(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        lr = np.array([.1, .8, .2, .6, .7, .3])
        gbdt = np.array([.4, .5, .1, .9, .6, .2])
        weight, table = choose_weight(y, lr, gbdt)
        expected = [average_precision_score(y, w * lr + (1 - w) * gbdt) for w in WEIGHTS]
        np.testing.assert_allclose(table.validation_average_precision, expected)
        self.assertEqual(table.loc[table.selected, 'lr_weight'].tolist(), [weight])
        self.assertEqual(choose_weight(y, lr, lr)[0], .5)

    def test_top20_quota_is_allocated_within_each_month(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 3 + ['2021-02-01'] * 6),
            'customer_id': [str(i) for i in range(9)],
            'split': ['validation'] * 9,
        })
        scores = np.array([.9, .8, .7, .1, .2, .3, .4, .5, .6])
        labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        metrics = top20_metrics(frame, scores, labels)
        self.assertEqual(metrics['top20_count'], 3)  # ceil(.2*3) + ceil(.2*6)
        self.assertEqual(metrics['top20_hits'], 2)
        self.assertAlmostEqual(metrics['precision_at_20pct'], 2 / 3)


class StageSixTests(unittest.TestCase):
    def test_model_selection_uses_validation_ap(self):
        rows = pd.DataFrame([
            {'feature_set': 'baseline', 'model': 'rfm', 'validation_average_precision': .9},
            {'feature_set': 'A_base_8', 'model': 'GBDT', 'validation_average_precision': .6},
            {'feature_set': 'B_base_8_plus_3_distances', 'model': 'LR',
             'validation_average_precision': .59},
        ])
        winner = select_model(rows)
        self.assertEqual((winner.feature_set, winner.model), ('A_base_8', 'GBDT'))

    def test_threshold_is_selected_on_given_labels_and_scores(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .4, .5, .9])
        threshold = choose_f1_threshold(labels, scores)
        self.assertEqual(threshold, .5)

    def test_top20_mask_matches_existing_monthly_metric(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 3 + ['2021-02-01'] * 6),
            'customer_id': [str(i) for i in range(9)],
            'split': ['test'] * 9,
        })
        scores = np.array([.9, .8, .7, .1, .2, .3, .4, .5, .6])
        labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        chosen = top20_mask(frame, scores)
        metrics = top20_metrics(frame, scores, labels)
        self.assertEqual(int(chosen.sum()), metrics['top20_count'])
        self.assertEqual(int(labels[chosen].sum()), metrics['top20_hits'])

    def test_forward_calibration_uses_only_completed_labels(self):
        dates = pd.date_range('2010-10-01', '2011-04-01', freq='MS')
        frame = pd.DataFrame({
            'snapshot_date': dates,
            'customer_id': [str(i) for i in range(len(dates))],
            'repurchase_30d': [0, 1, 0, 1, 0, 1, 0],
        })
        fit, calibration, target = forward_parts(frame, pd.Timestamp('2011-03-01'))
        self.assertEqual(fit.snapshot_date.dt.strftime('%Y-%m-%d').tolist(),
                         ['2010-10-01'])
        self.assertEqual(calibration.snapshot_date.dt.strftime('%Y-%m-%d').tolist(),
                         ['2010-11-01', '2010-12-01', '2011-01-01'])
        self.assertEqual(target.snapshot_date.dt.strftime('%Y-%m-%d').tolist(),
                         ['2011-03-01'])
        self.assertNotIn(pd.Timestamp('2011-02-01'), calibration.snapshot_date.tolist())

    def test_calibration_candidate_must_improve_brier_and_gap_without_rank_loss(self):
        baseline = pd.Series({'method': BASELINE, 'mean_brier': .20,
                              'mean_absolute_calibration_gap': .08,
                              'mean_ece_10': .09, 'mean_ap': .60,
                              'mean_precision_at_20pct': .65})
        better = baseline.copy()
        better['method'] = 'sigmoid_1m'
        better['mean_brier'] = .18
        better['mean_absolute_calibration_gap'] = .05
        better['mean_ece_10'] = .08
        self.assertTrue(passes(better, baseline))
        worse_rank = better.copy()
        worse_rank['mean_ap'] = .58
        self.assertFalse(passes(worse_rank, baseline))
        tuning = pd.DataFrame([
            {**baseline.to_dict(), 'months': 3},
            {**better.to_dict(), 'months': 3},
            {**worse_rank.to_dict(), 'method': 'sigmoid_2m', 'months': 3},
        ])
        # The selector operates on per-month metrics, not precomputed means.
        monthly = pd.DataFrame([{
            'method': row['method'], 'target_date': str(month), 'brier': row['mean_brier'],
            'absolute_calibration_gap': row['mean_absolute_calibration_gap'],
            'ece_10': row['mean_ece_10'], 'average_precision': row['mean_ap'],
            'precision_at_20pct': row['mean_precision_at_20pct'],
        } for row in tuning.to_dict('records') for month in range(3)])
        self.assertEqual(choose_candidate(monthly), 'sigmoid_1m')

    def test_incumbent_ranking_guard_checks_both_metrics(self):
        incumbent = pd.Series({'mean_ap': .60, 'mean_precision_at_20pct': .65})
        self.assertTrue(ranking_guard(pd.Series({
            'mean_ap': .596, 'mean_precision_at_20pct': .647}), incumbent))
        self.assertFalse(ranking_guard(pd.Series({
            'mean_ap': .594, 'mean_precision_at_20pct': .65}), incumbent))
        self.assertFalse(ranking_guard(pd.Series({
            'mean_ap': .60, 'mean_precision_at_20pct': .644}), incumbent))




class EnglishRepositoryTests(unittest.TestCase):
    def test_public_documentation_and_source_text_are_english(self):
        import re
        paths = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md')),
                 ROOT / 'docs/project_learning_guide.html']
        paths += [path for directory in ('app', 'src', 'pipelines', 'archive', 'tests')
                  for path in (ROOT / directory).rglob('*.py')]
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertIsNone(re.search(r'[\u3400-\u9fff]', path.read_text(encoding='utf-8-sig')))
                self.assertIsNone(re.search(r'[\u3400-\u9fff]', path.name))

    def test_public_markdown_local_links_resolve(self):
        import re
        from urllib.parse import unquote, urlsplit
        for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]:
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8-sig')):
                if urlsplit(target).scheme or target.startswith('#'):
                    continue
                local = (path.parent / unquote(target.split('#', 1)[0])).resolve()
                with self.subTest(document=path.name, target=target):
                    self.assertTrue(local.exists())


class RepositoryLayoutTests(unittest.TestCase):
    def test_current_interface_and_domain_code_do_not_import_archive(self):
        import ast
        for directory in ('app', 'src'):
            for path in (ROOT / directory).rglob('*.py'):
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                dependencies = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        dependencies.append(node.module)
                    elif isinstance(node, ast.Import):
                        dependencies.extend(item.name for item in node.names)
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assertFalse(any(name.split('.')[0] in {
                        'archive', 'stage4', 'stage5', 'stage6', 'value_prediction',
                        'recommendation'} for name in dependencies))

    def test_model_locations_and_pipeline_entries_are_explicit(self):
        import importlib
        from src.paths import ROOT as project_root
        self.assertEqual(ROOT, project_root)
        for name in ('snapshots', 'prepare_demo', 'segmentation', 'repurchase',
                     'integrate', 'spending', 'recommendation'):
            with self.subTest(pipeline=name):
                importlib.import_module('pipelines.' + name)
        for name in ('segmentation', 'repurchase', 'integration', 'spending', 'recommendation'):
            self.assertTrue((ROOT / 'artifacts' / name / 'current').is_dir())
        self.assertTrue((ROOT / 'archive/artifacts').is_dir())

if __name__ == '__main__':
    unittest.main()
