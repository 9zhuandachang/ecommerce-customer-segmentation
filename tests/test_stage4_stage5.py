"""Regression checks for cluster comparison and formal validation experiments."""

import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from streamlit.testing.v1 import AppTest

from stage4.compare_segmentation import (behavior_features, candidates,
                                         customer_holdout, one_state_per_customer)
from stage4.stability import align_clusters
from stage5.extended import FEATURE_SETS, fold_features, make_fold
from stage5.experiments import WEIGHTS, choose_weight, top20_metrics
from stage5.compare_repurchase import (FEATURE_SETS as REPURCHASE_FEATURE_SETS,
                                       add_features as add_repurchase_features,
                                       choose_winner as choose_repurchase_winner,
                                       forward_fold as repurchase_forward_fold,
                                       make_model as make_repurchase_model,
                                       paired_customer_bootstrap)
from stage6.finalize import choose_f1_threshold, select_model, top20_mask
from stage6.optimize import BASELINE, choose_candidate, forward_parts, passes, ranking_guard
from app.stage6_data import ROOT, export_review, load_stage6_replay, review_list


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
        self.assertEqual(dashboard.title[0].value, '数据导入')
        self.assertEqual(dashboard.session_state['workspace_page'], 'upload')
        self.assertEqual([button.label for button in dashboard.sidebar.button],
                         ['数据导入', '客户分群', '复购优先名单', '高消费潜力', '画像与商品推荐'])
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
        self.assertTrue(any('本批客户 · 3,444位' in caption.value for caption in dashboard.caption))
        next(button for button in dashboard.button if button.key == 'change_data').click().run()
        with patch('streamlit.file_uploader', return_value=BytesIO(b'bad,file\n1,2')):
            next(button for button in dashboard.button if button.key == 'run_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertTrue(dashboard.error)
        self.assertFalse(dashboard.dataframe)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')
        next(button for button in dashboard.button if button.key == 'cancel_upload').click().run()
        self.assertFalse(dashboard.exception)
        self.assertTrue(any('本批客户 · 3,444位' in caption.value for caption in dashboard.caption))

    def test_old_rule_session_requires_regeneration_before_opening_business_pages(self):
        dashboard = AppTest.from_file(str(ROOT / 'app/dashboard.py'), default_timeout=40)
        dashboard.session_state['uploaded_workspace'] = dict(self.workspace, recommendation_method='personal_repeat')
        dashboard.session_state['workspace_page'] = 'recommendations'
        dashboard.run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.title[0].value, '数据导入')
        self.assertTrue(any('推荐方案已更新' in message.value for message in dashboard.info))
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
        self.assertEqual(dashboard.title[0].value, '数据导入')
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
            '群 3｜近期低客单', '群 1｜较久未购', '群 2｜近期高频', '群 4｜近期高客单'])
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(shown.drop(columns='segment_name'), before.drop(columns='segment_name'))
        self.assertEqual(group_summary(shown).segment_name.tolist(), list(GROUP_NAMES.values()))
        for cluster, name in GROUP_NAMES.items():
            filtered = customer_filter(shown, name)
            self.assertEqual(filtered.cluster.tolist(), [cluster])
            exported = business_export(customer_table(filtered, include_score=False), '2011-09-01')
            self.assertEqual(exported['客户群体'].tolist(), [name])
            self.assertEqual(exported['客户 ID'].tolist(), frame.loc[frame.cluster.eq(cluster), 'CustomerID'].tolist())

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
        self.assertEqual([button.label for button in dashboard.sidebar.button], ['数据导入'] + [value[0] for value in PAGES.values()])
        self.assertEqual(len(dashboard.sidebar.radio), 0)
        self.assertEqual(len(dashboard.radio), 0)
        for key in PAGES:
            self.open_business_page(dashboard, key)
            self.assertFalse(dashboard.exception)
            self.assertFalse(dashboard.error)
            text = ' '.join(str(element.value) for category in ('title', 'caption', 'subheader', 'info')
                            for element in getattr(dashboard, category))
            for redundant in ('sha256', 'XGBoost', 'LightGBM', 'NDCG', 'ROC-AUC', 'Brier', '验证期', '复核', '模型版本'):
                self.assertNotIn(redundant, text)
            self.assertEqual(sum('本批客户 · 3,444位' in caption.value for caption in dashboard.caption), 1)
            self.assertEqual([expander.label for expander in dashboard.expander],
                             ['画像详情'] if key == 'recommendations' else [])
            if key != 'recommendations':
                self.assertFalse(dashboard.metric)
            else:
                self.assertEqual([metric.label for metric in dashboard.metric],
                                 ['近180天购买次数', '购买商品种数', '历史订单老商品平均占比'])
        self.assertEqual(dashboard.session_state['workspace_page'], 'recommendations')

    def test_dashboard_keeps_analysis_date_between_segments_and_repurchase(self):
        dashboard = self.open_dashboard()
        names = ['群 1｜较久未购', '群 2｜近期高频', '群 3｜近期低客单', '群 4｜近期高客单']
        summary = dashboard.dataframe[0].value
        self.assertFalse(dashboard.metric)
        self.assertIn('共匹配 3,444 位客户，当前显示 50 位', [caption.value for caption in dashboard.caption])
        self.assertTrue(any(f'近180天消费中位数 £{self.workspace["customers"].monetary.median():,.2f}' in caption.value
                            for caption in dashboard.caption))
        self.assertEqual(summary['客户群体'].tolist(), names)
        self.assertEqual(summary['客户数'].tolist(), [590, 816, 704, 1334])
        self.assertEqual(next(widget for widget in dashboard.selectbox if widget.label == '客户群体').options,
                         ['全部群体'] + names)
        for chart in dashboard.get('plotly_chart'):
            import json
            spec = json.loads(chart.proto.spec)
            self.assertEqual(spec['layout']['yaxis']['categoryarray'], names)
            self.assertEqual(spec['layout']['yaxis']['autorange'], 'reversed')
        self.open_business_page(dashboard, 'repurchase')
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(dashboard.session_state['uploaded_workspace']['date'], '2011-12-01')
        self.assertFalse(any(widget.label == '分析日期' for widget in dashboard.selectbox))
        shown = dashboard.dataframe[0].value
        self.assertEqual(len(shown), 50)
        self.assertEqual(shown['优先名次'].tolist(), list(range(1, 51)))
        self.assertTrue(shown['复购评分'].is_monotonic_decreasing)
        group = next(widget for widget in dashboard.selectbox if widget.label == '客户群体')
        self.assertEqual(group.options, ['全部群体'] + names)
        group.set_value(group.options[1]).run()
        self.assertFalse(dashboard.exception)
        self.assertEqual(dashboard.dataframe[0].value['客户群体'].nunique(), 1)
        self.assertIn('共匹配 590 位客户，当前显示 50 位', [caption.value for caption in dashboard.caption])
        next(widget for widget in dashboard.selectbox if widget.label == '显示客户数').set_value(25).run()
        self.assertEqual(len(dashboard.dataframe[0].value), 25)
        self.assertIn('共匹配 590 位客户，当前显示 25 位', [caption.value for caption in dashboard.caption])
        next(widget for widget in dashboard.text_input if widget.label == '客户 ID').set_value('not-a-customer').run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertTrue(dashboard.dataframe[0].value.empty)
        self.assertIn('共匹配 0 位客户，当前显示 0 位', [caption.value for caption in dashboard.caption])

    def test_dashboard_high_value_only_exposes_actionable_customer_columns(self):
        from app.business_views import VALUE_COLUMNS
        dashboard = self.open_dashboard()
        self.open_business_page(dashboard, 'value')
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        shown = dashboard.dataframe[0].value
        self.assertEqual(len(shown), 50)
        self.assertEqual(list(shown), list(VALUE_COLUMNS.values()))
        self.assertTrue(shown['关注级别'].eq('优先关注').all())
        self.assertEqual(shown['潜力排名'].tolist(), list(range(1, 51)))
        self.assertFalse(dashboard.metric)
        self.assertIn('前20%消费潜力客户 · **689位**', [caption.value for caption in dashboard.caption])
        self.assertIn('共匹配 689 位客户，当前显示 50 位', [caption.value for caption in dashboard.caption])
        self.assertEqual(len(dashboard.dataframe[0].value), 50)
        next(widget for widget in dashboard.selectbox if widget.label == '客户范围').set_value('全部客户').run()
        self.assertIn('共匹配 3,444 位客户，当前显示 50 位', [caption.value for caption in dashboard.caption])
        next(widget for widget in dashboard.text_input if widget.label == '客户 ID').set_value('not-a-customer').run()
        self.assertFalse(dashboard.exception)
        self.assertTrue(dashboard.dataframe[0].value.empty)
        self.assertIn('共匹配 0 位客户，当前显示 0 位', [caption.value for caption in dashboard.caption])
        self.assertIn('前20%消费潜力客户 · **689位**', [caption.value for caption in dashboard.caption])
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
        self.assertEqual(shown['商品编码'].nunique(), 10)
        self.assertNotIn('排序分数', shown)
        self.assertEqual(len(dashboard.dataframe), 1)
        self.assertEqual([metric.label for metric in dashboard.metric],
                         ['近180天购买次数', '购买商品种数', '历史订单老商品平均占比'])
        self.assertEqual([expander.label for expander in dashboard.expander], ['画像详情'])
        self.assertEqual({widget.label for widget in dashboard.selectbox}, {'客户 ID'})
        customer = next(widget for widget in dashboard.selectbox if widget.label == '客户 ID').value
        expected = recommend_customer(self.workspace, customer)
        self.assertTrue(expected.recommendation_method.eq('D_classifier').all())
        self.assertEqual(shown['商品编码'].tolist(), expected.stock_code.tolist())
        self.assertTrue(shown['推荐依据'].eq('结合购买记录、客户习惯与商品关联').all())
        profile = self.workspace['profiles'].loc[self.workspace['profiles'].customer_id.eq(customer)].iloc[0]
        self.assertEqual([metric.value for metric in dashboard.metric],
                         [str(int(profile.order_count)), str(int(profile.distinct_products)),
                          f'{profile.historical_repeat_share:.1%}'])
        self.assertIn(f'常购商品集中度：**{profile.top3_concentration:.1%}**',
                      [caption.value for caption in dashboard.expander[0].caption])
        picker = next(widget for widget in dashboard.selectbox if widget.label == '客户 ID')
        picker.set_value(picker.options[0]).run()
        self.assertFalse(dashboard.exception)
        self.assertFalse(dashboard.error)
        self.assertEqual(len(dashboard.dataframe[0].value), 10)

    def test_business_exports_strip_research_metadata_and_future_answers(self):
        from app.business_views import customer_table, value_table, product_table, business_export
        customer = pd.DataFrame({'rank': [1], 'CustomerID': ['a'], 'segment_name': ['群A'], 'repurchase_score': [.7],
                                 'recency_days': [5], 'frequency': [3], 'monetary': [120.], 'aov': [40.],
                                 'repurchase_30d': [1], 'model_version': ['SECRET']})
        out = business_export(customer_table(customer), '2011-09-01')
        self.assertEqual(out['客户 ID'].tolist(), ['a'])
        self.assertNotIn('model_version', out)
        self.assertNotIn('repurchase_30d', out)
        self.assertEqual(out['分析日期'].tolist(), ['2011-09-01'])
        priority = pd.DataFrame({'priority_rank': [2], 'customer_id': ['a'], 'priority_band': ['Top 20%'],
                                 'recency_days': [5], 'monetary_value_180d': [120.], 'baseline_rank': [7],
                                 'ranking_method': ['SECRET'], 'spend_30d': [999]})
        value = business_export(value_table(priority), '2011-09-01')
        self.assertEqual(value['潜力排名'].tolist(), [2])
        self.assertFalse({'baseline_rank', 'ranking_method', 'spend_30d'} & set(value))
        products = pd.DataFrame({'rank': [1], 'stock_code': ['sku'], 'description': ['cup'],
                                 'purchased_in_past_180d': [True], 'purchase_count_180d': [2],
                                 'days_since_item_purchase': [5.], 'recommendation_method': ['D_classifier'],
                                 'ranking_score': [10.], 'target_items': ['SECRET']})
        profile = dict(order_count=3, distinct_products=5, historical_repeat_share=.4, top3_concentration=.6)
        exported = business_export(product_table(products), '2011-09-01', 'a', profile)
        self.assertEqual(exported['客户历史重复购买比例'].iloc[0], .4)
        self.assertEqual(exported['客户 ID'].iloc[0], 'a')
        self.assertEqual(exported['推荐依据'].iloc[0], '结合购买记录、客户习惯与商品关联')
        self.assertFalse({'ranking_score', 'target_items', 'recommendation_method'} & set(exported))


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
        from recommendation.item_features import FEATURE_SETS
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
        item_path = ROOT / 'recommendation/outputs/item_v3_1/model_bundle.joblib'
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
        saved = pd.read_csv(ROOT / 'stage6/outputs/reanalysis_v2/test_predictions.csv', dtype={'customer_id': 'string'})
        saved = saved.loc[saved.snapshot_date.eq('2011-11-01')].sort_values('customer_id')
        np.testing.assert_array_equal(actual['customers'].cluster, saved.cluster)
        np.testing.assert_allclose(actual['customers'].repurchase_score, saved.repurchase_score, rtol=0, atol=1e-12)
        ranks = pd.read_csv(ROOT / 'value_prediction/outputs/ranking_comparison_v1/customer_priority.csv', dtype={'customer_id': 'string'})
        ranks = ranks.loc[ranks.snapshot_date.eq('2011-11-01')].sort_values('priority_rank')
        self.assertEqual(actual['priority'].customer_id.tolist(), ranks.customer_id.tolist())
        lists = pd.read_csv(ROOT / 'recommendation/outputs/item_v3_1/test_lists.csv', dtype={'customer_id': 'string'})
        saved_list = lists.loc[lists.snapshot_date.eq('2011-11-01') & lists.customer_id.eq('12347')
                               & lists.method.eq('D_classifier')].iloc[0]
        recommendations = recommend_customer(actual, '12347')
        self.assertEqual(recommendations.stock_code.tolist(), json.loads(saved_list.recommended_items))
        np.testing.assert_allclose(recommendations.ranking_score, json.loads(saved_list.ranking_scores), rtol=0, atol=1e-12)

    def test_uploaded_d_classifier_reproduces_september_and_october_full_catalog_lists(self):
        import json
        from app.upload_inference import analyze_upload, recommend_customer
        lists = pd.read_csv(ROOT / 'recommendation/outputs/item_v3_1/test_lists.csv', dtype={'customer_id': 'string'})
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
        with self.assertRaisesRegex(UploadError, '重新上传'):
            recommend_customer(dict(workspace, recommendation_method='personal_repeat'), '12347')
        with self.assertRaisesRegex(UploadError, '找不到该客户'):
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



if __name__ == '__main__':
    unittest.main()
