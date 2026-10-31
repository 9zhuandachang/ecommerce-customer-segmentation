"""Long-term next-basket boundaries, recommender equations, and fair evaluation."""

from src.paths import ROOT

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.recommendation.data import ROOT, history_counts, make_training, make_context, next_basket_labels, read_inputs
from pipelines.recommendation_baselines import choose_config, choose_method
from src.recommendation.metrics import evaluate, per_customer_metrics, paired_intervals
from archive.adapters.recommendation_data import export_recommendations, validate_recommendations, load_recommendations, load_comparison_list
from src.recommendation.algorithms import (configurations, ease_weights, fit_model, implicit_loss,
                                    implicit_step, recommend, recommendation_table, topk)
from src.recommendation.profiles import (FEATURES, purchase_profiles, attach_repeat_target,
                                     mature_examples, fit_propensity, predict_propensity)
from archive.recommendation.hybrid import (NAMES as HYBRID_NAMES, prepare_routes, fuse_routes,
                                   mixing_weights, metric_context, fast_metrics)
from pipelines.recommendation_profiles import choose_default, recommend_all
from archive.adapters.profile_recommendation_data import (load_profile_package, profile_recommendation_list,
                                             export_profile_recommendations, EXPORT_COLUMNS as PROFILE_EXPORT_COLUMNS)
from src.recommendation.features import (ALL_FEATURES, FEATURE_SETS, FEATURE_INDEX, item_context,
                                          pair_features, negative_sample, sample_training_pairs, mature_training, sampling_weights)
from src.recommendation.models import model_specs, fit_item_model, score_catalog
from pipelines.recommendation import (choose_ease, common_specs, choose_default as choose_item_default,
                                           NAMES as ITEM_NAMES, old_predictions)
from archive.adapters.item_recommendation_data import (load_item_package, item_recommendation_list,
                                          export_item_recommendations, EXPORT_COLUMNS as ITEM_EXPORT_COLUMNS)


def example_transactions():
    return pd.DataFrame([dict(customer_id=f'u{i % 3}', invoice=f'inv{i}', stock_code=f'p{i:02}',
                              description=f'old name {i}', invoice_date=pd.Timestamp('2011-05-01'))
                         for i in range(12)])


class RecommendationTests(unittest.TestCase):
    def test_saved_profile_models_reproduce_profiles_predictions_and_all_eleven_lists(self):
        package = load_profile_package()
        artifact = joblib.load(ROOT / 'archive/artifacts/recommendation/profile_v2/model_bundle.joblib')
        tx, snapshots, _ = read_inputs()
        self.assertEqual(artifact['selected_method'], package['lock']['selected_method'])
        for date in sorted(package['profiles'].snapshot_date.unique()):
            phase = 'validation' if date < '2011-09-01' else 'test'
            ids = package['profiles'].loc[package['profiles'].snapshot_date.eq(date), 'customer_id'].sort_values().tolist()
            sample = [ids[0], ids[len(ids) // 2], ids[-1]]
            subset = snapshots.loc[snapshots.customer_id.isin(sample)]
            profile = purchase_profiles(tx, sample, pd.Timestamp(date))
            context, actual_profiles, ranks, weights = recommend_all(
                artifact[phase]['recommenders'],
                artifact[phase]['propensity_models'][artifact['selected_propensity']['family']],
                artifact['hybrid_specs'], tx, subset, pd.Timestamp(date), profile)
            codes = context['catalog'].stock_code.to_numpy()
            expected_profiles = package['profiles'].loc[package['profiles'].snapshot_date.eq(date)
                                                        & package['profiles'].customer_id.isin(sample)].sort_values('customer_id')
            np.testing.assert_allclose(actual_profiles[FEATURES].to_numpy(), expected_profiles[FEATURES].to_numpy(),
                                       rtol=1e-10, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(actual_profiles.predicted_repeat_share, expected_profiles.predicted_repeat_share)
            for method in HYBRID_NAMES:
                for index, customer in enumerate(context['customers']):
                    output = profile_recommendation_list(package, date, customer, method)
                    self.assertEqual(output.stock_code.tolist(), list(codes[ranks[method][index]]))
                    if method.startswith(('adaptive_', 'fixed_')):
                        self.assertAlmostEqual(output.repeat_weight.iloc[0], weights[method][index])
                    self.assertEqual(set(output), set(PROFILE_EXPORT_COLUMNS))
                    self.assertFalse(any('target' in name or 'hit' in name for name in output))

    def test_profile_export_strips_future_answers_and_rejects_duplicate_products(self):
        package = load_profile_package()
        row = package['profiles'].iloc[0]
        output = profile_recommendation_list(package, row.snapshot_date, row.customer_id, 'adaptive_ease')
        safe = export_profile_recommendations(output.assign(repeat_share_target=.5, target_items='SECRET'))
        self.assertEqual(set(safe), set(PROFILE_EXPORT_COLUMNS))
        bad = output.copy()
        bad.loc[1, 'stock_code'] = bad.loc[0, 'stock_code']
        with self.assertRaisesRegex(ValueError, 'ten distinct'):
            export_profile_recommendations(bad)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            profile_recommendation_list(package, '2099-01-01', row.customer_id, 'adaptive_ease')

    def test_purchase_habits_match_the_five_order_explanation_and_ignore_quantity(self):
        baskets = [('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'B', 'C'), ('A', 'E', 'F'), ('A', 'G', 'H')]
        tx = pd.DataFrame([dict(customer_id='a', invoice=str(i), invoice_date=pd.Timestamp('2011-05-01') + pd.Timedelta(days=i),
                                stock_code=code, quantity=1000) for i, basket in enumerate(baskets) for code in basket])
        # Duplicated lines and a separate invoice at the same timestamp do not create extra habit events.
        tx = pd.concat([tx, tx.iloc[[0]].assign(invoice='same_time')], ignore_index=True)
        profile = purchase_profiles(tx, ['a'], pd.Timestamp('2011-06-01')).iloc[0]
        self.assertEqual(profile.order_count, 5)
        self.assertEqual(profile.comparable_order_count, 4)
        self.assertEqual(profile.distinct_products, 8)
        self.assertAlmostEqual(profile.historical_repeat_share, 7 / 12)
        self.assertAlmostEqual(profile.top3_concentration, 2 / 3)
        self.assertAlmostEqual(profile.repeat_share_change, -.5)

    def test_profile_only_uses_visible_180_day_history_and_does_not_invent_new_preference(self):
        tx = example_transactions()
        cutoff = pd.Timestamp('2011-06-01')
        first = purchase_profiles(tx, ['u0', 'none'], cutoff)
        poisoned = pd.concat([tx, tx.assign(invoice_date=cutoff, stock_code='future'),
                              tx.assign(invoice_date=pd.Timestamp('2010-01-01'), stock_code='too_old')])
        second = purchase_profiles(poisoned, ['u0', 'none'], cutoff)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.historical_repeat_share.isna().all())
        self.assertTrue(first.repeat_share_change.isna().all())
        self.assertTrue(first.history_status.eq('Insufficient history').all())

    def test_repeat_share_target_is_undefined_for_nonbuyer_and_training_purges_unfinished_months(self):
        tx = example_transactions()
        cutoff = pd.Timestamp('2011-06-01')
        profiles = purchase_profiles(tx, ['u0', 'none'], cutoff)
        labels = pd.DataFrame({'snapshot_date': cutoff, 'customer_id': ['u0', 'none'],
                               'target_items': [['p00', 'UNKNOWN'], []], 'repeat_target_items': [['p00'], []]})
        frame = attach_repeat_target(profiles, labels)
        self.assertEqual(frame.repeat_share_target.iloc[0], .5)
        self.assertTrue(pd.isna(frame.repeat_share_target.iloc[1]))
        self.assertTrue(mature_examples(frame, cutoff + pd.Timedelta(days=29)).empty)
        self.assertEqual(mature_examples(frame, cutoff + pd.Timedelta(days=30)).customer_id.tolist(), ['u0'])

    def test_propensity_model_preprocessing_and_targets_are_strictly_forward(self):
        tx = example_transactions()
        profiles = purchase_profiles(tx, ['u0', 'u1', 'u2'], pd.Timestamp('2011-06-01'))
        profiles['label_end'] = pd.Timestamp('2011-07-01')
        profiles['repeat_share_target'] = [.2, .4, .6]
        future = profiles.assign(label_end=pd.Timestamp('2011-08-01'), repeat_share_target=1., order_count=99999)
        spec = dict(family='ridge', id='ridge', alpha=1.)
        first = fit_propensity(spec, profiles, pd.Timestamp('2011-07-01'))
        second = fit_propensity(spec, pd.concat([profiles, future]), pd.Timestamp('2011-07-01'))
        self.assertEqual(first['fit_rows'], 3)
        self.assertEqual(first['prior'], second['prior'])
        np.testing.assert_allclose(predict_propensity(first, profiles), predict_propensity(second, profiles))
        self.assertTrue(np.all(predict_propensity(first, profiles) == first['prior']))
        self.assertFalse(any('target' in feature or 'future' in feature for feature in FEATURES))

    def test_hybrid_fast_shortlist_exactly_matches_full_catalog_rank_fusion(self):
        rng = np.random.default_rng(42)
        personal = rng.integers(0, 4, size=(8, 23)).astype(float)
        personal[0] = 0
        personal[1] = 1
        algorithm = rng.normal(size=personal.shape)
        routes = prepare_routes(personal, algorithm)
        for value in (0., .2, .5, .8, 1.):
            expected = np.zeros_like(personal)
            for row in range(len(personal)):
                old = np.flatnonzero(personal[row] > 0)
                old = old[np.argsort(-personal[row, old], kind='stable')]
                new = np.flatnonzero(personal[row] == 0)
                new = new[np.argsort(-algorithm[row, new], kind='stable')]
                expected[row, old] = value / np.arange(1, len(old) + 1)
                expected[row, new] = (1 - value) / np.arange(1, len(new) + 1)
            actual = fuse_routes(routes, np.full(len(personal), value))
            np.testing.assert_array_equal(actual, topk(expected))
        np.testing.assert_array_equal(fuse_routes(routes, np.ones(len(personal))), topk(personal))

    def test_fast_hybrid_metrics_keep_unknown_targets_and_match_official_metrics(self):
        tx = example_transactions()
        cutoff = pd.Timestamp('2011-06-01')
        training = make_training(tx, cutoff)
        snapshots = pd.DataFrame({'snapshot_date': cutoff, 'customer_id': ['u0', 'u1']})
        context = make_context(tx, snapshots, cutoff, training['catalog'])
        labels = pd.DataFrame({'customer_id': ['u0', 'u1'], 'target_items': [['p00', 'p01', 'UNKNOWN'], []],
                               'repeat_target_items': [['p00'], []], 'new_target_items': [['p01', 'UNKNOWN'], []]})
        ranks = np.broadcast_to(np.arange(10), (2, 10))
        official, _ = evaluate(ranks, context, labels, 'hybrid', 'validation', 'example')
        fast = fast_metrics(ranks, metric_context(context, labels))
        for metric in fast:
            self.assertAlmostEqual(fast[metric], official[metric])

    def test_mixing_weights_depend_on_profile_and_are_not_item_quotas(self):
        spec = dict(intercept=.7, strength=1.)
        weights = mixing_weights(spec, np.array([.2, .8]), .5)
        np.testing.assert_allclose(weights, [.4, 1.])
        personal = np.zeros((2, 20))
        personal[:, :2] = [2, 1]
        routes = prepare_routes(personal, np.ones_like(personal))
        ranks = fuse_routes(routes, weights)
        self.assertTrue(np.all(ranks[1, :2] < 2))
        self.assertEqual(np.sum(ranks[0] < 2), 2)  # w=.4 cannot mean four old slots when only two exist.

    def test_profile_default_selection_ignores_review_results_and_requires_fixed_control(self):
        records = []
        for method in HYBRID_NAMES:
            score = .30 if method == 'personal_repeat' else .25
            if method.startswith('fixed_'):
                score = .32
            if method.startswith('adaptive_'):
                score = .321  # Better than pure repeat, insufficient gain over fixed.
            for date in ['2011-06-01', '2011-07-01', '2011-08-01']:
                records.append(dict(phase='validation', method=method, snapshot_date=date,
                                    ndcg_10=score, recall_10=.2, new_recall_10=.03 if 'adaptive' in method else .01))
        frame = pd.DataFrame(records)
        decision = choose_default(frame)
        self.assertFalse(decision['profile_value_demonstrated'])
        self.assertTrue(decision['selected_method'].startswith('fixed_'))
        poisoned = pd.concat([frame, frame.assign(phase='test', ndcg_10=999)])
        self.assertEqual(choose_default(poisoned), decision)
        frame.loc[frame.method.str.startswith('adaptive_'), 'ndcg_10'] = .33
        positive = choose_default(frame)
        self.assertTrue(positive['profile_value_demonstrated'])
        self.assertTrue(positive['selected_method'].startswith('adaptive_'))

    def test_comparison_algorithms_expose_saved_recommendations_without_answers(self):
        path = ROOT / 'archive/artifacts/recommendation/v1/test_customer_metrics.csv'
        expected = pd.read_csv(path, usecols=['snapshot_date', 'customer_id', 'method', 'recommended_items'],
                                dtype={'customer_id': 'string'})
        for method in ('itemknn', 'implicit_als', 'ease'):
            rows = load_comparison_list('2011-11-01', '12347', method)
            record = expected.loc[expected.snapshot_date.eq('2011-11-01') & expected.customer_id.eq('12347')
                                   & expected.method.eq(method)].iloc[0]
            self.assertEqual(rows.stock_code.tolist(), json.loads(record.recommended_items))
            self.assertNotIn('hit_10', rows)
            self.assertNotIn('target_items', rows)
            self.assertEqual(len(rows), 10)

    def test_saved_recommender_reproduces_lists_for_every_supported_month(self):
        rows, monthly, lock, comparison = load_recommendations()
        self.assertEqual(len(rows), 170210)
        self.assertEqual(len(monthly), 30)
        self.assertEqual(len(comparison), 10)
        self.assertFalse(lock['test_labels_used_for_selection'])
        self.assertFalse(lock['global_test_model_updates'])
        artifact = joblib.load(ROOT / 'archive/artifacts/recommendation/v1/selected_model.joblib')
        self.assertEqual(artifact['selected_method'], lock['selected_method'])
        tx, snapshots, _ = read_inputs()
        for phase in ('validation', 'test'):
            model = artifact[phase]
            dates = sorted(monthly.loc[monthly.phase.eq(phase), 'snapshot_date'].unique())
            for date in dates:
                cohort = snapshots.loc[snapshots.snapshot_date.eq(pd.Timestamp(date))]
                ids = cohort.customer_id.sort_values().to_numpy()
                sample = {ids[0], ids[len(ids) // 2], ids[-1]}
                context = make_context(tx, cohort.loc[cohort.customer_id.isin(sample)], pd.Timestamp(date), model['catalog'])
                ranks = recommend(model, context)
                codes = model['catalog'].stock_code.to_numpy()[ranks]
                for position, customer in enumerate(context['customers']):
                    recorded = rows.loc[rows.snapshot_date.eq(date) & rows.customer_id.eq(customer)].sort_values('rank')
                    self.assertEqual(recorded.stock_code.tolist(), list(codes[position]))
        self.assertNotIn('target_items', rows)
        self.assertNotIn('next_purchase_datetime', rows)

    def test_itemknn_is_cosine_with_shared_customer_shrinkage(self):
        x = sparse.csr_matrix([[1., 0, 1], [1, 1, 0], [0, 1, 1]])
        catalog = pd.DataFrame({'stock_code': ['a', 'b', 'c']})
        training = dict(x=x, catalog=catalog, cutoff=pd.Timestamp('2011-06-01'),
                        popularity={180: np.ones(3)}, _gram=None, _support=None)
        model = fit_model(dict(method='itemknn', neighbors=2, shrink=2), training)
        expected = np.full((3, 3), .5 / 3)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model['weights'].toarray(), expected, rtol=1e-6)

    def test_evaluation_generates_for_all_customers_but_reports_buyer_denominator(self):
        context = dict(cutoff=pd.Timestamp('2011-06-01'), customers=['a', 'b'],
                       catalog=pd.DataFrame({'stock_code': ['p', 'q']}), no_known_history=np.array([False, False]))
        labels = pd.DataFrame({'customer_id': ['a', 'b'], 'target_items': [['p'], []],
                               'repeat_target_items': [['p'], []], 'new_target_items': [[], []]})
        summary, rows = evaluate(np.array([[0, 1], [0, 1]]), context, labels, 'popular', 'validation', 'rule')
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary['buyers'], 1)
        self.assertEqual(summary['hit_10'], 1.)
        self.assertEqual(summary['all_customer_hit_10'], .5)

    def test_safe_export_rejects_future_columns_and_missing_products(self):
        rows = pd.DataFrame([dict(snapshot_date='2011-06-01', customer_id='a', rank=i + 1,
                                  stock_code=str(i), description=str(i), purchased_in_past_180d=False,
                                  recommendation_method='ease', reason='known purchases') for i in range(10)])
        validate_recommendations(rows, 'ease')
        self.assertNotIn('target_items', export_recommendations(rows.assign(target_items='future')).columns)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            validate_recommendations(rows.assign(target_items='future'), 'ease')
        with self.assertRaisesRegex(ValueError, 'ten distinct'):
            validate_recommendations(rows.iloc[:9], 'ease')

    def test_next_basket_is_first_timestamp_and_end_is_exclusive(self):
        tx = pd.DataFrame([
            ['a', 'old', '2011-05-31', 'A'],
            ['a', 'next', '2011-06-01', 'A'],
            ['a', 'next', '2011-06-01', 'B'],
            ['a', 'next', '2011-06-01', 'B'],
            ['a', 'same_time', '2011-06-01', 'C'],
            ['a', 'later', '2011-06-02', 'D'],
            ['b', 'excluded_end', '2011-07-01', 'E'],
        ], columns=['customer_id', 'invoice', 'invoice_date', 'stock_code'])
        tx.invoice_date = pd.to_datetime(tx.invoice_date)
        snapshots = pd.DataFrame({'snapshot_date': pd.Timestamp('2011-06-01'), 'customer_id': ['a', 'b'],
                                  'split': 'validation', 'repurchase_30d': [1, 0]})
        labels = next_basket_labels(tx, snapshots, pd.Timestamp('2011-06-01'))
        self.assertEqual(labels.target_items.tolist(), [['A', 'B', 'C'], []])
        self.assertEqual(labels.repeat_target_items.tolist(), [['A'], []])
        self.assertEqual(labels.new_target_items.tolist(), [['B', 'C'], []])
        self.assertEqual(labels.next_purchase_datetime.iloc[0], pd.Timestamp('2011-06-01'))

    def test_history_catalog_and_names_do_not_use_future_rows(self):
        tx = example_transactions()
        future = pd.DataFrame([dict(customer_id='u0', invoice='future', stock_code='p00',
                                    description='FUTURE NAME', invoice_date=pd.Timestamp('2011-06-01')),
                               dict(customer_id='u0', invoice='future', stock_code='NEW',
                                    description='NEW PRODUCT', invoice_date=pd.Timestamp('2011-06-02'))])
        cutoff = pd.Timestamp('2011-06-01')
        first = make_training(tx, cutoff)
        second = make_training(pd.concat([tx, future], ignore_index=True), cutoff)
        pd.testing.assert_frame_equal(first['catalog'], second['catalog'])
        np.testing.assert_array_equal(first['counts'].toarray(), second['counts'].toarray())
        self.assertNotIn('NEW', set(second['catalog'].stock_code))
        counts = history_counts(pd.concat([tx, future]), cutoff, first['catalog'], ['u0', 'unseen'])
        self.assertEqual(counts[0].sum(), 4)
        self.assertEqual(counts[1].sum(), 0)

    def test_implicit_als_step_matches_weighted_normal_equations(self):
        x = sparse.csr_matrix(np.log1p([[2, 0, 1], [0, 1, 0], [0, 0, 0]]))
        items = np.array([[.2, .8], [.7, .1], [.6, .4]])
        regularization, alpha = .3, 4.
        result = implicit_step(x, items, regularization, alpha)
        for row in range(x.shape[0]):
            observed = x.toarray()[row]
            c = np.diag(1 + alpha * observed)
            p = (observed > 0).astype(float)
            expected = np.linalg.solve(items.T @ c @ items + regularization * np.eye(2), items.T @ c @ p)
            np.testing.assert_allclose(result[row], expected, atol=1e-12)
        users = result
        before = implicit_loss(x, users, items, regularization, alpha)
        updated_items = implicit_step(x.T.tocsr(), users, regularization, alpha)
        after = implicit_loss(x, users, updated_items, regularization, alpha)
        self.assertLessEqual(after, before + 1e-10)

    def test_ease_matches_separate_ridge_regressions_with_zero_diagonal(self):
        x = np.array([[1., 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
        weights = ease_weights(x.T @ x, 2.)
        np.testing.assert_array_equal(np.diag(weights), 0.)
        for item in range(3):
            other = np.arange(3) != item
            expected = np.linalg.solve(x[:, other].T @ x[:, other] + 2 * np.eye(2), x[:, other].T @ x[:, item])
            np.testing.assert_allclose(weights[other, item], expected, rtol=1e-6, atol=1e-7)

    def test_known_purchases_remain_recommendable_and_empty_history_uses_popularity(self):
        tx = example_transactions()
        cutoff = pd.Timestamp('2011-06-01')
        training = make_training(tx, cutoff)
        snapshots = pd.DataFrame({'snapshot_date': cutoff, 'customer_id': ['u0', 'unseen']})
        context = make_context(tx, snapshots, cutoff, training['catalog'])
        model = fit_model(dict(method='personal_repeat', id='personal', half_life=None), training)
        ranks = recommend(model, context)
        self.assertEqual(list(training['catalog'].stock_code.iloc[ranks[0, :4]]), ['p00', 'p03', 'p06', 'p09'])
        np.testing.assert_array_equal(ranks[1], np.arange(10))
        exported = recommendation_table(model, context, ranks)
        self.assertNotIn('target_items', exported)
        self.assertNotIn('hit_10', exported)
        self.assertTrue(exported.loc[exported.customer_id.eq('unseen'), 'reason'].str.contains('popular', case=False).all())

    def test_full_target_denominator_includes_unknown_products(self):
        result = per_customer_metrics(['a', 'b'], ['a', 'never_seen'], ['a'], ['never_seen'], {'a', 'b'})
        self.assertEqual(result['target_count'], 2)
        self.assertEqual(result['known_target_count'], 1)
        self.assertEqual(result['recall_10'], .5)
        self.assertEqual(result['new_recall_10'], 0.)
        self.assertAlmostEqual(result['ndcg_10'], 1 / (1 + 1 / np.log2(3)))
        empty = per_customer_metrics(['a', 'b'], [], [], [], {'a', 'b'})
        self.assertEqual(empty['ndcg_10'], 0.)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            per_customer_metrics(['a', 'a'], ['a'], [], ['a'], {'a'})

    def test_three_algorithms_have_equal_search_budget_and_validation_ignores_test(self):
        specs = configurations()
        self.assertEqual(len(specs), 30)
        for method in ('itemknn', 'implicit_als', 'ease'):
            self.assertEqual(sum(spec['method'] == method for spec in specs), 8)
        summary = pd.DataFrame([dict(phase='validation', method=method, mean_ndcg_10=.1 + i / 100)
                                for i, method in enumerate(['popular', 'personal_repeat', 'itemknn', 'implicit_als', 'ease'])])
        monthly = pd.DataFrame([dict(phase='validation', method=row.method, snapshot_date=date,
                                     ndcg_10=row.mean_ndcg_10, recall_10=.1)
                                for row in summary.itertuples()
                                for date in ['2011-06-01', '2011-07-01', '2011-08-01']])
        poison = pd.DataFrame([dict(phase='test', method='popular', mean_ndcg_10=999)])
        decision = choose_method(pd.concat([summary, poison]), monthly)
        self.assertEqual(decision['selected_method'], 'ease')
        monthly.loc[monthly.method.eq('ease'), 'recall_10'] = 0
        self.assertEqual(choose_method(summary, monthly)['selected_method'], 'personal_repeat')
        candidates = [spec for spec in specs if spec['method'] == 'ease']
        tuning = pd.DataFrame([dict(phase='tuning', method='ease', config_id=spec['id'],
                                    mean_ndcg_10=.1, months=5) for spec in candidates])
        self.assertEqual(choose_config(tuning, 'ease', specs)['regularization'], 30000)

    def test_pairing_keeps_same_customer_months_and_is_reproducible(self):
        frame = pd.DataFrame([dict(snapshot_date=date, customer_id=customer, target_count=2,
                                   ndcg_10=.5, recall_10=.4)
                              for date in ['2011-06-01', '2011-07-01'] for customer in ['a', 'b', 'c']])
        a, interval = paired_intervals(frame, frame, resamples=10)
        b, _ = paired_intervals(frame, frame, resamples=10)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(interval['metrics']['ndcg_10']['lower_95'], 0.)
        with self.assertRaisesRegex(ValueError, 'targets differ'):
            paired_intervals(frame, frame.assign(target_count=999), resamples=10)

    def test_algorithms_serialize_and_use_only_prediction_context(self):
        tx = example_transactions()
        cutoff = pd.Timestamp('2011-06-01')
        training = make_training(tx, cutoff)
        snapshot = pd.DataFrame({'snapshot_date': cutoff, 'customer_id': ['u0', 'u1']})
        context = make_context(tx, snapshot, cutoff, training['catalog'])
        specs = [dict(id='knn', method='itemknn', neighbors=3, shrink=20),
                 dict(id='ease', method='ease', regularization=10),
                 dict(id='als', method='implicit_als', factors=3, regularization=.1, alpha=1., iterations=3)]
        with tempfile.TemporaryDirectory() as directory:
            for spec in specs:
                model = fit_model(spec, training)
                before = recommend(model, context)
                if spec['method'] != 'implicit_als':
                    weights = model['weights']
                    diagonal = weights.diagonal() if sparse.issparse(weights) else np.diag(weights)
                    np.testing.assert_array_equal(diagonal, 0.)
                path = Path(directory) / 'model.joblib'
                joblib.dump(model, path)
                after = recommend(joblib.load(path), dict(context, target_items=['future']))
                np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(topk(np.zeros((1, 12))), np.arange(10)[None, :])


class ItemScoringTests(unittest.TestCase):
    def test_full_item_audit_replayed_every_list_and_preserved_old_controls(self):
        from src.recommendation.data import sha256
        path = ROOT / 'artifacts/recommendation/audit'
        report = json.loads((path / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'completed_full_item_audit')
        self.assertEqual(report['verified_customer_lists'], 255315)
        self.assertEqual(report['verified_product_rows'], 2553150)
        self.assertEqual(report['verified_profile_rows'], 17021)
        self.assertEqual(report['old_control_months_unchanged'], 18)
        self.assertEqual(report['code_sha256'], sha256(ROOT / 'pipelines/audit_recommendation.py'))
        self.assertEqual(report['source_manifest_sha256'], sha256(ROOT / 'artifacts/recommendation/current/manifest.json'))
        self.assertEqual(report['best_new'], 'D_classifier')
        self.assertEqual(report['profile_reference'], 'C_classifier')

    def test_saved_item_models_reproduce_all_feature_groups_and_old_controls(self):
        package = load_item_package()
        artifact = joblib.load(ROOT / 'artifacts/recommendation/current/model_bundle.joblib')
        old = joblib.load(ROOT / 'archive/artifacts/recommendation/profile_v2/model_bundle.joblib')
        tx, snapshots, _ = read_inputs()
        self.assertEqual(package['lock']['selected_method'], artifact['selected_method'])
        for date in sorted(package['profiles'].snapshot_date.unique()):
            phase = 'validation' if date < '2011-09-01' else 'test'
            ids = package['profiles'].loc[package['profiles'].snapshot_date.eq(date), 'customer_id'].sort_values().tolist()
            subset = snapshots.loc[snapshots.customer_id.isin([ids[0], ids[len(ids) // 2], ids[-1]])]
            context = item_context(tx, subset, pd.Timestamp(date), artifact[phase]['ease'])
            ranks, scores = score_catalog(artifact[phase]['models'], context, batch_customers=3)
            old_ranks, old_scores = old_predictions(old, phase, context)
            ranks.update(old_ranks)
            scores.update(old_scores)
            catalog = context['catalog'].stock_code.to_numpy()
            for method in ITEM_NAMES:
                for row, customer in enumerate(context['customers']):
                    exported = item_recommendation_list(package, date, customer, method)
                    self.assertEqual(exported.stock_code.tolist(), list(catalog[ranks[method][row]]))
                    np.testing.assert_allclose(exported.ranking_score, scores[method][row], atol=2e-10, rtol=1e-10)
                    counts = context['counts_dense'][row, ranks[method][row]]
                    np.testing.assert_array_equal(exported.purchase_count_180d, counts)
                    self.assertEqual(set(exported), set(ITEM_EXPORT_COLUMNS))
                    self.assertFalse(any('target' in name or 'hit' in name for name in exported))

    def test_item_export_strips_future_answers_and_preserves_unbought_time_as_unknown(self):
        package = load_item_package()
        row = package['profiles'].iloc[0]
        exported = item_recommendation_list(package, row.snapshot_date, row.customer_id, 'D_ranker')
        safe = export_item_recommendations(exported.assign(target_items='SECRET', hit_10=1))
        self.assertEqual(set(safe), set(ITEM_EXPORT_COLUMNS))
        self.assertTrue(safe.loc[~safe.purchased_in_past_180d, 'days_since_item_purchase'].isna().all())
        bad = exported.copy()
        bad.loc[1, 'stock_code'] = bad.loc[0, 'stock_code']
        with self.assertRaisesRegex(ValueError, 'ten safe distinct'):
            export_item_recommendations(bad)

    def test_saved_item_audit_has_purged_labels_forward_ease_and_matched_parameters(self):
        path = ROOT / 'artifacts/recommendation/current'
        audit = json.loads((path / 'training_audit.json').read_text(encoding='utf-8'))
        for row in audit['item_models']:
            self.assertLessEqual(pd.Timestamp(row['latest_label_end']), pd.Timestamp(row['fit_cutoff']))
        for row in audit['ease_features']:
            self.assertLess(pd.Timestamp(row['latest_transaction']), pd.Timestamp(row['fit_cutoff']))
            for date in row['selection_label_dates']:
                self.assertLessEqual(pd.Timestamp(date) + pd.Timedelta(days=30), pd.Timestamp(row['feature_date']))
        model = joblib.load(path / 'model_bundle.joblib')
        for phase in ('validation', 'test'):
            for family in ('lr', 'classifier', 'ranker'):
                specs = [model[phase]['models'][f'{group}_{family}']['spec'] for group in FEATURE_SETS]
                self.assertTrue(all(spec == specs[0] for spec in specs))
        samples = joblib.load(path / 'training_samples.joblib')['parts']
        for part in samples:
            self.assertEqual(part['groups'].sum(), len(part['y']))
            self.assertEqual(len(part['groups']), len(part['queries']))
            start = 0
            for size in part['groups']:
                target = part['y'][start:start + size]
                self.assertGreater(target.sum(), 0)
                self.assertLess(target.sum(), size)
                self.assertEqual(len(set(part['product_columns'][start:start + size])), size)
                start += size

    def toy_context(self):
        tx = example_transactions()
        extra = tx.loc[tx.stock_code.isin(['p00', 'p01', 'p02'])].assign(
            invoice='second', invoice_date=pd.Timestamp('2011-05-21'))
        tx = pd.concat([tx, extra], ignore_index=True)
        cutoff = pd.Timestamp('2011-06-01')
        training = make_training(tx, cutoff)
        ease = fit_model(dict(id='ease_10', method='ease', regularization=10), training)
        snapshots = pd.DataFrame({'snapshot_date': cutoff, 'customer_id': ['u0', 'u1', 'u2']})
        context = item_context(tx, snapshots, cutoff, ease)
        return tx, snapshots, ease, context

    def test_pair_features_are_pre_cutoff_and_have_correct_counts_and_item_timing(self):
        tx, snapshots, ease, first = self.toy_context()
        future = tx.assign(invoice_date=pd.Timestamp('2011-06-01'), invoice='future')
        too_old = tx.assign(invoice_date=pd.Timestamp('2009-01-01'), invoice='too_old')
        second = item_context(pd.concat([tx, future, too_old]), snapshots, pd.Timestamp('2011-06-01'), ease)
        code = list(ease['catalog'].stock_code).index('p00')
        a = pair_features(first, np.array([0]), np.array([code]))
        b = pair_features(second, np.array([0]), np.array([code]))
        np.testing.assert_allclose(a, b, equal_nan=True)
        row = dict(zip(ALL_FEATURES, a[0]))
        self.assertEqual(row['bought_before'], 1)
        self.assertAlmostEqual(row['log_count_180d'], np.log(3), places=6)
        self.assertAlmostEqual(row['item_recency_180d'], 11 / 180, places=6)
        self.assertAlmostEqual(row['mean_item_gap_180d'], 20 / 180, places=6)
        self.assertEqual(row['has_item_gap'], 1)
        self.assertEqual(a.shape[1], len(ALL_FEATURES))

    def test_four_feature_groups_isolate_profiles_and_ease_and_have_no_identifiers(self):
        self.assertTrue(set(FEATURE_SETS['A']) < set(FEATURE_SETS['B']))
        self.assertTrue(set(FEATURE_SETS['A']) < set(FEATURE_SETS['C']))
        self.assertEqual(set(FEATURE_SETS['D']), set(ALL_FEATURES))
        self.assertFalse(any('profile_' in x or 'ease' in x for x in FEATURE_SETS['A']))
        self.assertFalse(any('ease' in x for x in FEATURE_SETS['B']))
        self.assertFalse(any('profile_' in x or x.startswith('repeat_x') for x in FEATURE_SETS['C']))
        self.assertFalse(any('customer_id' in x or 'stock_code' in x or 'target' in x for x in ALL_FEATURES))
        for family in ('lr', 'classifier', 'ranker'):
            self.assertEqual(len(model_specs(family)), 4)

    def test_negative_samples_are_shared_reproducible_and_never_contain_positives(self):
        positive = np.array([0, 3])
        ease = np.array([0., 5, 4, 0, 3, 2, 1, 0, -1, -2, -3, -4])
        personal = np.zeros(12)
        personal[4] = 10
        a = negative_sample(positive, ease, personal, 42, 7)
        b = negative_sample(positive, ease, personal, 42, 7)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 7)
        self.assertFalse(set(a) & set(positive))
        self.assertIn(1, a)
        self.assertIn(4, a)
        self.assertEqual(len(negative_sample(np.arange(12), ease, personal, 42, 10)), 0)
        self.assertEqual(len(negative_sample(positive, ease, personal, 42, 1)), 1)

    def test_sampling_weights_represent_the_full_negative_pool_for_classification(self):
        queries = pd.DataFrame([dict(positives=2, negatives=6), dict(positives=1, negatives=9)])
        weights = sampling_weights(queries, catalog_size=20)
        self.assertEqual(len(weights), 18)
        # All positives + three certain hard negatives; three uniform draws represent 15 remaining negatives.
        np.testing.assert_allclose(weights[:8], [1, 1, 1, 1, 1, 5, 5, 5])
        self.assertAlmostEqual(weights[:8].sum(), 20)
        self.assertAlmostEqual(weights[8:].sum(), 20)

    def test_item_training_uses_buyer_groups_mature_targets_and_keeps_unknowns_out_of_candidates(self):
        _, _, _, context = self.toy_context()
        labels = pd.DataFrame({'customer_id': ['u0', 'u1', 'u2'],
                               'target_items': [['p00', 'UNKNOWN'], [], ['ONLY_UNKNOWN']]})
        part = sample_training_pairs(context, labels)
        self.assertEqual(len(part['groups']), 1)
        self.assertEqual(part['y'].sum(), 1)
        self.assertEqual(part['unknown_only_buyers'], 1)
        self.assertNotIn('UNKNOWN', part['catalog_codes'])
        with self.assertRaisesRegex(ValueError, 'No mature'):
            mature_training([part], pd.Timestamp('2011-06-30'))
        fit = mature_training([part], pd.Timestamp('2011-07-01'))
        self.assertEqual(len(fit['y']), fit['groups'].sum())

    def test_historical_ease_parameter_choice_ignores_not_yet_mature_labels(self):
        specs = [x for x in configurations() if x['method'] == 'ease']
        self.assertEqual(choose_ease([], pd.Timestamp('2011-01-01'))['regularization'], 30000)
        history = [dict(config_id=s['id'], label_end='2010-12-31', ndcg_10=1 / s['regularization']) for s in specs]
        future = [dict(config_id=s['id'], label_end='2011-02-01', ndcg_10=s['regularization']) for s in specs]
        self.assertEqual(choose_ease(history, pd.Timestamp('2011-01-01')), choose_ease(history + future, pd.Timestamp('2011-01-01')))
        self.assertEqual(choose_ease(history, pd.Timestamp('2011-01-01'))['regularization'], 10)

    def test_common_hyperparameters_are_selected_across_all_four_groups_and_five_months(self):
        rows = []
        for family in ('lr', 'classifier', 'ranker'):
            for index, spec in enumerate(model_specs(family)):
                for group in FEATURE_SETS:
                    for date in pd.date_range('2011-01-01', '2011-05-01', freq='MS'):
                        rows.append(dict(phase='development', group=group, config_id=spec['id'],
                                         snapshot_date=str(date.date()), ndcg_10=.1 + index * .01))
        frame = pd.DataFrame(rows)
        selected = common_specs(frame)
        for family in selected:
            self.assertEqual(selected[family], model_specs(family)[-1])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            common_specs(frame.iloc[1:])

    def test_item_default_gate_and_profile_attribution_ignore_review_results(self):
        rows = []
        for method in ITEM_NAMES:
            score = .36 if method == 'D_classifier' else .34 if method == 'C_classifier' else .30
            for date in ['2011-06-01', '2011-07-01', '2011-08-01']:
                rows.append(dict(phase='validation', method=method, snapshot_date=date, ndcg_10=score,
                                 recall_10=.23 if method == 'D_classifier' else .20,
                                 new_recall_10=.04 if method == 'D_classifier' else .01))
        frame = pd.DataFrame(rows)
        decision = choose_item_default(frame)
        self.assertEqual(decision['selected_method'], 'D_classifier')
        self.assertTrue(decision['selected_profile_value_demonstrated'])
        self.assertEqual(decision, choose_item_default(pd.concat([frame, frame.assign(phase='test', ndcg_10=999)])))
        frame.loc[frame.method.eq('D_classifier'), 'recall_10'] = .20
        self.assertEqual(choose_item_default(frame)['selected_method'], 'personal_repeat')

    def test_item_models_serialize_and_batch_scoring_matches_full_catalog_for_all_feature_groups(self):
        _, _, _, context = self.toy_context()
        labels = pd.DataFrame({'customer_id': ['u0', 'u1', 'u2'], 'target_items': [['p00'], ['p01'], ['p02']]})
        part = sample_training_pairs(context, labels)
        fit = mature_training([part], pd.Timestamp('2011-07-01'))
        models = {f'{group}_{family}': fit_item_model(model_specs(family)[0], group, fit)
                  for group in FEATURE_SETS for family in ('lr', 'classifier', 'ranker')}
        first, values = score_catalog(models, context, batch_customers=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'item_models.joblib'
            joblib.dump(models, path)
            second, scores_after = score_catalog(joblib.load(path), dict(context, target_items='POISON'), batch_customers=20)
        for method in models:
            np.testing.assert_array_equal(first[method], second[method])
            np.testing.assert_allclose(values[method], scores_after[method])
            self.assertEqual(first[method].shape, (3, 10))


if __name__ == '__main__':
    unittest.main()
