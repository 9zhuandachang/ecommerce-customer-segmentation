"""Long-term checks for spend labels, cohort joins, and model acceptance."""

import unittest

import joblib
import numpy as np
import pandas as pd

from archive.spending.experiments import (
    attach_labels, candidate_passes, choose_model, compare_months, evaluate,
    top20_mask,
)
from src.spending.features import build_temporal_features
from archive.spending.optimization import fit_rows_for_date
from archive.spending.conditional_amount import conditional_metrics
from archive.spending.calibration import (
    lagged_amount_factor, monotone_buyer_amount,
    segmented_buyer_amount_factors,
)
from archive.spending.two_year_validation import build_schedule as build_two_year_schedule
from archive.spending.seasonal_features import build_seasonal_features
from archive.spending.jewelry_validation import (
    adapt_transactions as adapt_jewelry_transactions,
    build_schedule as build_jewelry_schedule,
)
from archive.spending.hurdle_diagnostics import (
    decompose_spend, high_amount_mask, recent_amount_weights,
)
from archive.spending.uncertainty import bootstrap as bootstrap_customer_intervals
from archive.spending.intervals import (
    amount_interval, calibrated_interval, finite_quantile,
    recent_mature_history,
)
from archive.spending.amazon_validation import (
    adapt_transactions as adapt_amazon_transactions,
    build_schedule as build_amazon_schedule,
)
from archive.spending.ranking_audit import pair_stability, recency_group
from archive.spending.high_value import (
    EXPORT_COLUMNS as HIGH_VALUE_EXPORT_COLUMNS,
    demonstration_decision, monthly_metric, rank_month,
    validate_training_maturity,
)
from archive.spending.buyer_weight import buyer_weights, advancement_checks
from archive.spending.composition_features import build_composition_features
from archive.spending.tail_signal_audit import (
    mark_actual_top_buyers, predicted_tail_metrics,
)
from archive.spending.seasonality_experiment import (
    add_calendar_features, fit_predict_calendar,
)
from archive.spending.cdnow_validation import (
    adapt_transactions as adapt_cdnow_transactions,
    build_schedule as build_cdnow_schedule,
)
from archive.spending.count_amount import (
    build_future_invoice_counts, fit_predict_count_amount,
    advancement_checks as count_amount_advancement_checks,
)
from archive.spending.calibrated_hurdle import (
    combine_probability_amount,
    advancement_checks as calibrated_hurdle_advancement_checks,
)
from archive.spending.gated_median import gated_median
from archive.spending.order_dynamics_features import build_features as build_order_dynamics
from archive.spending.order_dynamics_uncertainty import paired_metrics as order_dynamics_paired_metrics
from archive.spending.acceptance_gate import assess_monthly as assess_spend_acceptance
from archive.spending.olist_feasibility import (
    adapt_sources as adapt_olist_sources,
    build_schedule as build_olist_schedule,
)
from src.spending.comparison import (
    FEATURES as RANKING_FEATURES, FAMILIES as RANKING_FAMILIES,
    configurations as ranking_configurations, training_rows as ranking_training_rows,
    predict as predict_ranking, fit_model as fit_ranking_model,
    make_priority as make_ranking_priority, paired_bootstrap as ranking_bootstrap,
    decide as decide_ranking, validation_leaders, weighted_capture,
    choose_tuning_config, load_data as load_ranking_data,
    OUTPUT as RANKING_OUTPUT, rule_score as ranking_rule_score,
)
from archive.adapters.high_value_data import load_high_value_comparison, validate_priority


class SpendPredictionTests(unittest.TestCase):
    def test_ranking_saved_model_reproduces_all_selected_scores_and_exports(self):
        customers, monthly, lock, comparison = load_high_value_comparison()
        self.assertEqual((len(customers), len(monthly), len(comparison)), (17021, 6, 12))
        saved = joblib.load(RANKING_OUTPUT / 'ranking_bundle.joblib')
        self.assertEqual(saved['selected_method'], lock['selected_method'])
        self.assertTrue(all(record['test_rows'] == 0 for record in lock['training_audit']))
        for phase, partitions in [('validation', ('validation',)), ('review', ('test',))]:
            data = load_ranking_data(partitions)
            path = RANKING_OUTPUT / f'{phase}_predictions.csv'
            predictions = pd.read_csv(path, dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
            for method in RANKING_FAMILIES:
                score = predict_ranking(saved[phase][method], data.drop(columns='spend_30d'))
                recorded = predictions.loc[predictions.method.eq(method)]
                self.assertEqual(recorded.customer_id.tolist(), data.customer_id.tolist())
                np.testing.assert_allclose(score, recorded.score, rtol=1e-12, atol=1e-10)
            selected = saved['selected_method']
            score = (predict_ranking(saved[phase][selected], data) if selected in RANKING_FAMILIES
                     else ranking_rule_score(selected, data))
            exported = make_ranking_priority(data, score, ranking_rule_score(lock['best_rule'], data), selected)
            exported.snapshot_date = exported.snapshot_date.dt.strftime('%Y-%m-%d')
            dates = exported.snapshot_date.unique()
            recorded = customers.loc[customers.snapshot_date.isin(dates)].reset_index(drop=True)
            pd.testing.assert_frame_equal(exported, recorded, check_dtype=False, atol=1e-9)
        with self.assertRaisesRegex(ValueError, 'export columns'):
            validate_priority(customers.assign(spend_30d=0), monthly, lock)
        broken = customers.copy()
        broken.loc[0, 'priority_percentile'] = 0
        with self.assertRaisesRegex(ValueError, 'percentiles'):
            validate_priority(broken, monthly, lock)

    def test_ranking_training_keeps_fixed_partitions_and_mature_windows(self):
        data = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-01-01', '2011-02-01', '2011-05-01',
                                             '2011-06-01', '2011-08-01', '2011-09-01']),
            'split': ['train', 'train', 'train', 'validation', 'validation', 'test'],
        })
        tune = ranking_training_rows(data, 'tuning', pd.Timestamp('2011-03-01'))
        self.assertEqual(tune.snapshot_date.tolist(), [pd.Timestamp('2011-01-01')])
        validation = ranking_training_rows(data, 'validation', pd.Timestamp('2011-08-01'))
        self.assertEqual(validation.split.unique().tolist(), ['train'])
        review = ranking_training_rows(data, 'review', pd.Timestamp('2011-10-01'))
        self.assertFalse(review.split.eq('test').any())
        self.assertTrue(review.split.eq('validation').any())

    def test_ranking_priority_accepts_features_only_and_is_label_independent(self):
        data = pd.DataFrame({'snapshot_date': pd.Timestamp('2011-06-01'),
                             'customer_id': ['b', 'a', 'c', 'd', 'e', 'f'],
                             'recency_days': [1] * 6, 'monetary_value_180d': [10.] * 6})
        score = np.array([10., 10., 8., 6., 4., 2.])
        base = score[::-1].copy()
        priority = make_ranking_priority(data, score, base, 'lgbm_tweedie')
        self.assertEqual(priority.customer_id.iloc[:2].tolist(), ['a', 'b'])
        self.assertEqual(priority.priority_band.eq('Top 20%').sum(), 2)
        self.assertEqual(priority.priority_rank.tolist(), list(range(1, 7)))
        poisoned = data.assign(spend_30d=[0, 0, 0, 0, 0, 1e12])
        pd.testing.assert_frame_equal(priority, make_ranking_priority(poisoned, score, base, 'lgbm_tweedie'))
        self.assertNotIn('score', priority.columns)
        self.assertNotIn('spend_30d', priority.columns)

    def test_ranking_customer_bootstrap_is_paired_and_keeps_partial_quota(self):
        y, weights = np.array([100., 10., 0.]), np.array([2, 1, 2])
        self.assertAlmostEqual(weighted_capture(y, weights, np.array([0, 1, 2])), 100 / 210)
        data = pd.DataFrame({'snapshot_date': np.repeat(pd.to_datetime(['2011-06-01', '2011-07-01']), 6),
                             'customer_id': ['a', 'b', 'c', 'd', 'e', 'f'] * 2,
                             'spend_30d': [100., 50., 20., 10., 5., 1.] * 2})
        score = np.array([6., 5., 4., 3., 2., 1.] * 2)
        a, interval = ranking_bootstrap(data, score, score, resamples=10)
        b, _ = ranking_bootstrap(data, score, score, resamples=10)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(interval['lower_95_pp'], 0.)
        self.assertEqual(interval['upper_95_pp'], 0.)

    def test_ranking_selection_ignores_review_and_can_choose_strong_rule(self):
        methods = ['spend_180d', 'spend_90d', 'spend_30d'] + list(RANKING_FAMILIES)
        summary = pd.DataFrame([{'phase': 'validation', 'method': m, 'mean_capture_20': .7 + i / 100}
                                for i, m in enumerate(methods)])
        monthly = pd.DataFrame([{'phase': 'validation', 'method': m, 'snapshot_date': date,
                                 'capture_20': .7 + i / 100}
                                for i, m in enumerate(methods)
                                for date in ['2011-06-01', '2011-07-01', '2011-08-01']])
        poison = pd.DataFrame([{'phase': 'review', 'method': 'spend_180d', 'mean_capture_20': 100}])
        base, model = validation_leaders(pd.concat([summary, poison]), monthly)
        self.assertEqual((base, model), ('spend_30d', 'lgbm_two_stage'))
        decision = decide_ranking(monthly, base, model, {'lower_95_pp': -.01})
        self.assertEqual(decision['selected_method'], base)
        decision = decide_ranking(monthly, base, model, {'lower_95_pp': .1})
        self.assertEqual(decision['selected_method'], model)

    def test_ranking_tuning_has_equal_budget_and_ignores_review_metrics(self):
        specs = ranking_configurations()
        self.assertEqual(len(specs), 24)
        for family in RANKING_FAMILIES:
            candidates = [s for s in specs if s['family'] == family]
            self.assertEqual(len(candidates), 8)
            rows = pd.DataFrame([{'phase': 'tuning', 'method': family, 'config_id': s['id'],
                                  'months': 5, 'mean_capture_20': .5} for s in candidates])
            rows = pd.concat([rows, pd.DataFrame([dict(rows.iloc[-1], phase='review', mean_capture_20=100)])])
            selected = choose_tuning_config(rows, specs, family)
            self.assertEqual((selected['n_estimators'], selected['num_leaves'], selected['profile']),
                             (120, 15, 'conservative'))

    def test_ranking_models_fit_and_predict_without_future_columns(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.uniform(0, 10, (240, len(RANKING_FEATURES))), columns=RANKING_FEATURES)
        data['spend_30d'] = np.where(np.arange(240) % 2, 10 + data.recency_days * 2, 0)
        for family in RANKING_FAMILIES:
            spec = dict(family=family, id='test', n_estimators=12, num_leaves=7, profile='standard')
            model = fit_ranking_model(spec, data)
            scores = predict_ranking(model, data.drop(columns='spend_30d'))
            self.assertEqual(len(scores), len(data))
            self.assertTrue(np.isfinite(scores).all())
            np.testing.assert_allclose(scores, predict_ranking(model, data.assign(spend_30d=1e12)))

    def test_high_value_ranking_uses_fixed_quota_and_excludes_future_from_export(self):
        cohort = pd.DataFrame({
            'snapshot_date': pd.Timestamp('2011-06-01'),
            'customer_id': pd.Series(['b', 'a', 'c', 'd', 'e', 'f'], dtype='string'),
            'spend_30d': [100., 60., 30., 5., 5., 0.],
            'extended_tweedie': [10., 10., 8., 4., 3., 2.],
            'historic_180d': [4., 3., 9., 8., 2., 1.],
            'recency_days': [1, 2, 3, 4, 5, 6],
            'monetary_value_180d': [24., 18., 54., 48., 12., 6.],
        })
        ranked = rank_month(cohort)
        self.assertEqual(ranked.customer_id.iloc[:2].tolist(), ['a', 'b'])
        self.assertEqual(ranked.priority_band.eq('Top 20%').sum(), 2)
        self.assertEqual(ranked.loc[ranked.customer_id.eq('c'),
                                    'baseline_rank'].iloc[0], 1)
        metric = monthly_metric(ranked)
        self.assertEqual(metric['top20_count'], 2)
        self.assertAlmostEqual(metric['model_capture'], 160 / 200)
        self.assertAlmostEqual(metric['baseline_capture'], 35 / 200)
        self.assertNotIn('spend_30d', HIGH_VALUE_EXPORT_COLUMNS)
        self.assertNotIn('extended_tweedie', HIGH_VALUE_EXPORT_COLUMNS)
        self.assertNotIn('historic_180d', HIGH_VALUE_EXPORT_COLUMNS)
        with self.assertRaisesRegex(ValueError, 'unique'):
            rank_month(pd.concat([cohort, cohort.iloc[[0]]], ignore_index=True))

    def test_high_value_demo_gate_rejects_monthly_regression(self):
        dates = pd.date_range('2011-06-01', periods=6, freq='MS')
        monthly = pd.DataFrame({
            'snapshot_date': dates.strftime('%Y-%m-%d'),
            'gain_pp': [2., 2., 2., 2., 2., 2.],
            'top20_count': [20] * 6,
            'customers': [100] * 6,
        })
        periods = pd.DataFrame({'gain_pp': [2., 2.]})
        self.assertTrue(demonstration_decision(monthly, periods)[
            'historical_demo_allowed'])
        monthly.loc[0, 'gain_pp'] = -0.1
        self.assertFalse(demonstration_decision(monthly, periods)[
            'historical_demo_allowed'])

    def test_high_value_rejects_immature_saved_training_fold(self):
        folds = []
        for date in pd.date_range('2011-06-01', periods=3, freq='MS'):
            folds.append({'target_date': str(date.date()), 'fit_count': 10,
                          'latest_fit_cutoff': str((date - pd.DateOffset(months=1)).date()),
                          'latest_fit_label_end': str(date.date())})
        retro = {'fit_cutoffs': [
            {'snapshot_date': str(date.date()), 'fit_count': 10,
             'latest_fit_label_end': str(date.date())}
            for date in pd.date_range('2011-09-01', periods=3, freq='MS')]}
        validate_training_maturity({'folds': folds}, retro)
        retro['fit_cutoffs'][0]['latest_fit_label_end'] = '2011-09-02'
        with self.assertRaisesRegex(ValueError, 'immature'):
            validate_training_maturity({'folds': folds}, retro)

    def test_acceptance_gate_requires_every_metric_and_matching_cohorts(self):
        rows = []
        for month in ['2022-01-01', '2022-02-01', '2022-03-01']:
            for method, mae, buyer_mae, predicted, top_actual, top_bias in [
                    ('historic_180d', 100., 100., 1400., 500., .2),
                    ('extended_tweedie', 80., 90., 1000., 550., .05)]:
                rows.append({
                    'snapshot_date': month, 'method': method,
                    'snapshot_count': 100, 'buyer_count': 40,
                    'actual_spend': 1000., 'predicted_spend': predicted,
                    'aggregate_bias': predicted / 1000. - 1,
                    'mae': mae, 'positive_mae': buyer_mae,
                    'top20_count': 20, 'top20_actual_spend': top_actual,
                    'top20_spend_capture': top_actual / 1000.,
                    'top20_group_bias': top_bias,
                })
        monthly = pd.DataFrame(rows)
        result = assess_spend_acceptance(monthly)
        self.assertTrue(result['technical_passed'])
        self.assertEqual(result['joint_stable_months'], 3)
        missing = monthly.drop(columns='top20_group_bias')
        self.assertIsNone(assess_spend_acceptance(missing)['criteria'][
            'top20_group_bias_within_10pct'])
        self.assertFalse(assess_spend_acceptance(missing)['technical_passed'])
        inconsistent = monthly.copy()
        inconsistent.loc[inconsistent.method.eq('extended_tweedie').idxmax(),
                         'buyer_count'] = 41
        with self.assertRaisesRegex(ValueError, 'cohorts differ'):
            assess_spend_acceptance(inconsistent)

    def test_order_dynamics_paired_metrics_keep_error_and_bias_separate(self):
        buyers = pd.DataFrame({
            'spend_30d': [100., 200.],
            'old': [90., 180.],
            'new': [95., 190.],
        })
        metric = order_dynamics_paired_metrics(buyers)
        self.assertAlmostEqual(metric['buyer_mae_reduction'], .5)
        self.assertAlmostEqual(metric['absolute_bias_difference_pp'], -5.)

    def test_order_dynamics_merge_invoice_lines_and_exclude_future(self):
        cutoff = pd.Timestamp('2021-07-01')
        snapshots = pd.DataFrame({
            'snapshot_date': [cutoff, cutoff],
            'customer_id': ['a', 'b'],
            'monetary_value_180d': [1000., 100.],
            'invoice_frequency_180d': [4, 1],
        })
        clean = pd.DataFrame({
            'customer_id': ['a'] * 7 + ['b'],
            'invoice': ['old', 'i1', 'i2', 'i3', 'i4', 'i4', 'future', 'j1'],
            'invoice_date': pd.to_datetime([
                '2021-01-01', '2021-01-02', '2021-06-01', '2021-06-15',
                '2021-06-20', '2021-06-20', '2021-07-01', '2021-06-10']),
            'line_amount': [900., 100., 200., 300., 150., 250., 500., 100.],
        })
        schedule = pd.DataFrame({
            'snapshot_date': [cutoff],
            'feature_start': [cutoff - pd.Timedelta(days=180)],
        })
        features, audit = build_order_dynamics(snapshots, clean, schedule)
        a = features.set_index('customer_id').loc['a']
        b = features.set_index('customer_id').loc['b']
        self.assertAlmostEqual(a.last_three_invoice_mean_180d, 300.)
        self.assertAlmostEqual(a.recent_three_vs_all_order_mean_180d, 1.2)
        self.assertGreater(a.invoice_value_cv_180d, 0)
        self.assertAlmostEqual(b.last_three_invoice_mean_180d, 100.)
        self.assertEqual(b.invoice_value_cv_180d, 0)
        self.assertAlmostEqual(b.recent_three_vs_all_order_mean_180d, 1.)
        self.assertLess(pd.Timestamp(audit.latest_input_datetime.iloc[0]), cutoff)
        clean.loc[clean.invoice.eq('future'), 'line_amount'] = 999.
        changed, _ = build_order_dynamics(snapshots, clean, schedule)
        pd.testing.assert_frame_equal(features, changed)

    def test_gated_median_uses_fixed_half_probability_boundary(self):
        values = gated_median(np.array([0, .5, .5001, 1]),
                              np.array([100, 200, 300, 400]))
        np.testing.assert_array_equal(values, np.array([0, 0, 300, 400]))
        for probability, amount in [(np.array([np.nan]), np.array([100])),
                                    (np.array([1.01]), np.array([100])),
                                    (np.array([.6]), np.array([-1]))]:
            with self.assertRaisesRegex(ValueError, 'Invalid gated median'):
                gated_median(probability, amount)

    def test_olist_adapter_uses_stable_customer_and_ignores_final_status(self):
        orders = pd.DataFrame({
            'order_id': ['o1', 'o2'], 'customer_id': ['c1', 'c2'],
            'order_status': ['delivered', 'canceled'],
            'order_purchase_timestamp': ['2018-01-01 12:00:00',
                                         '2018-02-01 12:00:00'],
        })
        customers = pd.DataFrame({
            'customer_id': ['c1', 'c2'], 'customer_unique_id': ['u1', 'u1'],
        })
        items = pd.DataFrame({
            'order_id': ['o1', 'o1', 'o2'], 'order_item_id': [1, 2, 1],
            'product_id': ['p1', 'p2', 'p1'], 'price': [10., 20., 30.],
        })
        clean, audit = adapt_olist_sources(orders, customers, items)
        self.assertEqual(clean.customer_id.tolist(), ['u1', 'u1', 'u1'])
        self.assertEqual(clean.invoice.nunique(), 2)
        self.assertEqual(audit['status_order_counts']['canceled'], 1)
        self.assertEqual(audit['adapted_amount_cents'], 6000)
        items.loc[1, 'order_item_id'] = 1
        with self.assertRaisesRegex(ValueError, 'primary key'):
            adapt_olist_sources(orders, customers, items)

    def test_olist_schedule_keeps_precommitted_months_and_full_windows(self):
        clean = pd.DataFrame({'invoice_date': pd.to_datetime(
            ['2016-09-04', '2018-09-03'])})
        schedule = build_olist_schedule(clean)
        self.assertEqual(len(schedule), 17)
        self.assertEqual(schedule.snapshot_date.iloc[0], pd.Timestamp('2017-04-01'))
        self.assertEqual(schedule.snapshot_date.iloc[-1], pd.Timestamp('2018-08-01'))
        self.assertEqual(schedule.loc[schedule.split.eq('locked_test'),
                                      'snapshot_date'].dt.strftime('%Y-%m').tolist(),
                         ['2018-06', '2018-07', '2018-08'])
        self.assertLessEqual(schedule.label_end_exclusive.max(),
                             pd.Timestamp('2018-09-03'))

    def test_probability_amount_product_rejects_invalid_forecasts(self):
        np.testing.assert_array_equal(
            combine_probability_amount(np.array([0, .4, 1]),
                                       np.array([100, 200, 300])),
            np.array([0, 80, 300]))
        with self.assertRaisesRegex(ValueError, 'Invalid probability'):
            combine_probability_amount(np.array([1.2]), np.array([100]))
        with self.assertRaisesRegex(ValueError, 'Invalid probability'):
            combine_probability_amount(np.array([.5]), np.array([-1]))

    def test_calibrated_hurdle_advancement_keeps_buyer_guard(self):
        summary = pd.DataFrame([
            {'period': 'rolling_jan_may', 'method': 'extended_tweedie',
             'positive_mae': 100, 'mae': 50, 'top20_spend_capture': .70,
             'aggregate_bias': .10},
            {'period': 'rolling_jan_may', 'method': 'stage6_probability_times_gamma',
             'positive_mae': 105, 'mae': 45, 'top20_spend_capture': .72,
             'aggregate_bias': .02},
        ])
        checks = calibrated_hurdle_advancement_checks(summary)
        self.assertFalse(checks['buyer_mae_reduction_at_least_5pct'])
        self.assertTrue(all(value for name, value in checks.items()
                            if name != 'buyer_mae_reduction_at_least_5pct'))

    def test_future_invoice_count_deduplicates_lines_and_respects_exclusive_end(self):
        cutoff = pd.Timestamp('2021-01-01')
        snapshots = pd.DataFrame({
            'snapshot_date': [cutoff, cutoff], 'customer_id': ['001', '002'],
        })
        clean = pd.DataFrame({
            'customer_id': ['001'] * 5,
            'invoice': ['old', 'a', 'a', 'b', 'next'],
            'invoice_date': pd.to_datetime([
                '2020-12-31', '2021-01-01', '2021-01-01',
                '2021-01-30', '2021-01-31',
            ]),
        })
        schedule = pd.DataFrame({
            'snapshot_date': [cutoff],
            'label_end_exclusive': [cutoff + pd.Timedelta(days=30)],
        })
        counts, audit = build_future_invoice_counts(snapshots, clean, schedule)
        self.assertEqual(counts.invoice_count_30d_future.tolist(), [2, 0])
        self.assertEqual(audit.cohort_invoice_count.tolist(), [2])
        self.assertTrue(audit.reconciled.all())

    def test_count_amount_fit_refuses_immature_labels(self):
        fit = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-15']),
        })
        target = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-02-01']),
        })
        with self.assertRaisesRegex(ValueError, 'immature'):
            fit_predict_count_amount(fit, target)

    def test_count_amount_advancement_requires_all_tradeoffs(self):
        summary = pd.DataFrame([
            {'period': 'rolling_jan_may', 'method': 'extended_tweedie',
             'positive_mae': 100, 'mae': 50, 'top20_spend_capture': .70,
             'aggregate_bias': .10},
            {'period': 'rolling_jan_may', 'method': 'count_times_order_amount',
             'positive_mae': 94, 'mae': 50.5, 'top20_spend_capture': .71,
             'aggregate_bias': .08},
        ])
        self.assertTrue(all(count_amount_advancement_checks(summary).values()))
        summary.loc[summary.method.eq('count_times_order_amount'), 'aggregate_bias'] = .12
        self.assertFalse(all(count_amount_advancement_checks(summary).values()))

    def test_cdnow_adapter_retains_duplicate_transactions_and_excludes_zero_amount(self):
        raw = pd.DataFrame({
            'customer_id': pd.Series(['00001', '00001', '00001'], dtype='string'),
            'date': pd.Series(['19970101', '19970101', '19980630'], dtype='string'),
            'quantity': [1, 1, 1], 'amount': [11.77, 11.77, 0.0],
        })
        clean, audit = adapt_cdnow_transactions(raw)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean.invoice.nunique(), 2)
        self.assertEqual(clean.line_amount.tolist(), [11.77, 11.77])
        self.assertEqual(audit['excluded_zero_amount_rows'], 1)
        self.assertEqual(audit['source_amount_cents'], 2354)
        self.assertEqual(audit['eligible_amount_cents'], 2354)

    def test_cdnow_schedule_locks_full_months_and_last_three_tests(self):
        clean = pd.DataFrame({'invoice_date': pd.to_datetime(
            ['1997-01-01', '1998-06-30'])})
        schedule = build_cdnow_schedule(clean)
        self.assertEqual(len(schedule), 12)
        self.assertEqual(schedule.snapshot_date.iloc[0], pd.Timestamp('1997-07-01'))
        self.assertEqual(schedule.snapshot_date.iloc[-1], pd.Timestamp('1998-06-01'))
        self.assertEqual(schedule.loc[schedule.split.eq('locked_test'),
                                      'snapshot_date'].dt.strftime('%Y-%m').tolist(),
                         ['1998-04', '1998-05', '1998-06'])
        self.assertTrue(schedule.label_end_exclusive.max() <= pd.Timestamp('1998-07-01'))

    def test_calendar_features_are_known_before_prediction_and_cycle(self):
        rows = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-01-01', '2011-07-01']),
            'spend_30d': [0., 100.],
        })
        result = add_calendar_features(rows)
        self.assertAlmostEqual(result.loc[0, 'month_sin'], .5)
        self.assertAlmostEqual(result.loc[0, 'month_cos'], np.sqrt(3) / 2)
        self.assertAlmostEqual(result.loc[1, 'month_sin'], -.5)
        rows.loc[0, 'spend_30d'] = 1000.
        changed = add_calendar_features(rows)
        self.assertEqual(result.month_sin.tolist(), changed.month_sin.tolist())
        self.assertEqual(result.month_cos.tolist(), changed.month_cos.tolist())
        immature = pd.DataFrame({'snapshot_date': pd.to_datetime(['2011-01-01'])})
        target = pd.DataFrame({'snapshot_date': pd.to_datetime(['2011-01-15'])})
        with self.assertRaisesRegex(ValueError, 'immature'):
            fit_predict_calendar(immature, target)

    def test_actual_extreme_group_is_deterministic_and_not_a_forecast(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-01-01'] * 5),
            'customer_id': ['b', 'a', 'c', 'd', 'e'],
            'spend_30d': [100., 100., 10., 0., 0.],
        })
        selected = mark_actual_top_buyers(frame)
        self.assertEqual(frame.loc[selected, 'customer_id'].tolist(), ['a'])
        frame.loc[0, 'snapshot_date'] = pd.Timestamp('2011-02-01')
        with self.assertRaisesRegex(ValueError, 'one calendar month'):
            mark_actual_top_buyers(frame)

    def test_predicted_tail_calibration_uses_predicted_membership(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2011-01-01'] * 5),
            'customer_id': list('abcde'),
            'spend_30d': [10., 100., 0., 0., 0.],
            'extended_tweedie': [20., 0., 0., 0., 0.],
        })
        monthly, period = predicted_tail_metrics(frame)
        self.assertEqual(monthly.loc[0, 'actual_tail_overlap_count'], 0)
        self.assertEqual(monthly.loc[0, 'predicted_top1pct_count'], 1)
        self.assertEqual(monthly.loc[0, 'predicted_to_actual'], 2.)
        self.assertAlmostEqual(period.loc[0, 'actual_spend_share'], 10 / 110)

    def test_composition_features_use_only_past_transactions_and_reconcile(self):
        date = pd.Timestamp('2021-07-01')
        snapshots = pd.DataFrame({
            'snapshot_date': [date], 'customer_id': ['001'],
            'monetary_value_180d': [29.],
        })
        schedule = pd.DataFrame({
            'snapshot_date': [date],
            'feature_start': [date - pd.Timedelta(days=180)],
        })
        clean = pd.DataFrame({
            'customer_id': ['001'] * 3,
            'invoice_date': pd.to_datetime(['2021-01-02', '2021-06-30',
                                            '2021-07-01']),
            'quantity': [12, 1, 100], 'unit_price': [2., 5., 10.],
            'line_amount': [24., 5., 1000.],
            'country': ['United Kingdom', 'France', 'France'],
        })
        features, audit = build_composition_features(snapshots, clean, schedule)
        self.assertAlmostEqual(features.loc[0, 'bulk_spend_share_180d'], 24 / 29)
        self.assertEqual(features.loc[0, 'median_unit_price_180d'], 3.5)
        self.assertEqual(features.loc[0, 'non_uk_180d'], 1)
        self.assertEqual(audit.loc[0, 'observed_transaction_rows'], 2)
        snapshots.loc[0, 'monetary_value_180d'] = 30.
        with self.assertRaisesRegex(ValueError, 'do not reconcile'):
            build_composition_features(snapshots, clean, schedule)

    def test_buyer_weight_is_fixed_and_requires_valid_mature_amounts(self):
        self.assertEqual(buyer_weights(pd.Series([0., 1., 25.])).tolist(),
                         [1., 2., 2.])
        with self.assertRaisesRegex(ValueError, 'nonnegative mature'):
            buyer_weights(pd.Series([0., -1.]))
        with self.assertRaisesRegex(ValueError, 'nonnegative mature'):
            buyer_weights(pd.Series([float('nan')]))

    def test_buyer_weight_development_gate_checks_all_four_tradeoffs(self):
        summary = pd.DataFrame([
            {'period': 'rolling_jan_may', 'method': 'extended_tweedie',
             'positive_mae': 100., 'mae': 100., 'top20_spend_capture': .5,
             'aggregate_bias': .1},
            {'period': 'rolling_jan_may', 'method': 'buyer_weight_2x_tweedie',
             'positive_mae': 94., 'mae': 101., 'top20_spend_capture': .51,
             'aggregate_bias': .05},
        ])
        self.assertTrue(all(advancement_checks(summary).values()))
        summary.loc[1, 'aggregate_bias'] = .2
        self.assertFalse(advancement_checks(summary)['aggregate_bias_abs_not_larger'])

    def test_ranking_audit_keeps_list_stability_separate_from_spend_capture(self):
        ids = list('abcde')
        previous = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 5),
            'customer_id': ids,
            'historic_180d': [10., 0., 0., 0., 0.],
            'extended_tweedie': [10., 0., 0., 0., 0.],
            'extended_tweedie_previous_year': [10., 0., 0., 0., 0.],
        })
        current = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-02-01'] * 5),
            'customer_id': ids,
            'spend_30d': [10., 100., 0., 0., 0.],
            'historic_180d': [0., 10., 0., 0., 0.],
            'extended_tweedie': [10., 0., 0., 0., 0.],
            'extended_tweedie_previous_year': [10., 0., 0., 0., 0.],
        })
        result = pair_stability(previous, current, min_shared=5).set_index('method')
        self.assertEqual(result.loc['historic_180d', 'retention'], 0.)
        self.assertEqual(result.loc['extended_tweedie', 'retention'], 1.)
        self.assertAlmostEqual(result.loc['historic_180d', 'current_list_capture'],
                               100 / 110)
        self.assertAlmostEqual(result.loc['extended_tweedie', 'current_list_capture'],
                               10 / 110)

    def test_ranking_audit_recency_groups_use_only_pre_cutoff_days(self):
        groups = recency_group(pd.Series([0, 30, 31, 90, 91, 180]))
        self.assertEqual(groups.astype(str).tolist(),
                         ['0-30', '0-30', '31-90', '31-90', '91-180', '91-180'])
        with self.assertRaisesRegex(ValueError, 'Recency outside'):
            recency_group(pd.Series([181]))

    def test_amazon_adapter_and_locked_censoring_cutoff(self):
        raw = pd.DataFrame({
            'Survey ResponseID': pd.Series(['R_a', 'R_a', 'R_b'], dtype='string'),
            'Order Date': ['2020-01-02', '2020-01-03', '2020-01-04'],
            'Purchase Price Per Unit': [10.25, 5.0, -1.0],
            'Quantity': [2., 1., 1.],
            'ASIN/ISBN (Product Code)': pd.Series(['A', 'B', 'C'], dtype='string'),
        })
        clean, audit = adapt_amazon_transactions(raw)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean.line_amount.tolist(), [20.5, 5.0])
        self.assertEqual(clean.invoice.tolist(), ['R_a/0', 'R_a/1'])
        self.assertEqual(audit['excluded_rows'], 1)
        coverage = pd.DataFrame({'invoice_date': pd.to_datetime([
            '2018-01-01', '2022-11-01'])})
        schedule = build_amazon_schedule(coverage)
        self.assertEqual(schedule.snapshot_date.tail(6).tolist(),
                         pd.date_range('2022-04-01', '2022-09-01', freq='MS').tolist())
        self.assertLessEqual(schedule.label_end_exclusive.max(),
                             pd.Timestamp('2022-10-01'))

    def test_grouped_interval_uses_pre_score_group_residuals(self):
        calibration = pd.DataFrame({
            'spend_30d': [1.] * 100 + [100.] * 100,
            'predicted_spend_30d': [0.] * 200,
            'high_unconditional': [False] * 100 + [True] * 100,
        })
        current = pd.DataFrame({
            'predicted_spend_30d': [0., 0.],
            'high_unconditional': [False, True],
        })
        low, high, audit = calibrated_interval(
            calibration, current, 'unconditional', 'absolute_grouped')
        self.assertEqual(low.tolist(), [0., 0.])
        self.assertEqual(high.tolist(), [1., 100.])
        self.assertFalse(audit['ordinary_fell_back_to_all'])
        self.assertFalse(audit['high_fell_back_to_all'])

    def test_interval_quantile_and_mature_history_boundaries(self):
        self.assertEqual(finite_quantile(np.arange(1, 10, dtype=float)), 9)
        with self.assertRaisesRegex(ValueError, 'Too few'):
            finite_quantile(np.arange(1, 9, dtype=float))
        low, high = amount_interval(np.array([0., 10.]), 2., 'absolute')
        self.assertEqual(low.tolist(), [0., 8.])
        self.assertEqual(high.tolist(), [2., 12.])
        low, high = amount_interval(np.array([0., 10.]), 1., 'log1p')
        self.assertEqual(low[0], 0.)
        self.assertGreater(high[1], 10.)
        history = pd.DataFrame({'snapshot_date': pd.to_datetime([
            '2021-01-01', '2021-02-01'])})
        mature, dates = recent_mature_history(
            history, pd.Timestamp('2021-03-01'))
        self.assertEqual(dates, ['2021-01-01'])
        self.assertEqual(len(mature), 1)

    def test_segmented_buyer_amount_uses_past_groups_only(self):
        frame = pd.DataFrame({'customer_id': [f'{i:02d}' for i in range(20)]})
        high = high_amount_mask(frame, np.arange(20, dtype=float))
        self.assertEqual(int(high.sum()), 2)
        self.assertTrue(high[-2:].all())
        history = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 4),
            'actual_spend_30d': [10., 10., 20., 20.],
            'raw_predicted_spend_30d': [10., 10., 10., 10.],
            'high_amount_segment': [False, False, True, True],
        })
        low, high_factor = segmented_buyer_amount_factors(
            history, pd.Timestamp('2021-02-01'))
        self.assertEqual(low, 1.)
        self.assertEqual(high_factor, 1.15)
        with self.assertRaisesRegex(ValueError, 'immature'):
            segmented_buyer_amount_factors(
                history, pd.Timestamp('2021-01-15'))

    def test_monotone_buyer_amount_uses_mature_oof_buyer_history(self):
        history = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 4),
            'actual_spend_30d': [20., 40., 80., 100.],
            'raw_predicted_spend_30d': [10., 20., 30., 40.],
        })
        result = monotone_buyer_amount(
            history, pd.Timestamp('2021-02-01'), np.array([10., 40.]),
            min_buyers=2)
        self.assertTrue((np.diff(result) >= 0).all())
        self.assertGreater(result[-1], 40)
        self.assertEqual(monotone_buyer_amount(
            history, pd.Timestamp('2021-02-01'), np.array([10.]),
            min_buyers=500).tolist(), [10.])
        with self.assertRaisesRegex(ValueError, 'immature'):
            monotone_buyer_amount(
                history, pd.Timestamp('2021-01-15'), np.array([10.]),
                min_buyers=2)

    def test_recent_buyer_amount_weights_use_only_mature_cutoffs(self):
        target = pd.Timestamp('2021-07-01')
        dates = pd.Series([target - pd.Timedelta(days=180),
                           target - pd.Timedelta(days=30)])
        weights = recent_amount_weights(dates, target)
        self.assertAlmostEqual(weights[0], .5)
        self.assertGreater(weights[1], weights[0])
        with self.assertRaisesRegex(ValueError, 'immature'):
            recent_amount_weights(pd.Series([target - pd.Timedelta(days=29)]), target)

    def test_customer_cluster_intervals_are_paired_and_reproducible(self):
        data = pd.DataFrame({
            'snapshot_date': [pd.Timestamp('2021-01-01')] * 100,
            'customer_id': [f'{i:03d}' for i in range(100)],
            'spend_30d': [0.] * 80 + [10.] * 20,
            'historic_180d': [0.] * 100,
            'extended_tweedie_previous_year': [0.] * 80 + [5.] * 20,
            'expected_spend_lag_calibrated': [0.] * 80 + [5.] * 20,
            'median_if_purchase_prior_year': [0.] * 80 + [10.] * 20,
            'average_order_value_180d': [1.] * 100,
        })
        first, details = bootstrap_customer_intervals(data, seed=7, resamples=5)
        second, _ = bootstrap_customer_intervals(data, seed=7, resamples=5)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(details['customer_clusters'], 100)
        self.assertGreater(first.loc[
            first.metric.eq('prior_year_tweedie_mae_reduction'),
            'point'].iloc[0], 0)

    def test_hurdle_error_decomposition_reconciles_without_label_input(self):
        result = decompose_spend(
            np.array([10., 0.]), np.array([.8, .1]),
            np.array([20., 5.]), np.array([9., 4.]),
            np.array([7., 3.]))
        self.assertAlmostEqual(result['incidence_component_bias'], -.35)
        self.assertAlmostEqual(result['buyer_amount_component_bias'], 1.0)
        self.assertAlmostEqual(result['aggregate_bias'], .65)
        self.assertAlmostEqual(result['conditional_median_buyer_mae'], 1)
        self.assertAlmostEqual(result['prior_average_order_buyer_mae'], 3)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            decompose_spend(np.array([10., 0.]), np.array([1.1, .1]),
                            np.array([20., 5.]), np.array([9., 4.]),
                            np.array([7., 3.]))

    def test_jewelry_adapter_preserves_ids_and_excludes_unpriced_rows(self):
        raw = pd.DataFrame({
            'event_time': ['2020-01-01 12:00:00 UTC'] * 3,
            'order_id': pd.Series(['1924719191579951782', '1924719191579951783',
                                   '1924719191579951784'], dtype='string'),
            'product_id': pd.Series(['1842195256808833386'] * 3, dtype='string'),
            'quantity': [2, 1, 1], 'price': [10.51, np.nan, -2.0],
            'user_id': pd.Series(['1515915625207851155'] * 3, dtype='string'),
        })
        clean, audit = adapt_jewelry_transactions(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.iloc[0].invoice,
                         '1515915625207851155/1924719191579951782')
        self.assertEqual(clean.iloc[0].line_amount, 21.02)
        self.assertEqual(clean.iloc[0].invoice_date, pd.Timestamp('2020-01-01 12:00'))
        self.assertEqual(audit['eligible_amount_cents'], 2102)
        self.assertEqual(audit['excluded_rows'], 2)

    def test_jewelry_schedule_reserves_last_three_complete_months(self):
        clean = pd.DataFrame({'invoice_date': pd.to_datetime([
            '2019-01-01', '2021-12-01'])})
        schedule = build_jewelry_schedule(clean)
        self.assertEqual(schedule.snapshot_date.iloc[0], pd.Timestamp('2019-07-01'))
        self.assertEqual(schedule.snapshot_date.iloc[-1], pd.Timestamp('2021-11-01'))
        self.assertEqual(schedule.split.tail(6).tolist(),
                         ['validation'] * 3 + ['test'] * 3)
        self.assertTrue((schedule.label_end_exclusive.iloc[-3:]
                         <= pd.Timestamp('2021-12-02')).all())

    def test_prior_year_window_uses_past_only_and_distinguishes_no_coverage(self):
        cutoff = pd.Timestamp('2021-07-01')
        snapshots = pd.DataFrame({
            'snapshot_date': [cutoff, cutoff], 'customer_id': ['1', '2'],
        })
        clean = pd.DataFrame({
            'customer_id': ['1', '1', '1', '1'],
            'invoice_date': [cutoff - pd.Timedelta(days=365),
                             cutoff - pd.Timedelta(days=336),
                             cutoff - pd.Timedelta(days=335),
                             cutoff],
            'line_amount': [10, 20, 30, 999],
        })
        result = build_seasonal_features(snapshots, clean)
        self.assertEqual(result.previous_year_30d_spend.tolist(), [30, 0])
        clean = clean.loc[clean.invoice_date.gt(cutoff - pd.Timedelta(days=365))]
        result = build_seasonal_features(snapshots, clean)
        self.assertTrue(result.previous_year_30d_spend.isna().all())

    def test_lagged_calibration_uses_only_mature_history_and_clips_ratio(self):
        history = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2020-12-01', '2021-01-01']),
            'actual_spend_30d': [100, 100],
            'raw_predicted_spend_30d': [50, 50],
        })
        self.assertEqual(lagged_amount_factor(history, pd.Timestamp('2021-02-01')),
                         1.15)
        with self.assertRaisesRegex(ValueError, 'immature'):
            lagged_amount_factor(history, pd.Timestamp('2021-01-15'))
        february = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-02-01']),
            'actual_spend_30d': [100], 'raw_predicted_spend_30d': [90],
        })
        with self.assertRaisesRegex(ValueError, 'immature'):
            lagged_amount_factor(february, pd.Timestamp('2021-03-01'))
        self.assertEqual(lagged_amount_factor(pd.DataFrame(),
                                              pd.Timestamp('2021-01-01')), 1)

    def test_two_year_cutoffs_keep_complete_future_labels_and_last_three(self):
        schedule = build_two_year_schedule(711)
        self.assertEqual(schedule.source_day.tolist(), list(range(181, 682, 30)))
        self.assertEqual(schedule.loc[schedule.split.eq('test'),
                                      'source_day'].tolist(), [601, 631, 661])
        self.assertTrue((schedule.source_day + 29 <= 711).all())
        self.assertTrue((schedule.snapshot_date - schedule.feature_start
                         == pd.Timedelta(days=180)).all())

    def test_conditional_metric_scores_buyers_only_and_rejects_invalid_scores(self):
        actual = np.array([0, 10, 20], dtype=float)
        predicted = np.array([999, 8, 24], dtype=float)
        metric = conditional_metrics(actual, predicted)
        self.assertEqual(metric['buyer_count'], 2)
        self.assertEqual(metric['buyer_mae'], 3)
        self.assertAlmostEqual(metric['buyer_amount_bias'], 32 / 30 - 1)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            conditional_metrics(actual, np.array([0, -1, 2], dtype=float))

    def test_forward_fit_waits_for_complete_label_window(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2020-11-01', '2020-12-01',
                                             '2020-12-03', '2021-01-01']),
        })
        fit = fit_rows_for_date(frame, pd.Timestamp('2021-01-01'))
        self.assertEqual(fit.snapshot_date.dt.strftime('%Y-%m-%d').tolist(),
                         ['2020-11-01', '2020-12-01'])

    def test_temporal_amounts_use_only_pre_cutoff_windows(self):
        cutoff = pd.Timestamp('2021-07-01')
        start = cutoff - pd.Timedelta(days=180)
        snapshots = pd.DataFrame({
            'snapshot_date': [cutoff], 'customer_id': ['001'],
            'monetary_value_180d': [360.0],
        })
        schedule = pd.DataFrame({'snapshot_date': [cutoff], 'feature_start': [start]})
        offsets = [-181, -180, -91, -90, -61, -60, -31, -30, -1, 0]
        amounts = [99, 10, 20, 30, 40, 50, 60, 70, 80, 999]
        clean = pd.DataFrame({
            'customer_id': ['001'] * len(offsets),
            'invoice': [str(i) for i in range(len(offsets))],
            'invoice_date': [cutoff + pd.Timedelta(days=d) for d in offsets],
            'line_amount': amounts,
        })
        result = build_temporal_features(snapshots, clean, schedule).iloc[0]
        self.assertEqual(result.monetary_value_30d, 150)
        self.assertEqual(result.monetary_value_60d, 260)
        self.assertEqual(result.monetary_value_90d, 330)
        self.assertEqual(result.monetary_value_previous_90d, 30)
        self.assertEqual(result.last_invoice_value_180d, 80)
        self.assertEqual(result.median_invoice_value_180d, 45)
        self.assertEqual(result.active_purchase_days_180d, 8)
        self.assertEqual(result.invoice_count_previous_90d, 2)
        self.assertEqual(result.lifetime_spend_before_cutoff, 459)
        self.assertEqual(result.spend_before_180d_window, 99)
        self.assertEqual(result.lifetime_invoice_count_before_cutoff, 9)
        snapshots.loc[0, 'monetary_value_180d'] = 359
        with self.assertRaisesRegex(ValueError, 'does not reconcile'):
            build_temporal_features(snapshots, clean, schedule)

    def test_label_join_rejects_split_or_repurchase_mismatch(self):
        snapshots = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01']),
            'customer_id': pd.Series(['001'], dtype='string'),
            'split': ['train'], 'repurchase_30d': [1],
        })
        labels = pd.DataFrame({
            'snapshot_date': ['2021-01-01'], 'customer_id': ['001'],
            'split': ['train'], 'repurchase_30d': [1], 'spend_30d': [12.50],
        })
        self.assertEqual(attach_labels(snapshots, labels).spend_30d.tolist(), [12.50])
        labels.loc[0, 'repurchase_30d'] = 0
        with self.assertRaisesRegex(ValueError, 'differs'):
            attach_labels(snapshots, labels)

    def test_spend_capture_uses_monthly_quota_and_actual_amount(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 5 + ['2021-02-01'] * 5),
            'customer_id': [str(i) for i in range(10)],
            'spend_30d': [100, 0, 0, 0, 0, 0, 200, 0, 0, 0],
        })
        scores = np.array([.9, .8, .7, .6, .5, .1, .9, .3, .2, .0])
        self.assertEqual(int(top20_mask(frame, scores).sum()), 2)
        metrics = evaluate(frame, scores)
        self.assertEqual(metrics['top20_actual_spend'], 300)
        self.assertEqual(metrics['top20_spend_capture'], 1)

    def test_selection_requires_error_capture_and_monthly_stability(self):
        metrics = pd.DataFrame([
            {'method': 'historic_30d', 'mae': 100, 'top20_spend_capture': .5,
             'aggregate_bias': 0},
            {'method': 'direct_tweedie', 'mae': 90, 'top20_spend_capture': .6,
             'aggregate_bias': -.08},
            {'method': 'two_stage', 'mae': 80, 'top20_spend_capture': .4,
             'aggregate_bias': 0},
            {'method': 'direct_l2', 'mae': 95, 'top20_spend_capture': .52,
             'aggregate_bias': -.15},
        ])
        monthly = pd.DataFrame([
            {'method': name, 'snapshot_date': str(month), 'mae': error,
             'top20_spend_capture': capture}
            for name, values in {
                'historic_30d': [(100, .5)] * 3,
                'direct_tweedie': [(90, .6), (95, .55), (120, .4)],
                'two_stage': [(80, .4)] * 3,
                'direct_l2': [(95, .52)] * 3,
            }.items()
            for month, (error, capture) in enumerate(values)
        ])
        self.assertEqual(choose_model(metrics, monthly), 'direct_tweedie')
        metrics.loc[metrics.method.eq('direct_tweedie'), 'top20_spend_capture'] = .49
        self.assertIsNone(choose_model(metrics, monthly))

    def test_zero_forecast_has_no_meaningful_top20_rank(self):
        frame = pd.DataFrame({
            'snapshot_date': pd.to_datetime(['2021-01-01'] * 5),
            'customer_id': [str(i) for i in range(5)],
            'split': ['validation'] * 5,
            'repurchase_30d': [1, 0, 0, 0, 0],
            'spend_30d': [100, 0, 0, 0, 0],
        })
        overall, monthly, rows = compare_months(
            frame, {'zero': np.zeros(5)}, 'validation')
        self.assertTrue(pd.isna(overall.loc[0, 'top20_spend_capture']))
        self.assertTrue(pd.isna(monthly.loc[0, 'top20_spend_capture']))
        self.assertFalse(rows.top20_selected.any())

    def test_display_gate_checks_total_amount_bias(self):
        metrics = pd.DataFrame([
            {'method': 'historic_30d', 'mae': 100, 'top20_spend_capture': .5,
             'aggregate_bias': 0},
            {'method': 'direct_tweedie', 'mae': 90, 'top20_spend_capture': .6,
             'aggregate_bias': -.17},
        ])
        monthly = pd.DataFrame([
            {'method': method, 'snapshot_date': str(month),
             'mae': mae, 'top20_spend_capture': capture}
            for method, mae, capture in [('historic_30d', 100, .5),
                                         ('direct_tweedie', 90, .6)]
            for month in range(3)
        ])
        self.assertFalse(candidate_passes(metrics, monthly, 'direct_tweedie'))
        metrics.loc[metrics.method.eq('direct_tweedie'), 'aggregate_bias'] = -.08
        self.assertTrue(candidate_passes(metrics, monthly, 'direct_tweedie'))
        incomplete = monthly.loc[monthly.snapshot_date.ne('2')]
        self.assertFalse(candidate_passes(metrics, incomplete, 'direct_tweedie'))


if __name__ == '__main__':
    unittest.main()
