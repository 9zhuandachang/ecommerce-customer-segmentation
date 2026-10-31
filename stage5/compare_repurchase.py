"""Compare four repurchase learners and three feature sets without test selection.

Run from the project root: python -B stage5/compare_repurchase.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from stage4.kmeans import FEATURES, KEYS, LABEL, load_snapshots
from stage5.experiments import top20_metrics

SNAPSHOTS = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
TRANSACTIONS = ROOT / 'data/processed/customer_snapshot_pipeline/clean_transactions.csv'
OUTPUT = ROOT / 'stage5/outputs/repurchase_comparison'
FOLD_DATES = pd.to_datetime(['2011-03-01', '2011-04-01', '2011-05-01'])
TREND = ['recent90_invoice_share', 'monthly_order_change', 'has_90d_history']
RHYTHM = ['active_30d_periods_180d', 'last_invoice_gap_days_180d', 'has_two_invoices_180d']
FEATURE_SETS = {
    'base_8': FEATURES,
    'base_8_plus_trend': FEATURES + TREND,
    'base_8_plus_trend_and_rhythm': FEATURES + TREND + RHYTHM,
}
MODEL_OPTIONS = {
    'LR': (0.1, 1.0),
    'SplineLR': (0.1, 1.0),
    'XGBoost': (2, 3),
    'LightGBM': (7, 15),
}
LOG_COLUMNS = {
    'invoice_frequency_180d', 'monetary_value_180d',
    'average_order_value_180d', 'product_diversity_180d',
    'last_invoice_gap_days_180d',
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_invoice_dates(path: Path) -> pd.DataFrame:
    lines = pd.read_csv(
        path, usecols=['customer_id', 'invoice', 'invoice_date'],
        dtype={'customer_id': 'string', 'invoice': 'string'},
        parse_dates=['invoice_date'],
    )
    if lines.isna().any().any():
        raise ValueError('Clean transactions have null invoice keys or dates')
    # Match the canonical snapshot builder: each customer/invoice has its latest line date.
    return (lines.groupby(['customer_id', 'invoice'], as_index=False,
                          observed=True).invoice_date.max()
            .sort_values(['invoice_date', 'customer_id', 'invoice'], kind='stable')
            .reset_index(drop=True))


def rhythm_for_cutoff(invoices: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Every field uses only invoices in [cutoff - 180 days, cutoff)."""
    window = invoices.loc[
        invoices.invoice_date.ge(cutoff - pd.Timedelta(days=180))
        & invoices.invoice_date.lt(cutoff)
    ].copy()
    window = window.sort_values(['customer_id', 'invoice_date', 'invoice'], kind='stable')
    grouped = window.groupby('customer_id', observed=True)
    count = grouped.invoice.size().rename('invoice_count_check')
    age = (cutoff - window.invoice_date).dt.total_seconds() / 86400
    window['age_bin'] = np.floor(age / 30).astype(int)
    if not window.age_bin.between(0, 5).all():
        raise AssertionError('An invoice was used outside the prior 180-day window')
    active = (window.groupby('customer_id', observed=True).age_bin.nunique()
              .rename('active_30d_periods_180d'))
    gaps = grouped.invoice_date.diff().dt.total_seconds() / 86400
    last_gap = gaps.groupby(window.customer_id, observed=True).last().rename(
        'last_invoice_gap_days_180d')
    output = pd.concat([count, active, last_gap], axis=1)
    output['has_two_invoices_180d'] = (output.invoice_count_check >= 2).astype(int)
    output['last_invoice_gap_days_180d'] = output.last_invoice_gap_days_180d.fillna(180)
    return output.reset_index()


def add_features(snapshots: pd.DataFrame, invoices: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cutoff, month in snapshots.groupby('snapshot_date', sort=True):
        extra = rhythm_for_cutoff(invoices, pd.Timestamp(cutoff))
        merged = month.merge(extra, on='customer_id', how='left', validate='one_to_one')
        if merged.invoice_count_check.isna().any():
            raise AssertionError(f'No historical invoice for some snapshots at {cutoff}')
        if not merged.invoice_count_check.eq(merged.invoice_frequency_180d).all():
            raise AssertionError(f'Invoice counts differ from canonical snapshots at {cutoff}')
        parts.append(merged)
    result = pd.concat(parts, ignore_index=True)
    result['recent90_invoice_share'] = (
        result.invoice_count_90d / result.invoice_frequency_180d)
    result['has_90d_history'] = (result.customer_history_days >= 90).astype(int)
    previous60 = result.invoice_count_90d - result.invoice_count_30d
    if (previous60 < 0).any():
        raise AssertionError('Nested invoice windows are inconsistent')
    result['monthly_order_change'] = np.where(
        result.has_90d_history.eq(1),
        result.invoice_count_30d - previous60 / 2, 0.0)
    if not np.isfinite(result[FEATURE_SETS['base_8_plus_trend_and_rhythm']].to_numpy()).all():
        raise AssertionError('Non-finite model input')
    return result.sort_values(KEYS[:2], kind='stable').reset_index(drop=True)


def forward_fold(train: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = train.loc[train.snapshot_date.add(pd.Timedelta(days=30)).le(cutoff)]
    hold = train.loc[train.snapshot_date.eq(cutoff)]
    if fit.empty or hold.empty or not fit.snapshot_date.lt(cutoff).all():
        raise ValueError(f'Invalid forward fold: {cutoff}')
    return fit, hold


def make_model(name: str, complexity: float | int, columns: list[str]):
    if name in ('LR', 'SplineLR'):
        logs = [column for column in columns if column in LOG_COLUMNS]
        linear = [column for column in columns if column not in LOG_COLUMNS
                  and column != 'recency_days']
        blocks = [
            ('log', Pipeline([('log1p', FunctionTransformer(np.log1p)),
                              ('scale', StandardScaler())]), logs),
            ('linear', StandardScaler(), linear),
        ]
        if name == 'SplineLR':
            blocks.append(('recency_spline', SplineTransformer(
                n_knots=4, degree=3, knots='quantile',
                extrapolation='constant', include_bias=False), ['recency_days']))
        else:
            blocks.append(('recency', StandardScaler(), ['recency_days']))
        return Pipeline([
            ('features', ColumnTransformer(blocks, sparse_threshold=0)),
            ('classifier', LogisticRegression(
                C=float(complexity), class_weight=None,
                max_iter=2000, random_state=42)),
        ])
    if name == 'XGBoost':
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=int(complexity),
            min_child_weight=10, reg_lambda=3, subsample=0.9,
            colsample_bytree=1, objective='binary:logistic',
            tree_method='hist', grow_policy='depthwise', eval_metric='logloss',
            random_state=42, n_jobs=4)
    if name == 'LightGBM':
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=int(complexity),
            min_child_samples=100, reg_lambda=3, class_weight=None,
            deterministic=True, force_col_wise=True,
            random_state=42, n_jobs=4, verbosity=-1)
    raise KeyError(name)


def score_model(model, fit: pd.DataFrame, hold: pd.DataFrame,
                columns: list[str]) -> np.ndarray:
    model.fit(fit[columns], fit[LABEL].to_numpy(dtype=int))
    scores = model.predict_proba(hold[columns])[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise AssertionError('Invalid probability scores')
    return scores


def evaluate(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    y = frame[LABEL].to_numpy(dtype=int)
    return {
        'ap': float(average_precision_score(y, scores)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'brier': float(brier_score_loss(y, scores)),
        'mean_score': float(scores.mean()),
        'observed_rate': float(y.mean()),
        **top20_metrics(frame, scores, y),
    }


def choose_configuration(fold_rows: pd.DataFrame) -> pd.DataFrame:
    summary = (fold_rows.groupby(['feature_set', 'model', 'complexity'], sort=False)
               .agg(mean_fold_ap=('ap', 'mean'),
                    min_fold_ap=('ap', 'min'),
                    mean_fold_precision_at_20pct=('precision_at_20pct', 'mean'))
               .reset_index())
    summary = summary.sort_values(
        ['feature_set', 'model', 'mean_fold_ap', 'complexity'],
        ascending=[True, True, False, True], kind='stable')
    summary['selected_configuration'] = ~summary.duplicated(['feature_set', 'model'])
    return summary


def choose_winner(comparison: pd.DataFrame) -> pd.DataFrame:
    table = comparison.copy()
    reference = table.loc[(table.feature_set == 'base_8') & (table.model == 'LR')]
    if len(reference) != 1:
        raise ValueError('Missing unique base-eight LR reference')
    base = reference.iloc[0]
    table['eligible'] = (
        table.mean_fold_ap.ge(base.mean_fold_ap - 1e-12)
        & table.precision_at_20pct.ge(base.precision_at_20pct - 1e-12))
    table = table.sort_values(
        ['eligible', 'ap', 'precision_at_20pct', 'feature_count', 'model'],
        ascending=[False, False, False, True, True], kind='stable')
    table['selected'] = False
    table.loc[table.index[0], 'selected'] = True
    return table.reset_index(drop=True)


def paired_customer_bootstrap(predictions: pd.DataFrame, comparison: pd.DataFrame,
                              draws: int = 400, seed: int = 1801) -> pd.DataFrame:
    """Post-selection descriptive AP differences; resample whole customers."""
    chosen = comparison.loc[comparison.selected].iloc[0]
    alternatives = [
        comparison.loc[~comparison.selected].sort_values('ap', ascending=False).iloc[0],
        comparison.loc[(comparison.feature_set == 'base_8')
                       & (comparison.model == 'LR')].iloc[0],
    ]
    key_columns = ['snapshot_date', 'customer_id']
    current = predictions.loc[(predictions.feature_set == chosen.feature_set)
                              & (predictions.model == chosen.model)]
    current = current.sort_values(key_columns, kind='stable').reset_index(drop=True)
    customer_codes, customers = pd.factorize(current.customer_id, sort=True)
    rng = np.random.default_rng(seed)
    rows = []
    for other in alternatives:
        reference = predictions.loc[(predictions.feature_set == other.feature_set)
                                    & (predictions.model == other.model)]
        reference = reference.sort_values(key_columns, kind='stable').reset_index(drop=True)
        if not current[key_columns + [LABEL]].equals(reference[key_columns + [LABEL]]):
            raise AssertionError('Paired validation records do not match')
        labels = current[LABEL].to_numpy(dtype=int)
        score_a = current.score.to_numpy(dtype=float)
        score_b = reference.score.to_numpy(dtype=float)
        deltas = []
        for _ in range(draws):
            selected_customers = rng.integers(0, len(customers), size=len(customers))
            weights = np.bincount(selected_customers, minlength=len(customers))[
                customer_codes]
            deltas.append(
                average_precision_score(labels, score_a, sample_weight=weights)
                - average_precision_score(labels, score_b, sample_weight=weights))
        rows.append({
            'selected_feature_set': chosen.feature_set, 'selected_model': chosen.model,
            'reference_feature_set': other.feature_set, 'reference_model': other.model,
            'observed_ap_delta': chosen.ap - other.ap,
            'bootstrap_low_2_5': float(np.percentile(deltas, 2.5)),
            'bootstrap_high_97_5': float(np.percentile(deltas, 97.5)),
            'resampling_unit': 'customer', 'unique_customers': len(customers),
            'draws': draws, 'seed': seed,
        })
    return pd.DataFrame(rows)


def main() -> None:
    snapshots, _ = load_snapshots(SNAPSHOTS)
    snapshots = snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy()
    invoices = read_invoice_dates(TRANSACTIONS)
    data = add_features(snapshots, invoices)
    train = data.loc[data.split.eq('train')].copy()
    validation = data.loc[data.split.eq('validation')].copy()
    if (train.snapshot_date.max() + pd.Timedelta(days=30)
            > validation.snapshot_date.min()):
        raise AssertionError('Training label window crosses validation start')

    folds = {date: forward_fold(train, date) for date in FOLD_DATES}
    fold_rows = []
    for feature_set, columns in FEATURE_SETS.items():
        for name, options in MODEL_OPTIONS.items():
            for complexity in options:
                for date, (fit, hold) in folds.items():
                    model = make_model(name, complexity, columns)
                    scores = score_model(model, fit, hold, columns)
                    fold_rows.append({
                        'feature_set': feature_set, 'model': name,
                        'complexity': complexity, 'fold': date.date().isoformat(),
                        'fit_rows': len(fit), 'hold_rows': len(hold),
                        'latest_fit_label_end':
                            (fit.snapshot_date.max() + pd.Timedelta(days=30)).date().isoformat(),
                        **evaluate(hold, scores),
                    })
                print(f'folds: {feature_set} / {name} / {complexity}', flush=True)
    fold_table = pd.DataFrame(fold_rows)
    configurations = choose_configuration(fold_table)
    selected_configs = configurations.loc[configurations.selected_configuration]

    comparison_rows, monthly_rows, predictions = [], [], []
    fitted = {}
    for config in selected_configs.itertuples(index=False):
        columns = FEATURE_SETS[config.feature_set]
        model = make_model(config.model, config.complexity, columns)
        scores = score_model(model, train, validation, columns)
        fitted[(config.feature_set, config.model)] = model
        comparison_rows.append({
            'feature_set': config.feature_set, 'feature_count': len(columns),
            'model': config.model, 'complexity': config.complexity,
            'mean_fold_ap': config.mean_fold_ap,
            'min_fold_ap': config.min_fold_ap,
            'train_rows': len(train), 'validation_rows': len(validation),
            **evaluate(validation, scores),
        })
        prediction = validation[KEYS + [LABEL]].copy()
        prediction['feature_set'] = config.feature_set
        prediction['model'] = config.model
        prediction['score'] = scores
        predictions.append(prediction)
        for date, group in prediction.groupby('snapshot_date', sort=True):
            monthly_rows.append({
                'feature_set': config.feature_set, 'model': config.model,
                'snapshot_date': date.date().isoformat(), 'rows': len(group),
                **evaluate(group, group.score.to_numpy()),
            })
        print(f'validation: {config.feature_set} / {config.model} '
              f'AP={comparison_rows[-1]["ap"]:.4f}', flush=True)

    comparison = choose_winner(pd.DataFrame(comparison_rows))
    winner = comparison.loc[comparison.selected].iloc[0]
    prediction_table = pd.concat(predictions, ignore_index=True)
    uncertainty = paired_customer_bootstrap(prediction_table, comparison)
    winner_key = (winner.feature_set, winner.model)
    selected_model = fitted[winner_key]
    if winner.model in ('LR', 'SplineLR'):
        selected_parameters = {
            'C': float(winner.complexity), 'regularization': 'L2',
            'class_weight': None, 'max_iter': 2000,
            'recency_spline_knots': 4 if winner.model == 'SplineLR' else None,
        }
    elif winner.model == 'XGBoost':
        selected_parameters = {key: selected_model.get_params()[key] for key in (
            'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
            'reg_lambda', 'subsample', 'colsample_bytree', 'tree_method',
            'grow_policy', 'random_state')}
    else:
        selected_parameters = {key: selected_model.get_params()[key] for key in (
            'n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples',
            'reg_lambda', 'deterministic', 'random_state')}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(OUTPUT / 'forward_folds.csv', index=False, encoding='utf-8-sig')
    configurations.to_csv(OUTPUT / 'configuration_selection.csv', index=False,
                          encoding='utf-8-sig')
    comparison.to_csv(OUTPUT / 'validation_comparison.csv', index=False,
                      encoding='utf-8-sig')
    pd.DataFrame(monthly_rows).to_csv(OUTPUT / 'monthly_validation.csv', index=False,
                                      encoding='utf-8-sig')
    prediction_table.to_csv(
        OUTPUT / 'validation_predictions.csv', index=False, encoding='utf-8-sig')
    uncertainty.to_csv(OUTPUT / 'paired_uncertainty.csv', index=False,
                       encoding='utf-8-sig')
    data[KEYS + RHYTHM].to_csv(OUTPUT / 'rhythm_features.csv', index=False,
                               encoding='utf-8-sig')
    joblib.dump({
        'model': selected_model, 'features': FEATURE_SETS[winner.feature_set],
        'feature_set': winner.feature_set, 'model_name': winner.model,
        'complexity': winner.complexity, 'parameters': selected_parameters,
        'trained_through': '2011-05-01',
    }, OUTPUT / 'selected_model.joblib')

    manifest = {
        'snapshot_sha256': file_hash(SNAPSHOTS),
        'clean_transactions_sha256': file_hash(TRANSACTIONS),
        'script_sha256': file_hash(Path(__file__)),
        'protocol_sha256': file_hash(ROOT / 'stage5/repurchase_comparison_protocol.md'),
        'feature_sets': FEATURE_SETS,
        'model_options': MODEL_OPTIONS,
        'forward_fold_dates': [date.date().isoformat() for date in FOLD_DATES],
        'selection_rule': 'validation AP maximum among candidates whose fold mean AP and validation Precision@20% are each >= base_8 LR',
        'selected': {'feature_set': winner.feature_set, 'model': winner.model,
                     'complexity': winner.complexity,
                     'parameters': selected_parameters},
        'post_selection_uncertainty': '400 paired customer bootstrap draws for AP differences; descriptive only, not used for selection',
        'train_rows': len(train), 'validation_rows': len(validation),
        'test_labels_used_for_fit_selection_or_metrics': False,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'lightgbm': lightgbm.__version__},
    }
    (OUTPUT / 'run_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Stage 5: Re-comparison of repurchase models', '',
        'This experiment compares four models and three sets of features on the same batch of customer monthly snapshots. The training period is 2010 - 06 to 2011 - 05,'
        'The formal verification period is 2011 - 06 to 2011 - 08; the test tags did not participate in this experiment.',
        'The project has looked at this historical data before, so the results are a retrospective comparison and not a new independent verification.', '',
        'Selection rules: First select the complexity for each combination in the three forward months of the training period, and then select the model according to the formal verification period AP;'
        'The candidate\'s forward average AP and validation period Top 20% accuracy cannot be lower than the basic eight items LR.', '',
        '| Feature group | Model | Parameters | Forward average AP | Validation AP | Top 20% accuracy | ROC-AUC | Brier | Selection criteria | Final selection |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for row in comparison.itertuples():
        lines.append(
            f'| {row.feature_set} | {row.model} | {row.complexity:g} | '
            f'{row.mean_fold_ap:.4f} | {row.ap:.4f} | '
            f'{row.precision_at_20pct:.4f} | {row.roc_auc:.4f} | '
            f'{row.brier:.4f} | {"Yes" if row.eligible else "No"} |'
            f'{"Yes" if row.selected else "No"} |')
    lines += [
        '', f'**Selected: {winner.feature_set} / {winner.model}, complexity parameter {winner.complexity:g}.**',
        'Complete parameters of the selected model: `' + json.dumps(selected_parameters, ensure_ascii=False) + '`。',
        f'Verify AP = {winner.ap:.4f}, Top 20% accuracy = {winner.precision_at_20pct:.4f};'
        f'The average rating during the verification period = {winner.mean_score:.4f}, and the actual repurchase rate = {winner.observed_rate:.4f}.',
        '', 'Afterwards, re-sample and check the AP difference according to customer groups (not involved in model selection):',
        '| Comparison object | Observed AP difference | 95% resampling interval |',
        '| --- | ---: | ---: |',
    ]
    for row in uncertainty.itertuples():
        lines.append(f'| {row.reference_feature_set} / {row.reference_model} | '
                     f'{row.observed_ap_delta:+.4f} | '
                     f'[{row.bootstrap_low_2_5:+.4f}, {row.bootstrap_high_97_5:+.4f}] |')
    lines += [
        'This interval reflects only the resampling fluctuations of these validation customers; the model was selected during the same validation period,'
        'and is only for three months, so it is not a confident guarantee of future performance.',
        '', 'The monthly results are shown in `monthly_validation.csv`; the forward results in each round are shown in `forward_folds.csv`;'
        'See `validation_predictions.csv` for all validation scores. The selected model is only fit in the training period,'
        'Save as `selected_model.joblib`. This experiment has not yet replaced the Stage Six or Dashboard models.',
        '', 'It is impossible to infer true marketing increments or future new data performance based on these historical indicators alone.',
    ]
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Selected: {winner.feature_set} / {winner.model} / {winner.complexity:g}', flush=True)
    print(f'Artifacts: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
