"""Mature-label-only rolling amount calibration for deployed forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def lagged_amount_factor(history: pd.DataFrame,
                         target_date: pd.Timestamp) -> float:
    """Use at most three earlier complete 30-day cohorts, with shrinkage."""
    if history.empty:
        return 1.0
    required = {'snapshot_date', 'actual_spend_30d', 'raw_predicted_spend_30d'}
    if not required.issubset(history.columns):
        raise ValueError(f'Missing calibration columns: {sorted(required - set(history.columns))}')
    dates = pd.to_datetime(history.snapshot_date, errors='raise')
    target_date = pd.Timestamp(target_date)
    if ((dates + pd.Timedelta(days=30) > target_date).any()
            or dates.isna().any()):
        raise ValueError('Calibration history includes an immature future label')
    values = history[['actual_spend_30d', 'raw_predicted_spend_30d']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Calibration values must be finite and nonnegative')
    allowed_dates = sorted(dates.unique())[-3:]
    recent = history.loc[dates.isin(allowed_dates)]
    predicted = float(recent.raw_predicted_spend_30d.sum())
    if predicted <= 0:
        raise ValueError('Calibration denominator must be positive')
    actual = float(recent.actual_spend_30d.sum())
    ratio = float(np.clip(actual / predicted, .7, 1.3))
    return 1 + .5 * (ratio - 1)


def monotone_buyer_amount(history: pd.DataFrame,
                          target_date: pd.Timestamp,
                          raw_amount: np.ndarray,
                          min_buyers: int = 500) -> np.ndarray:
    """Shrink a monotone buyer-amount map learned only from mature OOF buyers."""
    raw = np.asarray(raw_amount, dtype=float)
    if (raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any()
            or min_buyers < 2):
        raise ValueError('Invalid current buyer amount inputs')
    if history.empty:
        return raw.copy()
    required = {'snapshot_date', 'actual_spend_30d', 'raw_predicted_spend_30d'}
    if not required.issubset(history.columns):
        raise ValueError(f'Missing buyer calibration columns: {sorted(required - set(history.columns))}')
    dates = pd.to_datetime(history.snapshot_date, errors='raise')
    target_date = pd.Timestamp(target_date)
    if dates.isna().any() or (dates + pd.Timedelta(days=30) > target_date).any():
        raise ValueError('Buyer calibration includes an immature label')
    values = history[['actual_spend_30d', 'raw_predicted_spend_30d']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, 0] <= 0).any()
            or (values[:, 1] < 0).any()):
        raise ValueError('Buyer calibration needs positive actual buyer spend')
    allowed_dates = sorted(dates.unique())[-3:]
    recent = history.loc[dates.isin(allowed_dates)]
    if (len(recent) < min_buyers
            or recent.raw_predicted_spend_30d.nunique() < 2):
        return raw.copy()
    calibrator = IsotonicRegression(y_min=0, increasing=True,
                                    out_of_bounds='clip')
    calibrator.fit(recent.raw_predicted_spend_30d.to_numpy(dtype=float),
                   recent.actual_spend_30d.to_numpy(dtype=float))
    mapped = calibrator.predict(raw)
    if not np.isfinite(mapped).all() or (mapped < 0).any():
        raise ValueError('Invalid monotone buyer amount map')
    return .5 * raw + .5 * mapped


def segmented_buyer_amount_factors(history: pd.DataFrame,
                                   target_date: pd.Timestamp) -> tuple[float, float]:
    """Calibrate ordinary and high-amount cohorts with the same lagged rule."""
    if history.empty:
        return 1.0, 1.0
    if 'high_amount_segment' not in history.columns:
        raise ValueError('Missing historical amount segment')
    if not history.high_amount_segment.isin([True, False]).all():
        raise ValueError('Historical amount segment must be boolean')
    factors = []
    for high in (False, True):
        group = history.loc[history.high_amount_segment.eq(high)]
        factors.append(lagged_amount_factor(group, target_date))
    return float(factors[0]), float(factors[1])
