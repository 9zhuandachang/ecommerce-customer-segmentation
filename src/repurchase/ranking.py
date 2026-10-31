"""Deterministic customer ranking shared by the interface and exports."""

import pandas as pd


def review_list(frame: pd.DataFrame, segment: str, top_n: int) -> pd.DataFrame:
    """Apply the visible filter and deterministic rank used for CSV export."""
    if segment != 'All segments':
        frame = frame.loc[frame.segment_name.eq(segment)]
    if frame.empty:
        return frame.copy()
    if top_n < 1:
        raise ValueError('Top N must be positive')
    ranked = frame.sort_values(['repurchase_score', 'CustomerID'],
                               ascending=[False, True], kind='stable').head(top_n).copy()
    ranked = ranked.drop(columns='rank', errors='ignore')
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))
    return ranked
