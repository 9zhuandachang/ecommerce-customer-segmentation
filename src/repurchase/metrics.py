"""Monthly priority-list metrics shared by model comparisons."""

import numpy as np
import pandas as pd
from src.data.validation import KEYS


def top20_metrics(frame: pd.DataFrame, scores: np.ndarray, y: np.ndarray) -> dict:
    """Allocate ceil(20%) slots within each monthly cutoff, then pool hits."""
    ranked = frame[KEYS].copy()
    ranked['score'] = scores
    ranked['label'] = y
    ranked = ranked.sort_values(
        ['snapshot_date', 'score', 'customer_id'],
        ascending=[True, False, True], kind='stable',
    )
    position = ranked.groupby('snapshot_date', sort=False).cumcount() + 1
    quota = np.ceil(ranked.groupby('snapshot_date').customer_id.transform('size') * .2)
    chosen = ranked.loc[position <= quota]
    hits = int(chosen.label.sum())
    positives = int(np.sum(y))
    precision = hits / len(chosen)
    prevalence = positives / len(y)
    return {'top20_count': len(chosen), 'top20_hits': hits,
            'precision_at_20pct': precision,
            'recall_at_20pct': hits / positives if positives else np.nan,
            'lift_at_20pct': precision / prevalence if prevalence else np.nan}
