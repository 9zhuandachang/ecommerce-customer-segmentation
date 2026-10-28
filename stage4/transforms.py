"""Importable transforms used by saved stage-four pipelines."""

import numpy as np


def signed_log1p(values):
    """Compress magnitudes while preserving the direction of signed features."""
    return np.sign(values) * np.log1p(np.abs(values))
