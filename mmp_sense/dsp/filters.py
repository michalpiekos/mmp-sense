"""Time-domain operations that stay in the time domain."""

from __future__ import annotations

import numpy as np

__all__ = ["magnitude"]


def magnitude(values: np.ndarray) -> np.ndarray:
    """Euclidean norm across channels: sqrt(x^2 + y^2 + z^2).

    Takes (n_samples, n_channels) and returns (n_samples,). Works for any
    number of channels, not only three -- the maths does not care and
    neither should this.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.abs(values)
    return np.sqrt(np.sum(values**2, axis=1))
