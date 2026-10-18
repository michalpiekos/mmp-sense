"""Peaks of a spectrum, for a cursor to land on.

Each function takes a spectrum's bin frequencies and its values in dB.
A peak's frequency is refined between bins by a parabola through its
bin and the two beside it, which on a dB scale puts a tapered line's
centre within a small fraction of a bin.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

__all__ = ["nearest_bin", "refine_peak", "snap_to_peak", "step_bin", "next_peak",
           "PEAK_SPREADS"]

#: How far over its floor a bin must stand to be a peak `next_peak` stops
#: at, in the floor's own spreads: a single spectrum's noise spreads some
#: 5 dB, an averaged one's far less, and the bar follows. Floor and spread
#: are of the `FLOOR_HZ` around, the floor a median as `line_power` takes it.
PEAK_SPREADS = 3.0
FLOOR_HZ = 60.0
#: Of peaks closer than this, in bins, only the highest counts: a tapered
#: line's own side lobes are not lines of their own.
PEAK_SPACING_BINS = 8


def nearest_bin(frequency, hz: float) -> int:
    frequency = np.asarray(frequency, dtype=float)
    return int(np.argmin(np.abs(frequency - hz)))


def refine_peak(frequency, db, i: int) -> float:
    """Bin `i`'s frequency, moved to where a parabola through it and its
    neighbours peaks; the bin's own where it has no neighbours or is no peak."""
    frequency, db = np.asarray(frequency, dtype=float), np.asarray(db, dtype=float)
    hz = float(frequency[i])
    if not 0 < i < len(db) - 1 or not np.isfinite(db[i - 1:i + 2]).all():
        return hz
    a, b, c = db[i - 1], db[i], db[i + 1]
    curve = a - 2 * b + c
    if curve >= 0:
        return hz
    shift = float(np.clip(0.5 * (a - c) / curve, -0.5, 0.5))
    return hz + shift * float(frequency[i + 1] - frequency[i])


def snap_to_peak(frequency, db, hz: float, reach: int = 3) -> float:
    """The refined frequency of the highest bin within `reach` bins of `hz`."""
    db = np.asarray(db, dtype=float)
    if not len(db):
        return float(hz)
    at = nearest_bin(frequency, hz)
    lo, hi = max(0, at - reach), min(len(db), at + reach + 1)
    near = np.where(np.isfinite(db[lo:hi]), db[lo:hi], -np.inf)
    return refine_peak(frequency, db, lo + int(np.argmax(near)))


def step_bin(frequency, hz: float | None, direction: int) -> float:
    """The bin `direction` (+1 up, -1 down) from the one nearest `hz`;
    from below the lowest when `hz` is None."""
    frequency = np.asarray(frequency, dtype=float)
    if hz is None:
        return float(frequency[0])
    at = nearest_bin(frequency, hz) + (1 if direction > 0 else -1)
    return float(frequency[int(np.clip(at, 0, len(frequency) - 1))])


def next_peak(frequency, db, hz: float | None, direction: int,
              spreads: float = PEAK_SPREADS) -> float | None:
    """The refined frequency of the next peak above (`direction` +1) or
    below (-1) `hz` -- from 0 Hz when None -- or None if there is none.

    A peak stands `spreads` times the floor's spread over the floor, both
    taken over the `FLOOR_HZ` around it (the spread robustly: 1.4826
    median absolute deviations), and is the highest within
    `PEAK_SPACING_BINS`."""
    frequency, db = np.asarray(frequency, dtype=float), np.asarray(db, dtype=float)
    if len(db) < 3 or not np.isfinite(db).any():
        return None
    db = np.where(np.isfinite(db), db, np.nanmin(db[np.isfinite(db)]))
    step = float(frequency[1] - frequency[0])
    width = min(len(db), max(3, int(round(FLOOR_HZ / step))))
    over = db - median_filter(db, size=width, mode="reflect")
    spread = 1.4826 * median_filter(np.abs(over), size=width, mode="reflect")
    tops, _ = find_peaks(db, distance=PEAK_SPACING_BINS)
    tops = tops[over[tops] >= spreads * spread[tops]]
    if not len(tops):
        return None
    peaks = np.array([refine_peak(frequency, db, i) for i in tops])
    if hz is None:
        return float(peaks[0]) if direction > 0 else None
    # Half a bin: the peak the cursor sits on is not the next one.
    margin = 0.5 * step
    if direction > 0:
        ahead = peaks[peaks > hz + margin]
        return float(ahead[0]) if len(ahead) else None
    behind = peaks[peaks < hz - margin]
    return float(behind[-1]) if len(behind) else None
