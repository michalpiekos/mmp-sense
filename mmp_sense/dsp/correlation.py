"""How two signals relate in time: cross-correlation, delay, coherence.

`cross_correlation(x, y)` peaks at lag L when y is x delayed by L:
y heard the same thing L seconds after x. GCC-PHAT ("phat") whitens
the cross-spectrum first, so the peak is sharp whatever the sound's
colour and stays put through echoes; "plain" is the normalised
cross-correlation, -1..1, which suits slow, narrow-band sensors.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal
from scipy.fft import irfft, next_fast_len, rfft

__all__ = ["cross_correlation", "peak_lag", "coherence", "robust_line", "METHODS",
           "check_method"]

METHODS = ("phat", "plain")


def _clean(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)
    if finite.any():
        values = np.where(finite, values - values[finite].mean(), 0.0)
    return values


def check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")


def cross_correlation(x, y, rate: float, *, max_lag: float | None = None,
                      method: str = "phat") -> tuple[np.ndarray, np.ndarray]:
    """(lags in seconds, correlation) of `y` against `x`, two equally long
    runs of samples at `rate`; lags from -max_lag to +max_lag (all
    possible, by default). Gaps (NaN) count as silence."""
    check_method(method)
    x, y = _clean(x), _clean(y)
    n = len(x)
    if len(y) != n:
        raise ValueError(f"{n} and {len(y)} samples: the two runs must be as long")
    if n < 2:
        raise ValueError("a correlation needs at least 2 samples")
    size = next_fast_len(2 * n)
    spectrum = np.conj(rfft(x, size)) * rfft(y, size)
    if method == "phat":
        magnitude = np.abs(spectrum)
        spectrum = np.where(magnitude > 1e-12 * (magnitude.max() or 1.0),
                            spectrum / np.maximum(magnitude, 1e-300), 0.0)
        scale = 1.0
    else:
        energy = np.sqrt(np.sum(x * x) * np.sum(y * y))
        scale = 1.0 / energy if energy > 0 else 0.0
    full = irfft(spectrum, size) * scale
    most = n - 1 if max_lag is None else min(n - 1, int(np.floor(abs(max_lag) * rate)))
    correlation = np.concatenate((full[size - most:], full[:most + 1]))
    lags = np.arange(-most, most + 1) / float(rate)
    return lags, correlation


def peak_lag(lags: np.ndarray, correlation: np.ndarray) -> tuple[float, float, float]:
    """(lag, peak, snr) of the highest point, the lag refined between
    samples by a parabola through it and its neighbours; snr is the peak
    over the correlation's rms away from it. NaNs where there is no
    positive peak standing above the rest -- silence has no delay."""
    correlation = np.asarray(correlation, dtype=float)
    if len(correlation) < 3 or not np.isfinite(correlation).any():
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(correlation))
    peak = float(correlation[i])
    step = float(lags[1] - lags[0])
    lag = float(lags[i])
    if 0 < i < len(correlation) - 1:
        a, b, c = correlation[i - 1], correlation[i], correlation[i + 1]
        curve = a - 2 * b + c
        if curve < 0:
            shift = 0.5 * (a - c) / curve
            lag += float(np.clip(shift, -0.5, 0.5)) * step
            peak = float(b - 0.25 * (a - c) * shift)
    away = np.ones(len(correlation), dtype=bool)
    away[max(0, i - 2):i + 3] = False
    rest = correlation[away]
    if not peak > 0 or not np.ptp(correlation[np.isfinite(correlation)]) > 0:
        return np.nan, np.nan, np.nan               # silence: nothing stands out
    if not rest.size:
        return lag, peak, np.nan                    # too few lags to tell the rest
    noise = float(np.sqrt(np.mean(rest ** 2)))
    return lag, peak, peak / noise if noise > 0 else np.inf


def coherence(x, y, rate: float, *, segment: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(frequency, magnitude-squared coherence 0..1) of two equally long
    runs, by Welch's method over segments of `segment` samples."""
    x, y = _clean(x), _clean(y)
    segment = int(min(segment, len(x)))
    if segment < 4:
        raise ValueError("coherence needs at least 4 samples")
    return _signal.coherence(x, y, fs=rate, nperseg=segment)


def robust_line(t, v, *, floor: float = 0.0, rounds: int = 5):
    """(slope, intercept at t=0, kept): a least-squares line through
    (t, v), refitted without points further from it than three times the
    residuals' median absolute deviation (and `floor`)."""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    kept = np.isfinite(t) & np.isfinite(v)
    slope, intercept = 0.0, float(np.median(v[kept])) if kept.any() else np.nan
    for _ in range(rounds):
        if kept.sum() >= 2:
            slope, intercept = np.polyfit(t[kept], v[kept], 1)
        elif kept.sum() == 1:
            slope, intercept = 0.0, float(v[kept][0])
        residual = v - (slope * t + intercept)
        if kept.sum() < 3:
            break
        spread = float(np.median(np.abs(residual[kept] - np.median(residual[kept]))))
        tiny = 1e-9 * max(1.0, float(np.abs(v[kept]).max()))   # an exact fit has no spread
        limit = max(3.0 * 1.4826 * spread, floor, tiny)
        again = np.isfinite(residual) & (np.abs(residual) <= limit)
        if np.array_equal(again, kept):
            break
        kept = again
    if kept.sum() >= 2:
        slope, intercept = np.polyfit(t[kept], v[kept], 1)
    return float(slope), float(intercept), kept
