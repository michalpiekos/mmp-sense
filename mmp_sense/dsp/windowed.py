"""Causal measurements over trailing windows, and first-order filters.

Every function here is causal: the value at sample `i` depends on samples
`i - width + 1 .. i` and nothing later, so what it says about a recording
is what it would have said live. A window that does not yet hold `width`
samples has no honest value and reads NaN rather than something smaller.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

__all__ = [
    "highpass1",
    "highpass1_alpha",
    "lowpass1",
    "lowpass1_alpha",
    "trailing_sum",
    "trailing_max",
    "trailing_min",
    "zero_crossing_rate",
    "frame_starts",
    "framed_spectrum",
    "dominant_frequency",
]


def highpass1_alpha(rate: float, cutoff_hz: float) -> float:
    """The coefficient of `highpass1` for a cutoff: RC / (RC + dt)."""
    if rate <= 0:
        raise ValueError("a high-pass needs a known sample rate")
    if not 0 < cutoff_hz < rate / 2.0:
        raise ValueError(f"need 0 < cutoff < {rate / 2.0} (Nyquist), got {cutoff_hz}")
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / rate
    return rc / (rc + dt)


def highpass1(
    values: np.ndarray,
    rate: float,
    cutoff_hz: float,
    state: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """First-order IIR high-pass: y[n] = a * (y[n-1] + x[n] - x[n-1]).

    Filters along the first axis. `state` is (x[n-1], y[n-1]) from the
    previous block; the new state is returned, so a stream can be filtered
    in pieces and give the same answer as all at once. Without a state the
    filter starts at rest on the first sample (x[-1] = x[0], y[-1] = 0),
    so a constant offset such as gravity does not ring at the start.

    A NaN is a gap: it stays NaN and the filter starts at rest again on
    the next sample, rather than carrying NaN on for ever. A block that
    ends in a gap returns a NaN state, which starts the next block at rest.
    """
    alpha = highpass1_alpha(rate, cutoff_hz)
    return _first_order(values, [alpha, -alpha], -alpha,
                        lambda segment: (segment[0], 0.0), state)


def lowpass1_alpha(rate: float, cutoff_hz: float) -> float:
    """The coefficient of `lowpass1` for a cutoff: dt / (RC + dt)."""
    if rate <= 0:
        raise ValueError("a low-pass needs a known sample rate")
    if not 0 < cutoff_hz < rate / 2.0:
        raise ValueError(f"need 0 < cutoff < {rate / 2.0} (Nyquist), got {cutoff_hz}")
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / rate
    return dt / (rc + dt)


def lowpass1(
    values: np.ndarray,
    rate: float,
    cutoff_hz: float,
    state: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """First-order IIR low-pass: y[n] = y[n-1] + b * (x[n] - y[n-1]).

    The counterpart of `highpass1`, with the same state, blocks and gaps.
    At rest means y[-1] = x[0]: a constant such as gravity passes through
    from the first sample instead of rising from zero.
    """
    b = lowpass1_alpha(rate, cutoff_hz)
    return _first_order(values, [b, 0.0], -(1.0 - b),
                        lambda segment: (segment[0], segment[0]), state)


def _first_order(values, numerator, a1, at_rest, state):
    """y[n] = b0 x[n] + b1 x[n-1] - a1 y[n-1] along the first axis, by gap-free stretch."""
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        columns = [_first_order(values[:, i], numerator, a1, at_rest,
                                None if state is None else (state[0][i], state[1][i]))
                   for i in range(values.shape[1])]
        out = np.stack([c[0] for c in columns], axis=1) if columns else values.copy()
        return out, (np.array([c[1][0] for c in columns]), np.array([c[1][1] for c in columns]))

    out = np.full(len(values), np.nan)
    if len(values) == 0:
        return out, state if state is not None else (np.nan, np.nan)
    finite = np.isfinite(values)
    edges = np.diff(np.concatenate(([0], finite.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segment = values[a:b]
        resume = a == 0 and state is not None and np.isfinite(state[0]) and np.isfinite(state[1])
        x_prev, y_prev = (float(state[0]), float(state[1])) if resume else at_rest(segment)
        zi = [numerator[1] * x_prev - a1 * y_prev]
        out[a:b], _ = _signal.lfilter(numerator, [1.0, a1], segment, zi=zi)
    return out, (float(values[-1]), float(out[-1]))


def _gaps(values: np.ndarray, width: int) -> np.ndarray:
    """Whether each trailing window of `width` holds a NaN or is not full."""
    bad = np.cumsum(~np.isfinite(values), axis=0)
    counts = bad.copy()
    counts[width:] = bad[width:] - bad[:-width]
    out = counts > 0
    out[: min(width - 1, len(out))] = True
    return out


def trailing_sum(values: np.ndarray, width: int) -> np.ndarray:
    """Sum of the last `width` samples at each sample, along the first axis.

    A window that is not yet full, or holds a NaN, reads NaN -- and a NaN
    goes no further than the windows that contain it.
    """
    values = np.asarray(values, dtype=float)
    width = max(1, int(width))
    total = np.cumsum(np.where(np.isfinite(values), values, 0.0), axis=0)
    out = total.copy()
    out[width:] = total[width:] - total[:-width]
    out[_gaps(values, width)] = np.nan
    return out


def trailing_max(values: np.ndarray, width: int) -> np.ndarray:
    """Largest of the last `width` samples at each sample; NaN as
    `trailing_sum` has it."""
    values = np.asarray(values, dtype=float)
    width = max(1, int(width))
    if len(values) == 0:
        return values.copy()
    # A centred filter moved right by half its width looks only backwards.
    out = maximum_filter1d(np.where(np.isfinite(values), values, -np.inf), size=width,
                           axis=0, mode="nearest", origin=(width - 1) // 2)
    out[_gaps(values, width)] = np.nan
    return out


def trailing_min(values: np.ndarray, width: int) -> np.ndarray:
    """Smallest of the last `width` samples at each sample, along the
    first axis -- a row of a spectrum per sample is taken bin by bin; NaN
    as `trailing_sum` has it."""
    values = np.asarray(values, dtype=float)
    width = max(1, int(width))
    if len(values) == 0:
        return values.copy()
    out = minimum_filter1d(np.where(np.isfinite(values), values, np.inf), size=width,
                           axis=0, mode="nearest", origin=(width - 1) // 2)
    gaps = _gaps(values, width)
    if out.ndim > 1 and gaps.ndim > 1:
        gaps = gaps.any(axis=tuple(range(1, gaps.ndim)))
    out[gaps] = np.nan
    return out


def zero_crossing_rate(values: np.ndarray, rate: float, width: int) -> np.ndarray:
    """Sign changes per second over the last `width` samples.

    A crossing between samples i-1 and i is counted at i, so a window of
    `width` samples sees `width - 1` pairs.
    """
    values = np.asarray(values, dtype=float)
    width = max(2, int(width))
    signs = np.signbit(values)
    crossed = np.zeros(values.shape, dtype=float)
    crossed[1:] = signs[1:] != signs[:-1]
    crossed[~np.isfinite(values)] = np.nan
    crossed[1:][~np.isfinite(values[:-1])] = np.nan
    counts = trailing_sum(crossed, width - 1)
    out = counts * rate / (width - 1)
    out[: min(width - 1, len(out))] = np.nan
    return out


def frame_starts(n: int, width: int, hop: int, first_end: int = 0) -> np.ndarray:
    """Start indices of complete frames whose last sample is `first_end`
    plus a multiple of `hop` (or later, if that one would not fit)."""
    width, hop = max(1, int(width)), max(1, int(hop))
    first = width - 1 + ((int(first_end) - (width - 1)) % hop)
    ends = np.arange(first, n, hop)
    return ends - width + 1


def framed_spectrum(
    values: np.ndarray,
    rate: float,
    starts: np.ndarray,
    width: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of each frame `values[s : s + width]`.

    Returns (frequencies, amplitudes) with amplitudes shaped
    (n_frames, n_bins), scaled as `amplitude_spectrum` scales them: a
    sinusoid of amplitude A reads A. Each frame has its mean removed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("a spectrum is taken of one channel")
    width = int(width)
    if width < 2:
        raise ValueError("a spectrum needs frames of at least 2 samples")
    frequency = np.fft.rfftfreq(width, d=1.0 / rate)
    if len(starts) == 0:
        return frequency, np.zeros((0, len(frequency)))
    frames = np.lib.stride_tricks.sliding_window_view(values, width)[starts]
    frames = frames - frames.mean(axis=1, keepdims=True)
    taper = _signal.get_window(window, width, fftbins=True)
    amplitude = np.abs(np.fft.rfft(frames * taper, axis=1)) / taper.sum()
    amplitude[:, 1:] *= 2.0
    if width % 2 == 0:
        amplitude[:, -1] /= 2.0
    return frequency, amplitude


def dominant_frequency(
    frequency: np.ndarray, amplitude: np.ndarray, min_hz: float = 0.0
) -> np.ndarray:
    """The frequency of the largest bin of each spectrum, above `min_hz`.

    DC is never the answer: a spectrum's mean says nothing about what it
    oscillates at. A frame with no energy above `min_hz` reads NaN.
    """
    amplitude = np.asarray(amplitude, dtype=float)
    keep = (frequency > 0) & (frequency >= min_hz)
    if not keep.any() or amplitude.shape[0] == 0:
        return np.full(amplitude.shape[0], np.nan)
    bands = amplitude[:, keep]
    out = frequency[keep][np.argmax(bands, axis=1)].astype(float)
    out[~(bands.max(axis=1) > 0)] = np.nan
    return out
