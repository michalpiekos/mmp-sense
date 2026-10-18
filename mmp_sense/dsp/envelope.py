"""A signal's envelope, causally: the size of its analytic signal.

The imaginary part comes from an FIR Hilbert transformer -- the ideal
one, 2/(pi n) at odd n, windowed to `taps` samples -- and the real part
is the signal delayed by half of that, so both describe the same
instant, (taps - 1) / 2 samples ago. Nothing is looked ahead: the value
at a sample depends on that sample and the ones before, so a block
carried on from the last gives what the whole signal would.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal as _signal

__all__ = ["hilbert_taps", "envelope", "envelope_delay"]


def hilbert_taps(rate: float, min_hz: float) -> int:
    """An odd length long enough to shift `min_hz` by a quarter turn:
    about four of its periods."""
    if rate <= 0:
        raise ValueError("an envelope needs a known sample rate")
    if not 0 < min_hz < rate / 4:
        raise ValueError(f"need 0 < min_hz < {rate / 4:g} Hz (a quarter of the rate), "
                         f"got {min_hz:g}")
    return 2 * int(round(2.0 * rate / min_hz)) + 1


def envelope_delay(rate: float, min_hz: float) -> float:
    """Seconds the envelope lags the signal: half the transformer."""
    return (hilbert_taps(rate, min_hz) - 1) / 2 / rate


@lru_cache(maxsize=64)
def _transformer(taps: int) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) // 2
    h = np.zeros(taps)
    odd = n % 2 != 0
    h[odd] = 2.0 / (np.pi * n[odd])
    return h * _signal.get_window(("kaiser", 8.0), taps, fftbins=False)


def _run(segment: np.ndarray, before: np.ndarray, h: np.ndarray) -> np.ndarray:
    taps = len(h)
    joined = np.concatenate((before, segment))
    quadrature = _signal.fftconvolve(joined, h, mode="valid")
    middle = (taps - 1) // 2
    in_phase = joined[taps - 1 - middle:len(joined) - middle]
    return np.hypot(in_phase, quadrature)


def envelope(values, rate: float, min_hz: float = 20.0, state=None):
    """(the envelope along the first axis, state to carry on from).

    A run starts at rest on its first value, as if that value had held
    for ever, and again after a gap (NaN). `state` is the last inputs of
    the last block, per column, or None.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        columns = [envelope(values[:, i], rate, min_hz, None if state is None else state[i])
                   for i in range(values.shape[1])]
        out = np.stack([c[0] for c in columns], axis=1) if columns else values.copy()
        return out, [c[1] for c in columns]
    h = _transformer(hilbert_taps(rate, min_hz))
    out = np.full(len(values), np.nan)
    if not len(values):
        return out, state
    finite = np.isfinite(values)
    edges = np.diff(np.concatenate(([0], finite.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segment = values[a:b]
        if a == 0 and state is not None:
            before = state
        else:
            before = np.full(len(h) - 1, segment[0])
        out[a:b] = _run(segment, before, h)
        state = np.concatenate((before, segment))[-(len(h) - 1):]
    return out, (state if finite[-1] else None)
