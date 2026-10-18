"""Butterworth filters and a notch, run causally in blocks.

A filter is designed as second-order sections, which stay accurate at
high orders and low cutoffs where a single polynomial would not. It
starts at rest on its first value -- in the steady state that value
would have left it in -- so a constant such as gravity passes a
low-pass without a step and leaves a high-pass at zero, as the
first-order filters do. A NaN is a gap: it stays NaN and the filter
starts at rest again after it.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from scipy import signal as _signal

__all__ = ["KINDS", "design", "sos_filter", "group_delay", "response"]

KINDS = ("lowpass", "highpass", "bandpass", "bandstop", "notch")


def _check(rate: float, *frequencies: float) -> None:
    if rate <= 0:
        raise ValueError("a filter needs a known sample rate")
    nyquist = rate / 2.0
    for f in frequencies:
        if not 0 < f < nyquist:
            raise ValueError(f"need 0 < frequency < {nyquist:g} Hz (Nyquist), got {f:g}")


@lru_cache(maxsize=256)
def _design(kind: str, rate: float, low: float, high: float, order: int, q: float):
    if kind == "notch":
        _check(rate, low)
        b, a = _signal.iirnotch(low, q, fs=rate)
        return _signal.tf2sos(b, a)
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    if kind in ("lowpass", "highpass"):
        _check(rate, low)
        return _signal.butter(order, low, btype=kind, fs=rate, output="sos")
    _check(rate, low, high)
    if not low < high:
        raise ValueError(f"need low < high, got {low:g} and {high:g} Hz")
    return _signal.butter(order, (low, high), btype=kind, fs=rate, output="sos")


def design(kind: str, rate: float, *, cutoff_hz: float | None = None,
           low_hz: float | None = None, high_hz: float | None = None,
           freq_hz: float | None = None, order: int = 4, q: float = 30.0) -> np.ndarray:
    """Second-order sections of a Butterworth `lowpass`/`highpass` at
    `cutoff_hz`, `bandpass`/`bandstop` from `low_hz` to `high_hz`, or a
    `notch` at `freq_hz` of quality `q` (its width is freq/q)."""
    if kind not in KINDS:
        raise ValueError(f"unknown filter {kind!r}; expected one of {', '.join(KINDS)}")
    if kind in ("lowpass", "highpass"):
        low, high = cutoff_hz, 0.0
    elif kind == "notch":
        low, high = freq_hz, 0.0
    else:
        low, high = low_hz, high_hz
    if low is None or (kind in ("bandpass", "bandstop") and high is None):
        raise ValueError(f"a {kind} needs its frequencies")
    return _design(kind, float(rate), float(low), float(high), int(order), float(q))


def sos_filter(values, sos: np.ndarray, state=None):
    """(values filtered along the first axis, state to carry on from).

    `state` is what the last block returned: the sections' memory per
    column, or None for a column that ended in a gap or has not started.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        columns = [sos_filter(values[:, i], sos, None if state is None else state[i])
                   for i in range(values.shape[1])]
        out = np.stack([c[0] for c in columns], axis=1) if columns else values.copy()
        return out, [c[1] for c in columns]
    out = np.full(len(values), np.nan)
    if not len(values):
        return out, state
    finite = np.isfinite(values)
    edges = np.diff(np.concatenate(([0], finite.astype(np.int8), [0])))
    memory = None
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segment = values[a:b]
        if a == 0 and state is not None:
            zi = state
        else:
            zi = _signal.sosfilt_zi(sos) * segment[0]
        out[a:b], memory = _signal.sosfilt(sos, segment, zi=zi)
    return out, (memory if finite[-1] else None)


def _phase_slope(sos, rate: float, f: float) -> float:
    step = max(f * 1e-3, rate * 1e-7)
    lo = max(f - step, rate * 1e-9)
    w, h = _signal.sosfreqz(sos, worN=[lo, f + step], fs=rate)
    phase = np.unwrap(np.angle(h))
    return -(phase[1] - phase[0]) / (2 * np.pi * (w[1] - w[0]))


def group_delay(kind: str, **params) -> float:
    """Seconds a filter delays what it lets through: its group delay at
    the middle of its passband (near DC for a low-pass), designed at a
    rate well above its frequencies so the answer is the filter's, not
    the sampling's."""
    if kind == "notch":
        return 0.0
    top = max(v for k, v in params.items() if k.endswith("_hz") and v is not None)
    rate = 200.0 * top
    sos = design(kind, rate, **params)
    if kind == "lowpass":
        at = params["cutoff_hz"] / 100.0
    elif kind == "highpass":
        at = params["cutoff_hz"] * 4.0
    elif kind == "bandpass":
        at = float(np.sqrt(params["low_hz"] * params["high_hz"]))
    else:
        at = params["low_hz"] / 4.0
    return max(0.0, float(_phase_slope(sos, rate, at)))


def response(sos_list, rate: float, frequency: np.ndarray):
    """(gain, phase, group delay) of sections in a chain at `frequency`:
    the complex gain's magnitude, its angle (radians, unwrapped along
    `frequency`, so meaningful on a fine grid) and seconds of group delay."""
    frequency = np.asarray(frequency, dtype=float)
    h = np.ones(len(frequency), dtype=complex)
    delay = np.zeros(len(frequency))
    for sos in sos_list:
        h *= _signal.sosfreqz(sos, worN=frequency, fs=rate)[1]
        for section in np.atleast_2d(sos):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")      # singular at a notch's own frequency
                delay += _signal.group_delay((section[:3], section[3:]), w=frequency,
                                             fs=rate)[1] / rate
    return np.abs(h), np.unwrap(np.angle(h)), delay
