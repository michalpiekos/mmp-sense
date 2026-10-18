"""Transforms into the frequency domain."""

from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = ["amplitude_spectrum", "spectrogram", "WINDOWS"]

#: Window functions offered by name. scipy knows many more; these are the
#: ones worth a menu entry, and an unknown name is an error rather than a
#: silent rectangle.
WINDOWS = ("hann", "hamming", "blackman", "boxcar")


def _window(name: str, length: int) -> np.ndarray:
    if name not in WINDOWS:
        raise ValueError(f"unknown window {name!r}; expected one of {', '.join(WINDOWS)}")
    return signal.get_window(name, length, fftbins=True)


def amplitude_spectrum(
    values: np.ndarray,
    rate: float,
    *,
    nfft: int | None = None,
    window: str = "hann",
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of each channel.

    Returns (frequencies, amplitudes), amplitudes shaped
    (n_bins, n_channels). Scaled so that a sinusoid of amplitude A reads A
    at its bin -- coherent gain of the window divided out, and the
    one-sided doubling applied to every bin but DC and Nyquist. That is
    what makes the number on the chart mean something in the signal's own
    units rather than in FFT counts.

    `nfft` defaults to the number of samples given. A larger value
    zero-pads, which interpolates the display without adding resolution.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 samples for a spectrum")

    if detrend:
        values = values - values.mean(axis=0, keepdims=True)

    taper = _window(window, n)[:, None]
    tapered = values * taper

    size = int(nfft or n)
    if size < n:
        raise ValueError(f"nfft={size} is smaller than the {n} samples given")

    spectrum = np.fft.rfft(tapered, n=size, axis=0)
    frequency = np.fft.rfftfreq(size, d=1.0 / rate)

    # Coherent gain: a hann window halves a sinusoid's amplitude, and
    # dividing by the window's sum rather than by n restores it.
    amplitude = np.abs(spectrum) / taper.sum()
    amplitude[1:] *= 2.0
    if size % 2 == 0 and len(amplitude) > 1:
        amplitude[-1] /= 2.0  # Nyquist bin is not doubled
    return frequency, amplitude


def spectrogram(
    values: np.ndarray,
    rate: float,
    *,
    nperseg: int = 256,
    overlap: float = 0.5,
    nfft: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude spectrogram of one channel.

    Returns (frequencies, times, values) with values shaped
    (n_frequencies, n_times) -- scipy's orientation, and the one
    TimeFrequency stores. `overlap` is a fraction of the segment, which is
    how people think about it, rather than scipy's sample count.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("spectrogram takes a single channel")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    nperseg = int(min(nperseg, len(values)))
    if nperseg < 2:
        raise ValueError("need at least 2 samples for a spectrogram")

    frequency, time, magnitude = signal.spectrogram(
        values,
        fs=rate,
        window=_window(window, nperseg),
        nperseg=nperseg,
        noverlap=int(nperseg * overlap),
        nfft=int(nfft) if nfft else None,
        mode="magnitude",
        scaling="spectrum",
    )
    return frequency, time, magnitude
