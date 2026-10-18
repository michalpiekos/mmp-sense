"""Numbers that describe a spectrum: band power, centroid, flatness and kin.

Each function takes framed amplitude spectra as `framed_spectrum` makes
them -- (frames, bins), a sine of amplitude A reading A -- and gives one
value per frame. Power comes from the one-sided power spectral density,
so a sine of amplitude A carries A**2 / 2 however long the frame; the
taper's equivalent noise bandwidth (`enbw`, in bins) is what turns the
amplitude scaling into that. DC is left out of every total: a frame's
mean says nothing about what it oscillates at.

A frame with no power above DC has no centre, no shape and no share, and
reads NaN for those.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal
from scipy.ndimage import median_filter

__all__ = [
    "enbw",
    "power_bins",
    "band_power",
    "line_power",
    "spectral_centroid",
    "spectral_spread",
    "spectral_flatness",
    "spectral_rolloff",
    "spectral_flux",
]


def enbw(taper: str, width: int) -> float:
    """Equivalent noise bandwidth of a taper of `width` samples, in bins."""
    w = _signal.get_window(taper, int(width), fftbins=True)
    return float(len(w) * np.sum(w**2) / np.sum(w) ** 2)


def power_bins(amplitude: np.ndarray, bandwidth: float) -> np.ndarray:
    """Each bin's share of the mean-square value: amplitude**2 / 2 / ENBW."""
    return np.asarray(amplitude, dtype=float) ** 2 / (2.0 * bandwidth)


def _above_dc(frequency, amplitude, bandwidth):
    keep = np.asarray(frequency) > 0
    return np.asarray(frequency)[keep], power_bins(amplitude, bandwidth)[:, keep]


def _nan_where_silent(values, total):
    values = np.asarray(values, dtype=float)
    values[~(total > 0)] = np.nan
    return values


def band_power(frequency, amplitude, low_hz, high_hz, bandwidth, relative=False):
    """Mean-square content between `low_hz` and `high_hz` (inclusive), per frame.

    `relative` gives it as a fraction of all the power above DC.
    """
    if high_hz < low_hz:
        raise ValueError(f"band {low_hz}-{high_hz} Hz is upside down")
    f, power = _above_dc(frequency, amplitude, bandwidth)
    inside = (f >= low_hz) & (f <= high_hz)
    band = power[:, inside].sum(axis=1)
    if not relative:
        return band
    total = power.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nan_where_silent(band / total, total)


def line_power(frequency, amplitude, low_hz, high_hz, bandwidth, above=10.0,
               floor_hz=60.0, relative=False):
    """Power in the spectral lines between `low_hz` and `high_hz`, per frame.

    A line is a bin standing more than `above` times (10: 10 dB) over the
    floor around it -- the median power of the bins in the `floor_hz`
    centred on it -- and its power is what it has above that floor. An
    engine's harmonics are lines; wind, rain and hiss are floor however
    loud they are, and a bin of noise rises that far only by chance.

    `relative` gives instead the share of the band that is lines, 0..1:
    each bin is first measured against its own floor, so every part of
    the band counts alike however the floor slopes -- wind's falls
    steeply -- and the share is that of the lines' excess in the whole.
    Noise alone reads near 0, tones on a quiet floor near 1, and a
    silent band NaN.

    A frame holding a NaN reads NaN.
    """
    if high_hz < low_hz:
        raise ValueError(f"band {low_hz}-{high_hz} Hz is upside down")
    if not above > 1:
        raise ValueError(f"a line stands above its floor: need above > 1, got {above}")
    f, power = _above_dc(frequency, amplitude, bandwidth)
    out = np.full(power.shape[0], np.nan)
    step = float(f[1] - f[0]) if len(f) > 1 else np.inf
    width = max(3, int(round(min(floor_hz / step, len(f)))) | 1)
    # The floor of a bin near the band's edge reaches past the edge.
    near = (f >= low_hz - floor_hz) & (f <= high_hz + floor_hz)
    f, power = f[near], power[:, near]
    inside = (f >= low_hz) & (f <= high_hz)
    whole = np.isfinite(power).all(axis=1)
    if not whole.any() or not inside.any():
        out[whole] = 0.0
        return out
    good = power[whole]
    floor = median_filter(good, size=(1, width), mode="nearest")[:, inside]
    band = good[:, inside]
    if not relative:
        out[whole] = np.where(band > above * floor, band - floor, 0.0).sum(axis=1)
        return out
    ratio = band / np.maximum(floor, np.finfo(float).tiny)
    lines = np.where(ratio > above, ratio - 1.0, 0.0).sum(axis=1)
    total = band.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[whole] = _nan_where_silent(lines / ratio.sum(axis=1), total)
    return out


def spectral_centroid(frequency, amplitude, bandwidth):
    """The power-weighted mean frequency, per frame."""
    f, power = _above_dc(frequency, amplitude, bandwidth)
    total = power.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nan_where_silent(power @ f / total, total)


def spectral_spread(frequency, amplitude, bandwidth):
    """The power-weighted standard deviation of frequency about the centroid."""
    f, power = _above_dc(frequency, amplitude, bandwidth)
    total = power.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = power @ f / total
        variance = (power * (f[None, :] - centre[:, None]) ** 2).sum(axis=1) / total
        return _nan_where_silent(np.sqrt(variance), total)


def spectral_flatness(frequency, amplitude, bandwidth):
    """Geometric over arithmetic mean power: 1 for white noise, near 0 for a tone."""
    _f, power = _above_dc(frequency, amplitude, bandwidth)
    total = power.sum(axis=1)
    if power.shape[1] == 0:
        return np.full(power.shape[0], np.nan)
    floor = np.finfo(float).tiny
    with np.errstate(invalid="ignore", divide="ignore"):
        geometric = np.exp(np.log(np.maximum(power, floor)).mean(axis=1))
        return _nan_where_silent(geometric / power.mean(axis=1), total)


def spectral_rolloff(frequency, amplitude, bandwidth, fraction=0.85):
    """The lowest frequency with `fraction` of the power at or below it."""
    if not 0 < fraction <= 1:
        raise ValueError(f"need 0 < fraction <= 1, got {fraction}")
    f, power = _above_dc(frequency, amplitude, bandwidth)
    total = power.sum(axis=1)
    if power.shape[1] == 0:
        return np.full(power.shape[0], np.nan)
    reached = np.cumsum(power, axis=1) >= fraction * total[:, None] * (1 - 1e-12)
    out = f[np.argmax(reached, axis=1)].astype(float)
    return _nan_where_silent(out, total)


def spectral_flux(frequency, amplitude, bandwidth=1.0, normalize=False):
    """How much each frame's spectrum grew since the previous frame.

    By default the onset flux: the length of the increase, bin by bin,
    ignoring bins that fell, divided by the taper's spread (`bandwidth`,
    its ENBW in bins) so that a sine of amplitude A arriving reads A --
    near zero on a steady background, large where new energy arrives, as
    at a knock. It is in the signal's units and grows with loudness.

    `normalize` scales each spectrum to unit length and takes the whole
    change, so only a change of shape counts: 0 for the same sound louder
    or softer, sqrt(2) at most. Noise reshuffles its spectrum every frame,
    so on a noisy background this one is never near zero.

    The first frame has nothing before it and reads NaN; with `normalize`
    so do a silent frame and the one after it.
    """
    keep = np.asarray(frequency) > 0
    spectra = np.asarray(amplitude, dtype=float)[:, keep]
    out = np.full(len(spectra), np.nan)
    if len(spectra) < 2:
        return out
    if not normalize:
        rise = np.maximum(spectra[1:] - spectra[:-1], 0.0)
        out[1:] = np.linalg.norm(rise, axis=1) / np.sqrt(bandwidth)
        return out
    norm = np.linalg.norm(spectra, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = spectra / norm[:, None]
    unit[~(norm > 0)] = np.nan
    out[1:] = np.linalg.norm(unit[1:] - unit[:-1], axis=1)
    return out
