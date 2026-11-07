"""The arithmetic behind the session's answers, kept apart from it.

`Session.values`, `stats`, `spectrum` and `spectrum_summary` are built
from these, and the panes use `spectrum_of` and `spectrogram_of` for the
datasets they draw. Nothing here keeps state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly, welch

from .. import dsp
from .data import Spectrogram, Spectrum

__all__ = ["finite", "stretch", "summarise", "distribution", "spectrum_of",
           "spectrogram_of", "shape_of", "spectrum_table", "as_audio", "fade_in", "FADE_S",
           "band_only", "BAND_EDGE",
           "paired", "SyncEstimate", "fit_sync", "NoOverlap", "psd_table", "at_times"]

#: How long the ends of a sound fade in and out, so a cut is not a click.
FADE_S = 0.005
#: A band's gain falls from 1 to 0 over this share of its width outside
#: it (at least one bin): a hard edge would ring.
BAND_EDGE = 0.02
#: Amplified sound never passes this.
LIMIT = 0.9
#: How far around a loud moment the limiter turns the gain down, seconds.
LIMITER_S = 0.05
#: What "auto" raises the louder moments to, and the most it will raise them.
AUTO_LEVEL_DB = -20.0
AMP_MAX_DB = 60.0
#: The louder moments: this percentile of the loudness of frames this long.
LEVEL_FRAME_S = 0.05
LEVEL_PERCENTILE = 95.0


def finite(value) -> float | None:
    """A JSON-safe float: NaN and infinity become None."""
    value = float(value)
    return value if math.isfinite(value) else None


def stretch(session, t0=None, t1=None) -> tuple[float, float]:
    """The stretch of time a question is about.

    Given `t0` and `t1`, that; with neither, the selection if there is
    one, else everything held; with one, from it to the end (or the
    start to it).
    """
    start, end = session.span
    if t0 is None and t1 is None:
        return session.selection if session.selection is not None else (start, end)
    return (start if t0 is None else float(t0), end if t1 is None else float(t1))


def summarise(values: np.ndarray) -> dict[str, Any]:
    """The numbers worth having about any stretch of one signal."""
    values = np.asarray(values, dtype=float)
    ok = values[np.isfinite(values)]
    if not ok.size:
        return {"n": int(values.size), "min": np.nan, "max": np.nan, "mean": np.nan,
                "rms": np.nan, "std": np.nan, "peak_to_peak": np.nan}
    return {
        "n": int(values.size),
        "min": float(ok.min()),
        "max": float(ok.max()),
        "mean": float(ok.mean()),
        "rms": float(np.sqrt(np.mean(ok**2))),
        "std": float(ok.std()),
        "peak_to_peak": float(ok.max() - ok.min()),
    }


def distribution(values: np.ndarray, *, bins: int = 64, low: float | None = None,
                 high: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The bin edges and the density of the values between `low` and
    `high` (their own least and greatest if not given): a histogram
    whose area is 1, or all zeros when no value falls there."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low = float(values.min()) if low is None and values.size else low
    high = float(values.max()) if high is None and values.size else high
    low = 0.0 if low is None else float(low)
    high = low if high is None else float(high)
    if high < low:
        low, high = high, low
    if high == low:
        low, high = low - 0.5, high + 0.5
    values = values[(values >= low) & (values <= high)]
    counts, edges = np.histogram(values, bins=int(bins), range=(low, high))
    if not values.size:
        return edges, np.zeros(len(counts))
    return edges, counts / (values.size * np.diff(edges))


def spectrum_table(values: np.ndarray, rate: float, *, taper: str = "hann",
                   pad_to: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(frequency, amplitude) of one stretch, its mean removed."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    if len(values) < 2 or not rate:
        return np.zeros(0), np.zeros(0)
    frequency, amplitude = dsp.amplitude_spectrum(values, rate, nfft=pad_to,
                                                  window=taper, detrend=True)
    return frequency, np.asarray(amplitude, dtype=float).reshape(len(frequency), -1)[:, 0]


def psd_table(values: np.ndarray, rate: float, *, segment: int = 1024, overlap: float = 0.5,
              taper: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """(frequency, density in units^2/Hz) of one stretch by Welch's method:
    segments from the runs between gaps (NaN), averaged by how many
    segments each run gives. Empty when there is too little."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4 or not rate:
        return np.zeros(0), np.zeros(0)
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    edges = np.diff(np.concatenate(([0], finite.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    longest = max(b - a for a, b in runs)
    segment = int(min(segment, longest))
    if segment < 4:
        return np.zeros(0), np.zeros(0)
    step = segment - int(segment * overlap)
    total, weight, frequency = None, 0, None
    for a, b in runs:
        if b - a < segment:
            continue
        count = 1 + (b - a - segment) // step
        frequency, density = welch(values[a:b], fs=rate, window=taper, nperseg=segment,
                                   noverlap=segment - step, detrend="constant",
                                   scaling="density")
        total = density * count if total is None else total + density * count
        weight += count
    return frequency, total / weight


def shape_of(frequency: np.ndarray, amplitude: np.ndarray, samples: int,
             taper: str = "hann", bands=()) -> dict[str, float]:
    """The spectrum in a few numbers: where it peaks, its shape, its bands."""
    out: dict[str, float] = {
        "samples": float(samples),
        "resolution_hz": float(frequency[1] - frequency[0]) if len(frequency) > 1 else np.nan,
    }
    if not len(frequency):
        return out | {k: np.nan for k in ("dominant_hz", "centroid_hz", "spread_hz",
                                          "flatness", "rolloff_hz")}
    frame = amplitude[None, :]
    bandwidth = dsp.enbw(taper, int(samples))
    ac = frequency > 0
    out["dominant_hz"] = float(frequency[ac][np.argmax(amplitude[ac])]) if ac.any() else np.nan
    out["centroid_hz"] = float(dsp.spectral_centroid(frequency, frame, bandwidth)[0])
    out["spread_hz"] = float(dsp.spectral_spread(frequency, frame, bandwidth)[0])
    out["flatness"] = float(dsp.spectral_flatness(frequency, frame, bandwidth)[0])
    out["rolloff_hz"] = float(dsp.spectral_rolloff(frequency, frame, bandwidth)[0])
    for low, high in bands:
        tag = f"{low:g}-{high:g} Hz"
        out[f"power {tag}"] = float(dsp.band_power(frequency, frame, float(low), float(high),
                                                   bandwidth)[0])
        out[f"share {tag}"] = float(dsp.band_power(frequency, frame, float(low), float(high),
                                                   bandwidth, relative=True)[0])
    return out


def spectrum_of(session, signal: str, t0: float, t1: float, *, taper: str = "hann",
                pad_to: int | None = None) -> Spectrum:
    """The Spectrum dataset of one signal over [t0, t1] -- what a pane draws."""
    window = session.window(t0, t1).home(signal)
    values = np.asarray(window[signal], dtype=float)
    frequency, amplitude = spectrum_table(values, window.rate, taper=taper, pad_to=pad_to)
    return Spectrum(
        name=f"{signal}.spectrum",
        frequency=frequency,
        values=amplitude[:, None] if len(frequency) else np.zeros((0, 1)),
        channels=[signal],
        units=session.signals.units_for([signal]),
        source_range=(float(t0), float(t1)),
        metadata={"taper": taper, "samples": len(values), "rate": window.rate},
    )


def spectrogram_of(session, signal: str, t0: float, t1: float, *, nperseg: int = 256,
                   overlap: float = 0.5, nfft: int | None = None,
                   taper: str = "hann") -> Spectrogram:
    """The Spectrogram dataset of one signal over [t0, t1]."""
    window = session.window(t0, t1).home(signal)
    values = np.asarray(window[signal], dtype=float)
    if len(values) < 4 or not window.rate:
        return Spectrogram(name=f"{signal}.spectrogram")
    frequency, times, magnitude = dsp.spectrogram(
        values, window.rate, nperseg=nperseg, overlap=overlap, nfft=nfft, window=taper
    )
    return Spectrogram(
        name=f"{signal}.spectrogram",
        time=times + window.t_start,
        frequency=frequency,
        values=magnitude,
        units=session.signals.units_for([signal]),
        metadata={"nperseg": nperseg, "overlap": overlap},
    )


def band_only(values, rate: float, bands) -> np.ndarray:
    """`values` (sampled at `rate`) with only the frequencies inside any
    of `bands`, (low, high) Hz pairs, left: a zero-phase mask over the
    whole stretch. A gap (NaN) is taken as zero and stays NaN."""
    values = np.asarray(values, dtype=float)
    gaps = ~np.isfinite(values)
    n = len(values)
    if not n:
        return values.copy()
    size = sfft.next_fast_len(n, real=True)
    frequency = sfft.rfftfreq(size, 1.0 / rate)
    step = frequency[1] if len(frequency) > 1 else float(rate)
    gain = np.zeros(len(frequency))
    for low, high in bands:
        edge = max(BAND_EDGE * (high - low), step)
        inside = (frequency >= low) & (frequency <= high)
        below = np.clip((low - frequency) / edge, 0.0, 1.0)
        above = np.clip((frequency - high) / edge, 0.0, 1.0)
        skirt = 0.5 * (1.0 + np.cos(np.pi * np.maximum(below, above)))
        gain = np.maximum(gain, np.where(inside, 1.0, skirt))
    spectrum = sfft.rfft(np.where(gaps, 0.0, values), size, workers=-1)
    out = sfft.irfft(spectrum * gain, size, workers=-1)[:n]
    out[gaps] = np.nan
    return out


def _loud_moments(sound: np.ndarray, rate: int) -> float:
    """The RMS of the louder moments: louder than a mean of mostly quiet,
    and unmoved by one knock as a peak would be."""
    frame = max(1, int(LEVEL_FRAME_S * rate))
    frames = sound[:len(sound) // frame * frame].reshape(-1, frame)
    power = np.mean(frames ** 2, axis=1) if len(frames) else np.mean(sound ** 2, keepdims=True)
    power = power[power > 0]
    return float(np.sqrt(np.percentile(power, LEVEL_PERCENTILE))) if len(power) else 0.0


def _limited(sound: np.ndarray, rate: int) -> np.ndarray:
    """`sound` turned down around whatever passes LIMIT.

    The smoothing is no wider than the peak search, so every sample's
    gain is low enough for the loudest sample near it.
    """
    width = max(1, int(LIMITER_S * rate)) | 1
    envelope = maximum_filter1d(np.abs(sound), width, mode="nearest")
    gain = np.minimum(1.0, LIMIT / np.maximum(envelope, LIMIT))
    return sound * uniform_filter1d(gain, width, mode="nearest")


def amplified(sound: np.ndarray, rate: int, amp) -> np.ndarray:
    """`sound` raised by `amp` dB, or for "auto" to AUTO_LEVEL_DB, never past LIMIT."""
    if amp == "auto":
        level = _loud_moments(sound, rate)
        if level == 0:
            return sound
        db = min(AUTO_LEVEL_DB - 20.0 * math.log10(level), AMP_MAX_DB)
        amp = max(db, 0.0)
    return _limited(sound * 10.0 ** (float(amp) / 20.0), rate)


def check_amp(amp) -> None:
    if amp is None or amp == "auto":
        return
    if isinstance(amp, bool) or not isinstance(amp, (int, float)) or not 0 <= amp <= AMP_MAX_DB:
        raise ValueError(f'amp is None, "auto", or 0 to {AMP_MAX_DB:g} dB, not {amp!r}')


def as_audio(values: np.ndarray, rate: float, out_rate: int, *,
             speed: float = 1.0, bands=None, amp=None) -> np.ndarray:
    """Samples of one signal at `rate` as sound at `out_rate`: float32
    within -1..1, `speed` times as fast (and as high).

    Integers are divided by their type's full scale; values already
    within -1..1 are kept; larger ones lose their mean and peak at 0.9.
    A gap (NaN) is silence. Given `bands`, (low, high) Hz pairs, only
    those frequencies are kept and the result peaks at 0.9, so a quiet
    band can be heard. Given `amp`, dB or "auto", the mean goes and the
    sound is raised instead, held under 0.9 by a limiter.
    """
    check_amp(amp)
    values = np.asarray(values)
    if rate <= 0:
        raise ValueError("playing a signal needs its sample rate")
    if speed <= 0:
        raise ValueError("speed must be positive")
    if values.dtype.kind in "iu":
        info = np.iinfo(values.dtype)
        middle = (int(info.max) + int(info.min) + 1) / 2.0
        sound = (values.astype(float) - middle) / (info.max - middle + 1)
    else:
        sound = np.asarray(values, dtype=float)
        finite = np.isfinite(sound)
        peak = float(np.abs(sound[finite]).max()) if finite.any() else 0.0
        if peak > 1.0:
            sound = sound - sound[finite].mean()
            peak = float(np.abs(sound[finite]).max())
            sound = sound * (0.9 / peak) if peak > 0 else sound
    if bands is not None:
        sound = band_only(np.where(np.isinf(sound), np.nan, sound), rate, bands)
        finite = np.isfinite(sound)
        peak = float(np.abs(sound[finite]).max()) if finite.any() else 0.0
        if amp is None:
            sound = sound * (0.9 / peak) if peak > 0 else sound
    if amp is not None:
        finite = np.isfinite(sound)
        sound = sound - sound[finite].mean() if finite.any() else sound
    sound = np.nan_to_num(sound, nan=0.0, posinf=0.0, neginf=0.0)
    if not len(sound):
        return np.zeros(0, dtype=np.float32)
    ratio = Fraction(float(out_rate) / (float(rate) * float(speed))).limit_denominator(1000)
    if ratio != 1:
        sound = resample_poly(sound, ratio.numerator, ratio.denominator)
    if amp is not None:
        sound = amplified(sound, out_rate, amp)
    sound = np.clip(sound, -1.0, 1.0)
    fade = min(len(sound) // 2, int(FADE_S * out_rate))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        sound[:fade] *= ramp
        sound[len(sound) - fade:] *= ramp[::-1]
    return sound.astype(np.float32)


def fade_in(sound: np.ndarray, rate: int) -> np.ndarray:
    """A copy of `sound` that starts without a click, for sound begun partway."""
    out = sound.copy()
    fade = min(len(out), int(FADE_S * rate))
    out[:fade] *= np.linspace(0.0, 1.0, fade, endpoint=False, dtype=out.dtype)
    return out


def at_times(y: pd.Series, time: np.ndarray) -> np.ndarray:
    """`y` at each of `time`, by straight lines between its samples; NaN
    outside them and next to a gap."""
    ty, vy = y.index.to_numpy(float), y.to_numpy(float)
    if len(ty) < 2:
        return np.full(len(time), np.nan)
    out = np.interp(time, ty, np.nan_to_num(vy), left=np.nan, right=np.nan)
    after = np.clip(np.searchsorted(ty, time, side="left"), 0, len(ty) - 1)
    before = np.clip(np.searchsorted(ty, time, side="right") - 1, 0, len(ty) - 1)
    out[~(np.isfinite(vy[after]) & np.isfinite(vy[before]))] = np.nan
    return out


class NoOverlap(ValueError):
    """Two signals with too little time in common to compare."""


def paired(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time, x, y): `y` at each of `x`'s times, by straight lines between
    its samples, over the stretch where both have values."""
    tx, vx = x.index.to_numpy(float), x.to_numpy(float)
    ty, vy = y.index.to_numpy(float), y.to_numpy(float)
    good = np.isfinite(vy)
    ty, vy = ty[good], vy[good]
    if len(ty) < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    keep = (tx >= ty[0]) & (tx <= ty[-1])
    tx, vx = tx[keep], vx[keep]
    return tx, vx, np.interp(tx, ty, vy)


@dataclass
class SyncEstimate:
    """How late one stream's clock is against another's: `offset` seconds
    at `at`, and `drift` (a fraction; ppm is drift * 1e6) more per second."""

    offset: float
    drift: float
    at: float
    residual: float
    used: int
    windows: pd.DataFrame = field(repr=False)
    #: Of the windows used, how many share no time with one another.
    apart: int = 0

    @property
    def drift_ppm(self) -> float:
        return self.drift * 1e6

    def describe(self) -> dict[str, Any]:
        return {"offset_s": self.offset, "drift_ppm": self.drift_ppm, "at": self.at,
                "residual_s": self.residual, "windows_used": self.used,
                "windows_apart": self.apart,
                "windows": len(self.windows)}

    def __repr__(self) -> str:
        return (f"<SyncEstimate offset {self.offset * 1e3:.4f} ms at {self.at:.6g} s, "
                f"drift {self.drift_ppm:+.2f} ppm, residual {self.residual * 1e6:.1f} us, "
                f"{self.used} of {len(self.windows)} windows>")


def _apart(times: np.ndarray, width: float) -> int:
    count, last = 0, -np.inf
    for t in np.sort(times):
        if t - last >= width - 1e-12:
            count, last = count + 1, t
    return count


def fit_sync(windows: pd.DataFrame, *, min_snr: float, floor: float,
             width: float = 0.0) -> SyncEstimate:
    """The offset and drift through the windows' delays, against time:
    weak windows (snr below `min_snr`) and outliers left out."""
    strong = windows["snr"].to_numpy(float) >= min_snr
    t = windows["time"].to_numpy(float)
    v = np.where(strong, windows["delay"].to_numpy(float), np.nan)
    at = float(np.nanmean(t[strong])) if strong.any() else float(np.nanmean(t)) if len(t) else 0.0
    slope, intercept, kept = dsp.robust_line(t - at, v, floor=floor)
    residual = v[kept] - (slope * (t[kept] - at) + intercept)
    windows = windows.assign(used=kept)
    return SyncEstimate(offset=intercept, drift=slope, at=at,
                        residual=float(np.sqrt(np.mean(residual ** 2))) if kept.any() else np.nan,
                        used=int(kept.sum()), windows=windows,
                        apart=_apart(t[kept], width))


def frame(time: np.ndarray, values: np.ndarray, *, name: str,
          coords: dict | None = None) -> pd.Series | pd.DataFrame:
    """Values against time as pandas: a Series, or a DataFrame whose
    columns are the components (or, for frames, the frequencies)."""
    index = pd.Index(np.asarray(time, dtype=float), name="time")
    values = np.asarray(values)
    if values.ndim == 1:
        return pd.Series(values, index=index, name=name)
    coords = coords or {}
    if "frequency" in coords:
        columns = pd.Index(np.asarray(coords["frequency"], dtype=float), name="frequency_hz")
    elif "channel" in coords:
        columns = pd.Index(list(coords["channel"]), name=name)
    else:
        columns = pd.RangeIndex(values.shape[1], name=name)
    return pd.DataFrame(values.reshape(len(index), -1), index=index, columns=columns)
