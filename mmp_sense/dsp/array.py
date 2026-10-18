"""Where a sound came from, given where the microphones are.

Positions are metres, in two or three dimensions; `speed` is the speed
of sound. A pair's delay `tau` is how much later the second microphone
hears a sound than the first (positive when later), as
`cross_correlation` measures it.

A far source is a direction: a unit vector `u` pointing at it, so a
microphone at `p` hears it `p . u / speed` earlier than one at the
origin. A near source is a point: each microphone hears it its distance
from it, over `speed`, after it sounded.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import optimize
from scipy import signal as _signal

__all__ = ["far_field", "near_field", "steering", "srp_map", "delay_and_sum",
           "fractional_taps", "FRACTION_S"]


def _unit(azimuth_deg, elevation_deg=0.0, dims=2) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return u[:dims]


def steering(points: np.ndarray, speed: float, azimuth_deg: float,
             elevation_deg: float = 0.0) -> np.ndarray:
    """When each microphone hears a far source in that direction, in
    seconds, relative to the origin: earlier (negative) nearer to it."""
    points = np.asarray(points, dtype=float)
    return -points @ _unit(azimuth_deg, elevation_deg, points.shape[1]) / speed


def far_field(points, pairs, delays, speed: float) -> dict:
    """The direction of a far source from pair delays: a least-squares
    unit vector, its azimuth (degrees counter-clockwise from +x), its
    elevation -- from the x-y plane in 3-D; from the microphones' plane,
    sign unknown, when they lie in one -- and the fit's residual (s).
    With every microphone on one line only the angle from that line is
    known: `axis_angle_deg`, and azimuth is NaN."""
    points = np.asarray(points, dtype=float)
    pairs, delays = np.asarray(pairs, dtype=int).reshape(-1, 2), np.asarray(delays, float)
    good = np.isfinite(delays)
    pairs, delays = pairs[good], delays[good]
    out = {"azimuth_deg": np.nan, "elevation_deg": np.nan, "axis_angle_deg": np.nan,
           "residual_s": np.nan, "pairs": int(len(delays))}
    if not len(delays):
        return out
    baselines = points[pairs[:, 1]] - points[pairs[:, 0]]
    # tau = -(p_j - p_i) . u / speed, for each pair: linear in u.
    target = -speed * delays
    rank = np.linalg.matrix_rank(baselines, tol=1e-9)
    if rank < 2:
        axis = baselines[np.argmax(np.linalg.norm(baselines, axis=1))]
        length = np.linalg.norm(axis)
        cosine = np.clip(np.mean(target / (baselines @ axis / length)), -1, 1)
        out["axis_angle_deg"] = float(np.degrees(np.arccos(cosine)))
        return out
    u, *_ = np.linalg.lstsq(baselines, target, rcond=None)
    norm = np.linalg.norm(u)
    if norm == 0:
        return out
    if rank == 3:
        u = u / norm
        out["elevation_deg"] = float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))
    else:
        # Microphones in one plane see a source off it as a shorter u in
        # the plane; how far off is told, not to which side.
        out["elevation_deg"] = float(np.degrees(np.arccos(np.clip(norm, 0, 1))))
        u = u / norm
    out["azimuth_deg"] = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    fitted = u if rank == 3 else u * min(norm, 1.0)
    out["residual_s"] = float(np.sqrt(np.mean((baselines @ fitted - target) ** 2)) / speed)
    out["vector"] = u
    return out


def near_field(points, pairs, delays, speed: float) -> dict:
    """The point a near source sounded at, from pair delays: the
    least-squares fit of the differences of its distances to each
    microphone, with the residual (s). Needs one microphone more than
    the dimensions."""
    points = np.asarray(points, dtype=float)
    pairs, delays = np.asarray(pairs, dtype=int).reshape(-1, 2), np.asarray(delays, float)
    good = np.isfinite(delays)
    pairs, delays = pairs[good], delays[good]
    dims = points.shape[1]
    used = np.unique(pairs) if len(pairs) else []
    out = {"position": np.full(dims, np.nan), "residual_s": np.nan, "pairs": int(len(delays))}
    if len(used) < dims + 1:
        return out

    def misfit(s):
        distance = np.linalg.norm(points - s, axis=1)
        return (distance[pairs[:, 1]] - distance[pairs[:, 0]]) / speed - delays

    centre = points.mean(axis=0)
    spread = max(np.ptp(points, axis=0).max(), 1e-3)
    starts = [centre] + [centre + spread * _unit(a, 0.0, dims) for a in range(0, 360, 45)]
    best = None
    for start in starts:
        found = optimize.least_squares(misfit, start, x_scale=spread)
        if best is None or found.cost < best.cost:
            best = found
    out["position"] = best.x
    out["residual_s"] = float(np.sqrt(np.mean(best.fun ** 2)))
    return out


def srp_map(values, rate: float, points, speed: float, azimuths_deg,
            elevation_deg: float = 0.0) -> np.ndarray:
    """Steered response power (SRP-PHAT) for each azimuth: the sum over
    pairs of their GCC-PHAT at the delay that direction would give,
    over the number of pairs -- 1 where every pair agrees."""
    from .correlation import cross_correlation

    values = np.asarray(values, dtype=float)
    points = np.asarray(points, dtype=float)
    n = values.shape[1]
    reach = np.max(np.linalg.norm(points[:, None] - points[None], axis=2)) / speed
    lags, maps = None, []
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        lags, correlation = cross_correlation(values[:, i], values[:, j], rate,
                                              max_lag=reach + 2.0 / rate)
        maps.append(correlation)
    power = np.zeros(len(azimuths_deg))
    for k, azimuth in enumerate(azimuths_deg):
        arrival = steering(points, speed, azimuth, elevation_deg)
        for (i, j), correlation in zip(pairs, maps):
            power[k] += np.interp(arrival[j] - arrival[i], lags, correlation)
    return power / max(1, len(pairs))


#: Half the fractional-delay filter, in seconds: what it adds to a beam's
#: lag. Whole samples of it at the rate -- exact at multiples of 2 kHz,
#: within half a sample otherwise -- and at least 4, which is why a beam
#: wants 8 kHz or more.
FRACTION_S = 0.0005


def _half(rate: float) -> int:
    return max(4, int(round(FRACTION_S * rate)))


@lru_cache(maxsize=256)
def _fraction(delay_samples: float, half: int) -> np.ndarray:
    """An FIR delaying by `delay_samples` (whole and part) plus `half`
    samples: a windowed sinc of 2 * half + 1 taps."""
    whole = int(np.floor(delay_samples))
    part = delay_samples - whole
    length = 2 * half + 1
    n = np.arange(length)
    taps = np.sinc(n - (half + part)) * _signal.get_window(("kaiser", 6.0), length,
                                                             fftbins=False)
    taps /= taps.sum()
    return np.concatenate((np.zeros(whole), taps))


def fractional_taps(delays_s, rate: float) -> list[np.ndarray]:
    """Per channel, an FIR delaying it by its delay and FRACTION_S more
    (rounded to a thousandth of a sample, so designs are shared)."""
    return [_fraction(round(float(d) * rate, 3), _half(rate)) for d in delays_s]


def delay_and_sum(values, rate: float, delays_s, state=None):
    """(the mean of the channels, each delayed by its delay, state).

    Every channel is also delayed by half the fractional-delay filter,
    FRACTION_S. A run starts at rest on its first
    row and again after a row holding a gap (NaN); `state` is the last
    rows of the last block.
    """
    values = np.asarray(values, dtype=float)
    taps = fractional_taps(delays_s, rate)
    length = max(len(t) for t in taps)
    out = np.full(len(values), np.nan)
    if not len(values):
        return out, state
    finite = np.isfinite(values).all(axis=1)
    edges = np.diff(np.concatenate(([0], finite.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        block = values[a:b]
        before = state if a == 0 and state is not None else np.repeat(
            block[:1], length - 1, axis=0)
        joined = np.concatenate((before, block))
        total = np.zeros(len(block))
        for channel, h in enumerate(taps):
            padded = np.concatenate((h, np.zeros(length - len(h))))
            total += _signal.fftconvolve(joined[:, channel], padded, mode="valid")
        out[a:b] = total / len(taps)
        state = joined[-(length - 1):] if length > 1 else joined[:0]
    return out, (state if finite[-1] else None)
