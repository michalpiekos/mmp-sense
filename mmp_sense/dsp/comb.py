"""Combs: lines spaced by a rate, the rate itself, and lines that drift.

A source that repeats `r` times a second makes lines at multiples of
`r`. These functions find `r` from the lines (an autocorrelation of the
spectrum's lines), follow one line precisely, list the lines of a
spectrum, divide out a rate that changes (alignment, order tracking),
cut a signal into its cycles to average them, and score how well
spacings and pairs of rates explain a set of lines against chance.
Arrays in, arrays out.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, median_filter
from scipy.signal import butter, find_peaks, peak_prominences, sosfiltfilt

from .peaks import refine_peak

__all__ = ["over_floor_db", "repetition_rate", "line_peaks", "spectrum_lines",
           "align_frames", "order_resample", "cycle_count", "cut_cycles", "comb_counts",
           "place_lines", "fit_two_rates",
           "chance_fits", "FLOOR_HZ", "GUARD_DB", "PRESENT_DB"]

#: The floor a line stands over: the median of the bins in this many Hz
#: around it, as `line_power` takes it.
FLOOR_HZ = 60.0
#: How much higher, in dB, a rate's even multiples must stand than the
#: bins between them before it is doubled for want of odd ones.
GUARD_DB = 3.0
#: How much higher, in dB, the lines at a rate's multiples must stand,
#: on average, than the bins between them for the rate to be reported.
PRESENT_DB = 1.5
#: Of the autocorrelation's peaks at least this share of the highest, the
#: shortest lag is taken: a period's multiples peak nearly as high.
SHORTEST = 0.9
#: The most multiples of a rate the octave guard looks at.
GUARD_MULTIPLES = 200


def _floor_width(step: float, floor_hz: float, bins: int) -> int:
    return max(3, int(round(min(floor_hz / step, bins))) | 1)


def over_floor_db(frequency, power, floor_hz: float = FLOOR_HZ) -> np.ndarray:
    """Each bin's power over the median of the `floor_hz` around it, in
    dB; `power` is one spectrum or one per row."""
    frequency = np.asarray(frequency, dtype=float)
    power = np.atleast_2d(np.asarray(power, dtype=float))
    step = float(frequency[1] - frequency[0]) if len(frequency) > 1 else 1.0
    width = _floor_width(step, floor_hz, power.shape[1])
    clean = np.where(np.isfinite(power), power, 0.0)
    floor = median_filter(clean, size=(1, width), mode="nearest")
    tiny = np.finfo(float).tiny
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.maximum(clean, tiny) / np.maximum(floor, tiny))


def _comb_contrast(frequency, over, rate, low, high, offset):
    """Mean dB over the floor at (m + offset) * rate less that at the
    quarter points beside them, over the multiples within low..high."""
    step = float(frequency[1] - frequency[0])
    first = max(1, int(np.ceil(low / rate - offset)))
    m = np.arange(first, first + GUARD_MULTIPLES)
    m = m[(m + offset) * rate <= high]
    if not len(m):
        return np.nan

    def level(positions):
        at = np.searchsorted(frequency, positions)
        reach = 1 + np.ceil(0.002 * positions / step).astype(int)
        return np.array([over[max(0, a - r):min(len(over), a + r + 1)].max(initial=-np.inf)
                         for a, r in zip(at, reach)])

    on = level((m + offset) * rate)
    beside = np.concatenate([level((m + offset - 0.25) * rate),
                             level((m + offset + 0.25) * rate)])
    on, beside = on[np.isfinite(on)], beside[np.isfinite(beside)]
    if not len(on) or not len(beside):
        return np.nan
    return float(on.mean() - beside.mean())


def repetition_rate(frequency, power, min_hz: float, max_hz: float, *,
                    low_hz: float | None = None, high_hz: float | None = None,
                    floor_hz: float = FLOOR_HZ, guard_db: float = GUARD_DB,
                    present_db: float = PRESENT_DB) -> np.ndarray:
    """How often each spectrum's sound repeats, between `min_hz` and
    `max_hz`: one rate per row of `power` (a power spectrum per row),
    NaN where nothing repeats.

    The autocorrelation of the lines -- each bin's power over its floor,
    between `low_hz` (default `min_hz`) and `high_hz` (default 40 times
    `max_hz`) -- peaks at the period and its multiples; the shortest lag
    of those within 90 % of the highest peak, refined between lags, is
    the rate. A comb is also explained by half its rate and sometimes by
    double it, so the rate is then checked against the lines at its
    multiples, each measured over the bins between them and averaged:
    halved if the odd multiples of half of it stand `present_db` out,
    doubled if its own odd multiples do not while its even ones stand
    `guard_db` -- never out of `min_hz`..`max_hz`. A rate whose own
    multiples do not stand `present_db` out is no rate: NaN.
    """
    if not 0 < min_hz < max_hz:
        raise ValueError(f"need 0 < min_hz < max_hz, got {min_hz} and {max_hz}")
    frequency = np.asarray(frequency, dtype=float)
    power = np.atleast_2d(np.asarray(power, dtype=float))
    out = np.full(power.shape[0], np.nan)
    if len(frequency) < 3:
        return out
    low = float(min_hz if low_hz is None else low_hz)
    high = float(min(frequency[-1], 40.0 * max_hz if high_hz is None else high_hz))
    band = (frequency >= low) & (frequency <= high)
    if band.sum() < 3:
        return out
    f = frequency[band]
    # Lags fine enough that the highest line turns at most 0.3 rad a step.
    step = 0.3 / (2 * np.pi * f[-1])
    lags = np.arange(1.0 / max_hz - 2 * step, 1.0 / min_hz + 2 * step, step)
    cosines = np.cos(2 * np.pi * np.outer(f, lags))
    for row, spectrum in enumerate(power):
        if not np.isfinite(spectrum).all() or not (spectrum > 0).any():
            continue
        over = over_floor_db(frequency, spectrum, floor_hz)[0]
        excess = np.maximum(10.0 ** (over[band] / 10.0) - 1.0, 0.0)
        excess = excess - excess.mean()
        acf = excess @ cosines
        inside = np.flatnonzero((lags >= 1.0 / max_hz) & (lags <= 1.0 / min_hz))
        top = float(acf[inside].max())
        if top <= 0:
            continue
        # A period repeats at its multiples too, nearly as high: of the
        # peaks near the highest, the shortest is the period.
        peaks = inside[(acf[inside] >= SHORTEST * top)
                       & (acf[inside] >= acf[np.maximum(inside - 1, 0)])
                       & (acf[inside] >= acf[np.minimum(inside + 1, len(lags) - 1)])]
        if not len(peaks):
            continue                  # rising to an edge: the period lies outside
        best = int(peaks[0])
        a, b, c = acf[best - 1:best + 2]
        curve = a - 2 * b + c
        shift = float(np.clip(0.5 * (a - c) / curve, -0.5, 0.5)) if curve < 0 else 0.0
        rate = 1.0 / (lags[best] + shift * step)
        # The octave guard, on the spectrum itself.
        if (rate / 2.0 >= min_hz
                and _comb_contrast(frequency, over, rate, low, high, 0.5) >= present_db):
            rate /= 2.0
        elif (2.0 * rate <= max_hz
              and _comb_contrast(frequency, over, 2 * rate, low, high, 0.5) < present_db
              and _comb_contrast(frequency, over, 2 * rate, low, high, 0.0) >= guard_db):
            rate *= 2.0
        if _comb_contrast(frequency, over, rate, low, high, 0.0) >= present_db:
            out[row] = rate
    return out


def line_peaks(frequency, amplitude, low_hz: float, high_hz: float, *,
               near=None, within: float | None = None) -> np.ndarray:
    """The precise frequency of the strongest bin between `low_hz` and
    `high_hz` in each row of `amplitude` -- refined between bins by a
    parabola in dB. With `near` (a number, or one per row), only within
    `within` Hz of it (two bins if not given); NaN where nothing is there."""
    frequency = np.asarray(frequency, dtype=float)
    amplitude = np.atleast_2d(np.asarray(amplitude, dtype=float))
    rows = amplitude.shape[0]
    out = np.full(rows, np.nan)
    if len(frequency) < 3:
        return out
    step = float(frequency[1] - frequency[0])
    if near is not None:
        near = np.broadcast_to(np.asarray(near, dtype=float), (rows,))
        within = 2 * step if within is None else float(within)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.abs(amplitude))
    for row in range(rows):
        lo, hi = low_hz, high_hz
        if near is not None:
            if not np.isfinite(near[row]):
                continue
            lo, hi = max(lo, near[row] - within), min(hi, near[row] + within)
        a = int(np.searchsorted(frequency, lo, side="left"))
        b = int(np.searchsorted(frequency, hi, side="right"))
        values = db[row, a:b]
        if b <= a or not np.isfinite(values).any():
            continue
        out[row] = refine_peak(frequency, db[row], a + int(np.nanargmax(values)))
    return out


def spectrum_lines(frequency, power, low_hz: float, high_hz: float, *,
                   prominence: float = 8.0, near=None, within: float = 1.5,
                   floor_hz: float = FLOOR_HZ) -> dict[str, np.ndarray]:
    """The lines of one power spectrum: `hz` (refined between bins),
    `prominence_db`, `over_floor_db` and `level_db`.

    Without `near`, the peaks between `low_hz` and `high_hz` standing
    `prominence` dB over their neighbours (scipy's prominence, taken
    within the band). With `near`, a list of frequencies, the strongest
    bin within `within` Hz of each, one row each and NaN where there is
    none, and `near` as a column too.
    """
    frequency = np.asarray(frequency, dtype=float)
    power = np.asarray(power, dtype=float)
    tiny = np.finfo(float).tiny
    db = 10.0 * np.log10(np.maximum(np.nan_to_num(power), tiny))
    over = over_floor_db(frequency, power, floor_hz)[0]
    band = np.flatnonzero((frequency >= low_hz) & (frequency <= high_hz))
    columns = ("hz", "prominence_db", "over_floor_db", "level_db")
    if near is None:
        if len(band) < 3:
            return {c: np.zeros(0) for c in columns}
        tops, found = find_peaks(db[band], prominence=prominence)
        at = band[tops]
        prominences = found["prominences"]
    else:
        near = np.atleast_1d(np.asarray(near, dtype=float))
        at = np.full(len(near), -1)
        for i, hz in enumerate(near):
            inside = np.flatnonzero((frequency >= hz - within) & (frequency <= hz + within))
            if np.isfinite(hz) and len(inside):
                at[i] = inside[np.argmax(db[inside])]
        ok = at >= 0
        prominences = np.full(len(near), np.nan)
        if ok.any() and len(band) >= 3:
            inside = ok & np.isin(at, band)
            if inside.any():
                prominences[inside] = peak_prominences(db[band], np.searchsorted(
                    band, at[inside]))[0]
    hz = np.array([refine_peak(frequency, db, i) if i >= 0 else np.nan for i in at])
    table = {
        "hz": hz,
        "prominence_db": np.asarray(prominences, dtype=float),
        "over_floor_db": np.array([over[i] if i >= 0 else np.nan for i in at]),
        "level_db": np.array([db[i] if i >= 0 else np.nan for i in at]),
    }
    if near is not None:
        table["near"] = near
    return table


def align_frames(frequency, amplitude, ratio) -> np.ndarray:
    """Each row of `amplitude` read at `ratio` times each frequency: a
    line at `ratio * f` lands at `f`. NaN rows where `ratio` is not a
    positive number; beyond the top, zero."""
    frequency = np.asarray(frequency, dtype=float)
    amplitude = np.atleast_2d(np.asarray(amplitude, dtype=float))
    ratio = np.broadcast_to(np.asarray(ratio, dtype=float), (amplitude.shape[0],))
    out = np.full(amplitude.shape, np.nan)
    for row, r in enumerate(ratio):
        if np.isfinite(r) and r > 0:
            out[row] = np.interp(frequency * r, frequency, amplitude[row], right=0.0)
    return out


def order_resample(values, sample_rate: float, rate, ref_hz: float) -> np.ndarray:
    """`values` resampled so a source turning at `rate` (one per sample,
    Hz) sounds as if it turned steadily at `ref_hz`: what it does once a
    turn falls once every `sample_rate / ref_hz` samples. A gap (NaN) is
    taken as zero."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    rate = np.asarray(rate, dtype=float)
    if len(values) != len(rate):
        raise ValueError("one rate per sample is needed")
    if not (np.isfinite(rate).all() and (rate > 0).all()):
        raise ValueError("the rate must be known and positive all through the stretch")
    if not ref_hz > 0:
        raise ValueError(f"ref_hz must be positive, got {ref_hz}")
    # Turns so far, in units of the reference's samples.
    turns = np.concatenate(([0.0], np.cumsum((rate[1:] + rate[:-1]) / (2.0 * ref_hz))))
    grid = np.arange(0.0, turns[-1], 1.0)
    where = np.interp(grid, turns, np.arange(len(values), dtype=float))
    return map_coordinates(values, [where], order=3, mode="nearest")


def cycle_count(values, sample_rate: float, rate, *, lock: int | None = None,
                lock_hz: float = 4.0) -> np.ndarray:
    """How many cycles of a source turning at `rate` (one per sample,
    Hz) have passed at each sample, from 0. Given `lock`, the count
    follows the phase of the line at `lock` times the rate: its slow
    wander, below `lock_hz`, is added, so the rate's small errors do not
    pile up over many cycles. A gap (NaN) is taken as zero."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    rate = np.asarray(rate, dtype=float)
    if len(values) != len(rate):
        raise ValueError("one rate per sample is needed")
    if not (np.isfinite(rate).all() and (rate > 0).all()):
        raise ValueError("the rate must be known and positive all through the stretch")
    cycles = np.concatenate(([0.0], np.cumsum((rate[1:] + rate[:-1]) / (2.0 * sample_rate))))
    if lock is None:
        return cycles
    if not lock >= 1:
        raise ValueError(f"lock is an order, 1 or more, got {lock}")
    if not 0 < lock_hz < sample_rate / 2:
        raise ValueError(f"lock_hz must be between 0 and {sample_rate / 2:g}, got {lock_hz}")
    turned = values * np.exp(-2j * np.pi * lock * cycles)
    sos = butter(4, lock_hz, fs=sample_rate, output="sos")
    slow = sosfiltfilt(sos, turned.real) + 1j * sosfiltfilt(sos, turned.imag)
    wander = np.unwrap(np.angle(slow)) / (2 * np.pi * lock)
    return cycles + wander - wander[0]


def cut_cycles(values, cycles, per_cycle: int) -> np.ndarray:
    """`values` resampled at `per_cycle` points a cycle, one row per
    whole cycle of `cycles` (the count at each sample, as `cycle_count`
    gives it): row i runs from cycle `ceil(cycles[0]) + i`."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    cycles = np.asarray(cycles, dtype=float)
    if per_cycle < 2:
        raise ValueError(f"per_cycle must be 2 or more, got {per_cycle}")
    if len(cycles) < 2 or np.any(np.diff(cycles) <= 0):
        raise ValueError("the cycle count must rise all through the stretch")
    first = np.ceil(cycles[0])
    whole = int(np.floor(cycles[-1]) - first)
    if whole < 1:
        return np.zeros((0, per_cycle))
    grid = first + np.arange(whole * per_cycle) / per_cycle
    where = np.interp(grid, cycles, np.arange(len(values), dtype=float))
    return map_coordinates(values, [where], order=3, mode="nearest").reshape(whole, per_cycle)


def comb_counts(lines, spacings, within: float = 0.5):
    """For each spacing, how many `lines` lie within `within` Hz of one
    of its multiples (the first counting from 1), and the chance of a
    random frequency doing so: `2 * within / spacing`."""
    lines = np.asarray(lines, dtype=float)[:, None]
    spacings = np.asarray(spacings, dtype=float)[None, :]
    k = np.maximum(np.round(lines / spacings), 1)
    counts = (np.abs(lines - k * spacings) <= within).sum(axis=0)
    odds = np.minimum(1.0, 2.0 * within / spacings[0])
    return counts, odds


def place_lines(lines, x: float, b: float, most: int = 4) -> np.ndarray:
    """For each line, the (n, k) whose `n * x + k * b` is nearest it,
    with `n >= 0` and `|k| <= most`; of near ties, fewer sidebands win."""
    out = []
    for hz in np.asarray(lines, dtype=float):
        best = None
        for k in range(-most, most + 1):
            n = max(0, int(round((hz - k * b) / x)))
            score = abs(hz - n * x - k * b) + 0.05 * abs(k)
            if best is None or score < best[0]:
                best = (score, n, k)
        out.append(best[1:])
    return np.array(out, dtype=int).reshape(-1, 2)


def fit_two_rates(lines, x: float, b: float, *, within: float = 0.5, most: int = 4,
                  rounds: int = 4):
    """`x` and `b` fitted by least squares to the lines within `within`
    Hz of their `n * x + k * b`, placing and fitting `rounds` times.
    Returns (x, b, nk) with nk each line's (n, k) at the end."""
    lines = np.asarray(lines, dtype=float)
    for _ in range(rounds):
        nk = place_lines(lines, x, b, most)
        near = np.abs(lines - nk @ (x, b)) <= within
        if near.sum() < 2 or np.linalg.matrix_rank(nk[near]) < 2:
            break
        (x, b), *_ = np.linalg.lstsq(nk[near].astype(float), lines[near], rcond=None)
    return float(x), float(b), place_lines(lines, x, b, most)


def chance_fits(count: int, low_hz: float, high_hz: float, x: float, b: float, *,
                within: float = 0.5, most: int = 4, tries: int = 500, seed: int = 0):
    """How many of `count` random frequencies between `low_hz` and
    `high_hz` lie within `within` of some `n * x + k * b`, in each of
    `tries` draws."""
    rng = np.random.default_rng(seed)
    out = np.zeros(tries, dtype=int)
    for i in range(tries):
        hz = rng.uniform(low_hz, high_hz, count)
        out[i] = int((np.abs(hz - place_lines(hz, x, b, most) @ (x, b)) <= within).sum())
    return out
