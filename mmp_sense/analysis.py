"""Sessions without a window: a recording, or a table of your own.

    import mmp_sense

    recording = mmp_sense.load("bench.mcap")
    recording.events(triple)               # asked exactly as the viewer is

    table = pandas.DataFrame({"x": ...}, index=time_in_seconds)
    bench = mmp_sense.from_frame(table)

And what to make of a list of lines (`session.lines`): which spacings
explain them (`comb_scan`), two rates and their sums and differences
(`fit_rates`), how high given frequencies stand (`heights`), and
whether a chosen set of them stands higher than chance would put it
(`permutation_lift`) -- each against chance.

Nothing here imports Qt.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from dataclasses import dataclass

from scipy.stats import binom

from . import dsp
from .core.data import TimeSeries
from .core.cycles import CycleAverage
from .core.session import Session, stream_names

__all__ = ["from_frame", "load", "session_of", "comb_scan", "fit_rates", "RateFit",
           "CycleAverage", "heights", "permutation_lift"]


def session_of(series: TimeSeries, *, name: str | None = None,
               buffer_s: float | None = None) -> Session:
    """A session holding all of `series`, with nothing more to come."""
    from .sources.recording import SeriesSource

    return _loaded(SeriesSource(series, name=name), series.duration, buffer_s)


def from_frame(table: pd.DataFrame, *, name: str = "frame") -> Session:
    """A session holding a DataFrame: its index is time in seconds, its
    columns the channels."""
    time = np.asarray(table.index, dtype=float)
    series = TimeSeries(time=time, values=table.to_numpy(dtype=float),
                        channels=[str(c) for c in table.columns], name=name)
    return session_of(series, name=name)


def load(path, topic: str | None = None, *, topics=None, buffer_s: float | None = None,
         sample_rate: float | None = None, scale: float | None = None,
         units: str | None = None) -> Session:
    """A data file as a session holding the whole of it: an MCAP, a WAV or
    other audio, delimited text, or a NumPy array (see `io.files`).

    An MCAP opens every plottable topic, a stream each, unless `topic` or
    `topics` names them. Audio keeps its own rate and channels.
    `sample_rate` times a file that carries no time; a sensor's WAV
    (below 8 kHz) is in counts, or times `scale` in `units`.
    """
    from .sources.recording import FileSources

    if topic and not topics:
        topics = [topic]
    sources = FileSources(path, topics=topics, rate=sample_rate, scale=scale, units=units)
    if len(sources) > 1:
        return _loaded_all(sources, buffer_s)
    return _loaded(sources[0], sources[0].series.duration, buffer_s)


def file_buffer(sources, buffer_s: float) -> tuple[float, int]:
    """(seconds, samples) a session needs to hold all of these file
    sources: at least `buffer_s`, and the whole span however long."""
    held = [s.series for s in sources if s.series.n_samples]
    if held:
        span = max(x.time[-1] for x in held) - min(x.time[0] for x in held)
        buffer_s = max(buffer_s, span * 1.05 + 1.0)
    return buffer_s, max(whole_file_samples(s) for s in sources)


def whole_file_samples(source) -> int:
    """A buffer cap that holds all of a file source, however long.

    The default cap bounds a live stream's memory; a file is finite and
    is meant to be held whole, so it is sized to the file instead.
    """
    from .core.buffer import MAX_CAPACITY, MIN_CAPACITY

    return max(MAX_CAPACITY, source.series.n_samples + MIN_CAPACITY)


def _loaded_all(sources, buffer_s: float | None) -> Session:
    held = [s.series for s in sources if s.series.n_samples]
    duration = (max(x.time[-1] for x in held) - min(x.time[0] for x in held)) if held else 0.0
    session = Session(buffer_s=buffer_s or max(60.0, duration * 1.05 + 1.0),
                      max_samples=max(whole_file_samples(s) for s in sources))
    for source, name in zip(sources, stream_names(sources)):
        session.add(source, name)
    while session.poll():
        pass
    session.stop_intake()
    return session


def _loaded(source, duration: float, buffer_s: float | None) -> Session:
    session = Session(source, buffer_s=buffer_s or max(60.0, duration * 1.05 + 1.0),
                      max_samples=whole_file_samples(source))
    while session.poll():
        pass
    session.stop_intake()
    return session


# -- lines and the rates behind them ----------------------------------------


def _hz(lines) -> np.ndarray:
    if isinstance(lines, pd.DataFrame):
        lines = lines["hz"]
    hz = np.asarray(lines, dtype=float)
    return hz[np.isfinite(hz)]


def comb_scan(lines, spacings, *, within: float = 0.5, apart: float | None = 0.5,
              top: int | None = None) -> pd.DataFrame:
    """Which spacings explain `lines` (Hz, or `session.lines`' table)
    better than chance: for each, `lines` within `within` Hz of one of
    its multiples, `chance` -- how many random frequencies would, the
    odds being `2 * within / spacing` -- and `p`, how likely as many or
    more are by chance (binomial). Sorted by `p`; with `apart`, a
    spacing within `apart` Hz of a better one is left out, and `top`
    keeps the first so many."""
    hz = _hz(lines)
    spacings = np.asarray(spacings, dtype=float)
    counts, odds = dsp.comb_counts(hz, spacings, within)
    p = binom.sf(counts - 1, len(hz), odds)
    table = pd.DataFrame({"spacing": spacings, "lines": counts, "chance": len(hz) * odds,
                          "p": p}).sort_values(["p", "spacing"], kind="stable")
    if apart is not None:
        kept: list[float] = []
        rows = []
        for i, spacing in zip(table.index, table["spacing"]):
            if all(abs(spacing - k) > apart for k in kept):
                kept.append(spacing)
                rows.append(i)
                if top is not None and len(rows) >= top:
                    break
        table = table.loc[rows]
    elif top is not None:
        table = table.head(top)
    return table.reset_index(drop=True)


@dataclass(frozen=True)
class RateFit:
    """Two rates `x` and `b` fitted to lines at `n * x + k * b`.

    `table` has a row per line: `hz`, `n`, `k`, `off` (Hz from its
    `n * x + k * b`) and `fits` (within the tolerance); `fitted` is how
    many fit, `chance` how many random frequencies fit on average over
    the draws and `chance_max` the most any draw managed."""

    x: float
    b: float
    table: pd.DataFrame
    fitted: int
    chance: float
    chance_max: int

    def __repr__(self) -> str:
        return (f"RateFit(x={self.x:.2f}, b={self.b:.2f}, {self.fitted} of {len(self.table)} "
                f"lines, chance {self.chance:.1f}, at most {self.chance_max})")


def fit_rates(lines, x: float, b: float, *, within: float = 0.5, most: int = 4,
              rounds: int = 4, nk=None, band: tuple[float, float] | None = None,
              tries: int = 500, seed: int = 0) -> RateFit:
    """Lines as the multiples of two rates and their sums and
    differences, `n * x + k * b` with `|k| <= most`: each line placed on
    the nearest (fewer `k` winning a near tie), `x` and `b` fitted by
    least squares to those within `within` Hz, `rounds` times, starting
    from the `x` and `b` given.

    `nk`, (n, k) per line from an earlier fit, keeps the placing and
    fits every line given (NaN left out): the same lines measured again,
    e.g. in another stretch, where they have all moved.
    Chance is `tries` draws of as many random frequencies over `band`
    (the lines' own span if not given), placed on the fitted rates."""
    hz = np.asarray(_hz(lines) if nk is None else
                    (lines["hz"] if isinstance(lines, pd.DataFrame) else lines), dtype=float)
    if nk is None:
        x, b, nk = dsp.fit_two_rates(hz, x, b, within=within, most=most, rounds=rounds)
    else:
        nk = np.asarray(nk, dtype=int).reshape(-1, 2)
        if len(nk) != len(hz):
            raise ValueError(f"{len(nk)} placings for {len(hz)} lines")
        ok = np.isfinite(hz)
        if ok.sum() < 2 or np.linalg.matrix_rank(nk[ok]) < 2:
            raise ValueError("two rates need at least two lines placed apart")
        (x, b), *_ = np.linalg.lstsq(nk[ok].astype(float), hz[ok], rcond=None)
        x, b = float(x), float(b)
    off = hz - nk @ (x, b)
    fits = np.abs(off) <= within
    finite = hz[np.isfinite(hz)]
    low, high = band if band is not None else (finite.min(), finite.max())
    draws = dsp.chance_fits(len(finite), low, high, x, b, within=within, most=most,
                            tries=tries, seed=seed)
    table = pd.DataFrame({"hz": hz, "n": nk[:, 0], "k": nk[:, 1], "off": off, "fits": fits})
    return RateFit(x, b, table, int(fits.sum()), float(draws.mean()), int(draws.max()))


def heights(spectrum: pd.Series, hz, *, within: float = 1.0,
            floor_hz: float = dsp.comb.FLOOR_HZ) -> pd.Series:
    """How far the line nearest each of `hz` stands over the floor, in
    dB: the highest bin within `within` Hz, over the median of the
    `floor_hz` around it. `spectrum` is power against frequency, as
    `session.psd` gives it (square `session.spectrum` first)."""
    frequency = spectrum.index.to_numpy(float)
    over = dsp.over_floor_db(frequency, spectrum.to_numpy(float), floor_hz)[0]
    hz = np.atleast_1d(np.asarray(hz, dtype=float))
    out = []
    for f in hz:
        inside = (frequency >= f - within) & (frequency <= f + within)
        out.append(over[inside].max() if inside.any() else np.nan)
    return pd.Series(out, index=pd.Index(hz, name="hz"), name="over_floor_db")


def permutation_lift(values, chosen, *, tries: int = 5000, seed: int = 0) -> pd.Series:
    """How much higher the `chosen` values stand than the rest (the
    difference of their means: `lift`), and how often shuffling the
    values among the places does as well or better (`p`) -- a
    permutation test. `chosen` is a boolean mask, or a function of the
    index giving one."""
    values = pd.Series(values) if not isinstance(values, pd.Series) else values
    mask = np.asarray(chosen(values.index) if callable(chosen) else chosen, dtype=bool)
    v = values.to_numpy(float)
    if mask.all() or not mask.any():
        raise ValueError("choose some of the values, not none or all")

    def lift(x):
        return x[mask].mean() - x[~mask].mean()

    seen = lift(v)
    rng = np.random.default_rng(seed)
    shuffled = np.array([lift(rng.permutation(v)) for _ in range(tries)])
    return pd.Series({"lift": seen, "p": float(np.mean(shuffled >= seen))})
