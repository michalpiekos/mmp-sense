"""A source's cycles, averaged: what `Session.cycle_average` answers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["CycleAverage", "average_cycles"]


@dataclass(frozen=True)
class CycleAverage:
    """The mean of a stretch's cycles, each resampled to the same length.

    `mean` is the averaged cycle and `noise` what of the stretch's noise
    is left in it -- half the difference of the even and the odd cycles'
    means, where whatever repeats cancels -- both indexed by the position
    in the cycle, 0 to 1. `cycles` is how many were averaged, `rate_hz`
    the cycle's median rate over the stretch."""

    mean: pd.Series
    noise: pd.Series
    cycles: int
    rate_hz: float

    def __repr__(self) -> str:
        return (f"CycleAverage({self.cycles} cycles at {self.rate_hz:.2f} Hz, "
                f"{len(self.mean)} points a cycle)")

    def orders(self, top: int = 36) -> pd.DataFrame:
        """The averaged cycle's orders 1..`top`, a row each: `amplitude`,
        `noise` (the same of the noise) and `ratio`, the one over the
        other -- an order that repeats stands well over 1."""
        top = min(int(top), len(self.mean) // 2 - 1)
        amplitude = _orders(self.mean.to_numpy(float))[1:top + 1]
        noise = _orders(self.noise.to_numpy(float))[1:top + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = amplitude / noise
        return pd.DataFrame({"amplitude": amplitude, "noise": noise, "ratio": ratio},
                            index=pd.RangeIndex(1, top + 1, name="order"))

    def parts(self, n: int) -> pd.DataFrame:
        """The cycle cut into `n` equal parts -- six firings of a six --
        a row each: `start`, where in the cycle it begins, `rms` over the
        parts' mean RMS, and `lag`, how far the part sits from the parts'
        mean shape, in parts (-0.5..0.5). Alike parts read 1 and 0.

        The cycle's 0 is wherever the average began, so the parts are cut
        where their mean shape is quietest: a firing is not split between
        two parts."""
        self._parts(n)
        values = self.mean.to_numpy(float)
        width = len(values) // n
        quiet = _quietest(values.reshape(n, width).mean(axis=0))
        parts = np.roll(values, -quiet).reshape(n, width)
        rms = np.sqrt((parts ** 2).mean(axis=1))
        shape = parts.mean(axis=0)
        across = np.fft.irfft(np.fft.rfft(parts, axis=1) * np.conj(np.fft.rfft(shape)),
                              n=width, axis=1)
        best = across.argmax(axis=1)
        lag = ((best + width // 2) % width - width // 2) / width
        start = (quiet + np.arange(n) * width) / len(values)
        return pd.DataFrame({"start": start, "rms": rms / rms.mean(), "lag": lag},
                            index=pd.RangeIndex(1, n + 1, name="part"))

    def uneven(self, n: int, top: int | None = None) -> pd.Series:
        """How unlike its `n` parts are: the power of the orders 1..`top`
        (6 * n by default) that are not multiples of `n`, over that of
        those that are -- 0 for parts all alike -- and `noise`, the same
        of the noise over the mean's power on the multiples: what noise
        alone would give."""
        self._parts(n)
        top = 6 * n if top is None else int(top)
        top = min(top, len(self.mean) // 2 - 1)
        if top < n:
            raise ValueError(f"top must reach order {n}, got {top}")
        on = np.arange(1, top + 1) % n == 0
        mean = _orders(self.mean.to_numpy(float))[1:top + 1] ** 2
        noise = _orders(self.noise.to_numpy(float))[1:top + 1] ** 2
        total = mean[on].sum()
        return pd.Series({"uneven": mean[~on].sum() / total, "noise": noise[~on].sum() / total})

    def _parts(self, n: int) -> np.ndarray:
        values = self.mean.to_numpy(float)
        if n < 1 or len(values) % n:
            raise ValueError(f"{len(values)} points a cycle do not cut into {n} equal parts; "
                             "pick per_cycle a multiple of it")
        return values.reshape(n, -1)


def _quietest(shape: np.ndarray) -> int:
    """Where a tenth of `shape`, read round and round, holds least power: its start."""
    span = max(1, len(shape) // 10)
    power = np.convolve(np.tile(shape ** 2, 2), np.ones(span), "valid")[:len(shape)]
    return int(power.argmin())


def _orders(cycle: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(cycle)) * 2.0 / len(cycle)


def average_cycles(rows: np.ndarray, rate_hz: float) -> CycleAverage:
    """The `CycleAverage` of `rows`, one cycle each."""
    count, per_cycle = rows.shape
    if count < 2:
        raise ValueError(f"{count} whole cycle in the stretch; averaging needs 2 or more")
    pairs = count // 2
    noise = (rows[0:2 * pairs:2].mean(axis=0) - rows[1:2 * pairs:2].mean(axis=0)) / 2.0
    where = pd.Index(np.arange(per_cycle) / per_cycle, name="cycle")
    return CycleAverage(pd.Series(rows.mean(axis=0), index=where, name="mean"),
                        pd.Series(noise, index=where, name="noise"), count, float(rate_hz))
