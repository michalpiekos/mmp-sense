"""Values indexed by time, in one or more aligned channels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .base import Dataset

__all__ = ["TimeSeries"]


@dataclass
class TimeSeries(Dataset):
    """A time coordinate and a value per channel per instant.

    `values` is (n_samples, n_channels), even for a single channel, so
    every consumer indexes it the same way. `time` is in seconds, must be
    non-decreasing, and is kept as given: a module never replaces a
    producer's timestamps.

    The sample rate is an optional claim by whoever built the series. When
    it is absent, `rate` derives one from the timestamps if they are
    regular enough, because DSP needs a number and the timestamps are the
    honest source for it.
    """

    time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    channels: list[str] = field(default_factory=list)
    units: str = ""
    sample_rate: float | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("values must be 1-D or 2-D (samples, channels)")
        self.values = values

        if self.time.ndim != 1:
            raise ValueError("time must be 1-D")
        if len(self.time) != len(self.values):
            raise ValueError(
                f"{len(self.time)} timestamps for {len(self.values)} samples"
            )
        if np.any(np.diff(self.time) < 0):
            raise ValueError("time must be non-decreasing")

        if not self.channels:
            self.channels = [f"c{i}" for i in range(self.values.shape[1])]
        if len(self.channels) != self.values.shape[1]:
            raise ValueError(
                f"{len(self.channels)} channel names for "
                f"{self.values.shape[1]} channels"
            )
        self.preferred_view = "timeseries"

    # -- shape ---------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_channels(self) -> int:
        return self.values.shape[1]

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.n_samples else 0.0

    @property
    def t_start(self) -> float:
        return float(self.time[0]) if self.n_samples else 0.0

    @property
    def t_end(self) -> float:
        return float(self.time[-1]) if self.n_samples else 0.0

    # -- rate ----------------------------------------------------------

    def is_regular(self, tolerance: float = 0.01) -> bool:
        """True when the sampling intervals vary by less than `tolerance`
        of the mean interval -- jitter a filter can ignore."""
        if self.n_samples < 3:
            return False
        steps = np.diff(self.time)
        mean = steps.mean()
        return bool(mean > 0 and steps.std() <= tolerance * mean)

    @property
    def rate(self) -> float:
        """Samples per second.

        The declared rate wins when there is one, since the producer knows
        what it configured; otherwise it comes from the timestamps.
        """
        if self.sample_rate:
            return float(self.sample_rate)
        if self.n_samples < 2:
            raise ValueError(f"{self.name}: no sample rate and too few samples")
        return float((self.n_samples - 1) / (self.time[-1] - self.time[0]))

    @property
    def rate_or_none(self) -> float | None:
        """`rate`, or None when there is nothing to derive it from.

        A live stream is empty until its first samples arrive, and a
        module asked to work on it then should produce nothing rather
        than raise.
        """
        try:
            return self.rate
        except ValueError:
            return None

    # -- selection -----------------------------------------------------

    def channel(self, which: str | int) -> np.ndarray:
        """One channel's values, by name or by index."""
        return self.values[:, self._index(which)]

    def _index(self, which: str | int) -> int:
        if isinstance(which, (int, np.integer)):
            return int(which)
        try:
            return self.channels.index(which)
        except ValueError:
            raise KeyError(
                f"{self.name}: no channel {which!r}; have {self.channels}"
            ) from None

    def select(self, which: Iterable[str | int]) -> "TimeSeries":
        """A series holding only the named channels, in the order given."""
        picks = [self._index(w) for w in which]
        return replace(
            self,
            values=self.values[:, picks],
            channels=[self.channels[i] for i in picks],
        )

    def time_slice(self, t0: float, t1: float) -> "TimeSeries":
        """The samples in [t0, t1].

        Inclusive at both ends and never empty-by-surprise: a selection
        that falls between two samples yields a series with no samples,
        which modules downstream must handle rather than guess.
        """
        if t1 < t0:
            t0, t1 = t1, t0
        keep = (self.time >= t0) & (self.time <= t1)
        return replace(self, time=self.time[keep], values=self.values[keep])

    # -- conversion ----------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """A DataFrame indexed by time -- for ad-hoc work in a notebook,
        not as mmp-sense's internal representation."""
        return pd.DataFrame(
            self.values, columns=self.channels, index=pd.Index(self.time, name="time")
        )

    @classmethod
    def from_dataframe(
        cls, frame: pd.DataFrame, *, time_column: str | None = None, **kwargs: Any
    ) -> "TimeSeries":
        if time_column is not None:
            time = frame[time_column].to_numpy(dtype=float)
            frame = frame.drop(columns=[time_column])
        else:
            time = frame.index.to_numpy(dtype=float)
        return cls(
            time=time,
            values=frame.to_numpy(dtype=float),
            channels=[str(c) for c in frame.columns],
            **kwargs,
        )

    def summary(self) -> str:
        rate = f"{self.rate:.6g} Hz" if self.n_samples > 1 else "?"
        return (
            f"{self.name}: {self.n_samples} samples x {self.n_channels} "
            f"channels [{', '.join(self.channels)}], {self.duration:.6g} s "
            f"at {rate}" + (f", {self.units}" if self.units else "")
        )
