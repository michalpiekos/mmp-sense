"""A recording as a source: replayed in time, or handed over at once.

A file and a live stream reach the session the same way, which is what
lets a feature worked out on a knock this afternoon be run over last
week's dataset without changing a line of it.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np

from ..core.data import TimeSeries
from .base import EMPTY, Source

__all__ = ["SeriesSource", "FileSources", "RecordingSource", "RecordingSources",
           "CsvSource", "AudioSource", "replay_together"]


class SeriesSource(Source):
    """Any TimeSeries, given out in order.

    With `replay`, samples appear at the rate they were recorded at (times
    `speed`), which is what makes a recording feel like the bench it came
    from. Without it, the whole series arrives on the first read.
    """

    kind = "recording"

    def __init__(
        self,
        series: TimeSeries,
        *,
        name: str | None = None,
        replay: bool = False,
        speed: float = 1.0,
        clock=_time.monotonic,
    ) -> None:
        self.series = series
        self.name = name or series.name
        self.channels = list(series.channels)
        self.units = series.units
        self.rate = float(series.rate_or_none or 0.0)
        self.dtype = self.series.values.dtype
        self.replay = bool(replay)
        self.speed = float(speed)
        self.clock = clock
        #: The recorded time replay starts from: the series' first, unless
        #: several are replayed together (`replay_together`).
        self.origin: float | None = None
        self._started: float | None = None
        self._sent = 0

    def start(self) -> "SeriesSource":
        if self._started is None:
            self._started = self.clock()
        return self

    def stop(self) -> None:
        self._started = None

    def rewind(self) -> None:
        self._sent = 0
        self._started = self.clock()

    def restart(self) -> "SeriesSource":
        """Play the recording from its first sample again."""
        self.rewind()
        return self

    def read(self):
        if self._started is None or self._sent >= self.series.n_samples:
            return EMPTY
        if not self.replay:
            time, values = self.series.time, self.series.values
            self._sent = self.series.n_samples
            return time, values
        elapsed = (self.clock() - self._started) * self.speed
        origin = self.series.t_start if self.origin is None else self.origin
        end = int(np.searchsorted(self.series.time, origin + elapsed, side="right"))
        if end <= self._sent:
            return EMPTY
        time = self.series.time[self._sent:end]
        values = self.series.values[self._sent:end]
        self._sent = end
        return time, values

    @property
    def finished(self) -> bool:
        return self._sent >= self.series.n_samples


def FileSources(path: str | Path, *, replay: bool = False, speed: float = 1.0,
                **read_args) -> list[SeriesSource]:
    """What a data file holds, a source per series, replayed together.
    The format is the file's own (`io.files.read_file`, which takes
    `read_args`)."""
    from ..io.files import read_file

    sources = [SeriesSource(series, name=series.name, replay=replay, speed=speed)
               for series in read_file(path, **read_args)]
    return replay_together(sources)


def RecordingSource(path: str | Path, topic: str | None = None, **kwargs
                    ) -> SeriesSource:
    """One topic of an MCAP recording."""
    from ..io.mcap import read_mcap

    series = read_mcap(path, topic)
    return SeriesSource(series, name=kwargs.pop("name", None) or series.name, **kwargs)


def RecordingSources(path: str | Path, topics, **kwargs) -> list[SeriesSource]:
    """Several topics of an MCAP recording, on one time base, and when
    replayed, replayed together."""
    from ..io.mcap import read_mcap_topics

    sources = [SeriesSource(series, name=series.name, **kwargs)
               for series in read_mcap_topics(path, list(topics))]
    return replay_together(sources)


def replay_together(sources: list[SeriesSource]) -> list[SeriesSource]:
    """Start these replaying at once, from the earliest first sample of
    any, so they keep the time they were recorded with between them."""
    held = [s.series.t_start for s in sources if s.series.n_samples]
    if len(sources) < 2 or not held:
        return sources
    origin = min(held)
    now = sources[0].clock()
    for source in sources:
        source.origin = origin
        source._started = now
    return sources


def CsvSource(path: str | Path, **kwargs) -> SeriesSource:
    """Delimited text with a time column and a column per channel."""
    from ..io.csv import read_timeseries

    read_args = {k: kwargs.pop(k) for k in ("time_column", "channels", "rate", "sep")
                 if k in kwargs}
    series = read_timeseries(path, **read_args)
    return SeriesSource(series, **kwargs)


def AudioSource(path: str | Path, **kwargs) -> SeriesSource:
    """A voice clip or other audio file, at its own rate, as one channel
    called `audio` or, for several, `ch0`, `ch1`, ..."""
    from ..io.audio import read_audio

    return SeriesSource(read_audio(path), **kwargs)
