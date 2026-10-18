"""The live store: recent samples, oldest overwritten first.

This is the hot path, so it is NumPy and nothing else. A DataFrame view
of it is a convenience `SignalTable` offers on request; no incoming
sample passes through pandas.

The invariants are the ones a chart and an FFT both need: time comes
out non-decreasing, a sample older than the newest is dropped rather than
inserted, and a large step backwards is a new timeline rather than a late
sample.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass

import numpy as np

__all__ = ["ChannelBuffer", "Clock", "MIN_CAPACITY", "MAX_CAPACITY", "RESTART_S"]

_SERIALS = itertools.count(1)

MIN_CAPACITY = 1024
MAX_CAPACITY = 1 << 23
#: A step back in time larger than this is a new timeline (a source
#: restarted, a clock was set); a smaller one is a sample arriving late.
RESTART_S = 1.0


@dataclass(frozen=True)
class Clock:
    """A correction of a stream's clock: a sample stamped `t` happened at
    `t - offset - drift * (t - at)` -- `offset` seconds late at `at`, and
    `drift` (a fraction: 80 ppm is 80e-6) later per second after it."""

    offset: float = 0.0
    drift: float = 0.0
    at: float = 0.0

    def apply(self, time):
        time = np.asarray(time, dtype=float)
        return time - self.offset - self.drift * (time - self.at)

    def then(self, other: "Clock") -> "Clock":
        """This correction followed by `other`, as one, stated at `other.at`."""
        a1, b1 = 1.0 - self.drift, self.drift * self.at - self.offset
        a2, b2 = 1.0 - other.drift, other.drift * other.at - other.offset
        a, b = a2 * a1, a2 * b1 + b2
        return Clock(offset=other.at - (a * other.at + b), drift=1.0 - a, at=other.at)

    def describe(self) -> dict:
        return {"offset_s": self.offset, "drift_ppm": self.drift * 1e6, "at": self.at}


class ChannelBuffer:
    """`span_s` seconds of a multi-channel signal, in seconds and values.

    Grows, up to `max_capacity`, while it holds less than `span_s` of
    history, so a source whose declared rate is wrong or missing still
    ends up with the configured number of seconds.
    """

    def __init__(
        self,
        channels,
        *,
        span_s: float = 60.0,
        rate: float = 0.0,
        units: str = "",
        max_capacity: int = MAX_CAPACITY,
        dtype=float,
    ) -> None:
        self.channels = [str(c) for c in channels]
        #: What a sample is stored as: the source's own type, so 16-bit
        #: audio takes two bytes a channel and not eight.
        self.dtype = np.dtype(dtype)
        if not self.channels:
            raise ValueError("a buffer needs at least one channel")
        self.span_s = float(span_s)
        self.declared_rate = float(rate)
        self._nominal_rate = self.declared_rate
        self.units = units
        self.max_capacity = int(max_capacity)

        capacity = MIN_CAPACITY
        if rate > 0 and span_s > 0:
            capacity = max(MIN_CAPACITY, int(rate * span_s) + MIN_CAPACITY)
        self.capacity = min(capacity, self.max_capacity)
        self.time = np.zeros(self.capacity, dtype=float)
        self.values = np.zeros((self.capacity, len(self.channels)), dtype=self.dtype)
        #: The correction applied to the source's timestamps, or None.
        self.clock: Clock | None = None
        #: The timestamps as the source gave them, kept while corrected.
        self.raw: np.ndarray | None = None
        self.start = 0
        self.size = 0
        #: Bumped on every change, so a cache can tell whether it is stale.
        self.version = 0
        #: Unique to this buffer for the life of the process, unlike id().
        self.serial = next(_SERIALS)
        self.received = 0
        #: Bumped whenever what is held stops continuing what was held
        #: before -- a clear or a restart -- so a result can tell it may
        #: be extended rather than recomputed.
        self.epoch = 0
        self.late = 0
        self.restarts = 0
        # A source may append from a worker thread while the GUI or a
        # query reads, so every touch of the arrays happens under this.
        self._lock = threading.RLock()

    # -- shape ---------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def t_start(self) -> float:
        with self._lock:
            return float(self.time[self.start]) if self.size else 0.0

    @property
    def t_end(self) -> float:
        with self._lock:
            if not self.size:
                return 0.0
            return float(self.time[(self.start + self.size - 1) % self.capacity])

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def rate(self) -> float:
        """Samples per second: what the source declared, else measured."""
        if self.declared_rate > 0:
            return self.declared_rate
        with self._lock:
            if self.size < 2:
                return 0.0
            span = self.t_end - self.t_start
        return float((self.size - 1) / span) if span > 0 else 0.0

    def index(self, name: str) -> int:
        try:
            return self.channels.index(name)
        except ValueError:
            raise KeyError(
                f"no channel {name!r}; have {', '.join(self.channels)}"
            ) from None

    # -- writing -------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self.start = 0
            self.size = 0
            self.version += 1
            self.epoch += 1

    def set_clock(self, clock: Clock | None) -> None:
        """Correct every timestamp, held and to come, by `clock` (from the
        source's own), or put them back as the source gave them."""
        with self._lock:
            # A clock that runs slow stamps samples closer together than they came.
            self.declared_rate = (self._nominal_rate if clock is None
                                  else self._nominal_rate / (1.0 - clock.drift))
            if clock is None:
                if self.raw is not None:
                    self.time, self.raw = self.raw, None
            else:
                if self.raw is None:
                    self.raw = self.time.copy()
                self.time = clock.apply(self.raw)
            self.clock = clock
            self.version += 1
            self.epoch += 1

    def append(self, time, values) -> int:
        """Add samples; returns how many were kept."""
        time = np.asarray(time, dtype=float).reshape(-1)
        values = np.asarray(values)
        if values.dtype != self.dtype:
            values = values.astype(self.dtype)
        if values.ndim == 1:
            values = values[:, None]
        if len(time) == 0:
            return 0
        if values.shape != (len(time), self.n_channels):
            raise ValueError(
                f"{values.shape} values for {len(time)} timestamps "
                f"and {self.n_channels} channels"
            )
        with self._lock:
            time, values = self._in_order(time, values)
            if len(time) == 0:
                return 0
            if self._wants_more(time):
                self._grow_to(min(self.capacity * 2, self.max_capacity))
            self._write(time, values)
            self.received += len(time)
            self.version += 1
            return len(time)

    def _stamps(self) -> np.ndarray:
        """The source's own timestamps: what arriving samples are ordered by."""
        return self.raw if self.raw is not None else self.time

    def _in_order(self, time, values):
        if self.size:
            newest = float(self._stamps()[(self.start + self.size - 1) % self.capacity])
            if float(time[-1]) < newest - RESTART_S:
                self.start = 0
                self.size = 0
                self.restarts += 1
                self.epoch += 1
            else:
                keep = time >= newest
                dropped = int((~keep).sum())
                if dropped:
                    self.late += dropped
                    time, values = time[keep], values[keep]
        if len(time) > 1 and np.any(np.diff(time) < 0):
            # Within one batch, sorted rather than dropped: a batch is one
            # flush and its order is ours to fix.
            order = np.argsort(time, kind="stable")
            time, values = time[order], values[order]
        return time, values

    def _wants_more(self, time) -> bool:
        """Whether what survives this append would span less than span_s."""
        if self.capacity >= self.max_capacity or self.span_s <= 0:
            return False
        total = self.size + len(time)
        if total <= self.capacity:
            return False
        kept = total - self.capacity
        if kept < self.size:
            oldest = self._stamps()[(self.start + kept) % self.capacity]
        else:
            oldest = time[kept - self.size]
        span = float(time[-1]) - float(oldest)
        return 0 < span < self.span_s

    def _grow_to(self, capacity: int) -> None:
        time, values = self._view()
        raw = None if self.raw is None else self.raw[
            (self.start + np.arange(self.size)) % self.capacity]
        self.capacity = capacity
        self.time = np.zeros(capacity, dtype=float)
        self.values = np.zeros((capacity, self.n_channels), dtype=self.dtype)
        self.time[: len(time)] = time
        self.values[: len(time)] = values
        if raw is not None:
            self.raw = np.zeros(capacity, dtype=float)
            self.raw[: len(raw)] = raw
        self.start = 0
        self.size = len(time)

    def _write(self, time, values) -> None:
        n = len(time)
        if n >= self.capacity:
            time, values, n = time[-self.capacity:], values[-self.capacity:], self.capacity
        stamps = time
        if self.clock is not None:
            time = self.clock.apply(stamps)
        end = (self.start + self.size) % self.capacity
        first = min(n, self.capacity - end)
        self.time[end:end + first] = time[:first]
        self.values[end:end + first] = values[:first]
        if self.raw is not None:
            self.raw[end:end + first] = stamps[:first]
        if first < n:
            self.time[: n - first] = time[first:]
            self.values[: n - first] = values[first:]
            if self.raw is not None:
                self.raw[: n - first] = stamps[first:]
        overflow = max(0, self.size + n - self.capacity)
        self.start = (self.start + overflow) % self.capacity
        self.size = min(self.size + n, self.capacity)

    # -- reading -------------------------------------------------------

    def _view(self):
        """Oldest-first copies of everything held. Call under the lock."""
        if not self.size:
            return np.zeros(0), np.zeros((0, self.n_channels), dtype=self.dtype)
        index = (self.start + np.arange(self.size)) % self.capacity
        return self.time[index].copy(), self.values[index].copy()

    def snapshot(self, t0: float | None = None, t1: float | None = None):
        """(time, values) for [t0, t1], copied so the source may carry on.

        Either end may be None, meaning "as far as it goes". Only what is
        asked for is copied: half a second out of a minute costs half a
        second, which is what makes an FFT of the cursor window cheap
        however much history is held.
        """
        with self._lock:
            if not self.size:
                return np.zeros(0), np.zeros((0, self.n_channels), dtype=self.dtype)
            lo = 0 if t0 is None else self._position(t0, "left")
            hi = self.size if t1 is None else self._position(t1, "right")
            return self._copy(lo, max(lo, hi))

    def snapshot_all(self):
        """(time, values, origin, epoch): everything held, copied, with the
        running number of its first sample and the timeline it belongs to.

        Sample numbers count every sample kept since the buffer was made,
        so two snapshots of one epoch overlap exactly where their numbers do.
        """
        with self._lock:
            time, values = self.snapshot()
            return time, values, self.received - self.size, self.epoch

    def _segments(self):
        """The one or two runs the ring holds, oldest first. Under the lock."""
        end = self.start + self.size
        if end <= self.capacity:
            return [(self.start, end)]
        return [(self.start, self.capacity), (0, end - self.capacity)]

    def _position(self, t: float, side: str) -> int:
        """Where `t` falls in the oldest-first ordering. Under the lock."""
        offset = 0
        for begin, stop in self._segments():
            run = self.time[begin:stop]
            if len(run) and (t < run[-1] or (side == "right" and t == run[-1])):
                return offset + int(np.searchsorted(run, t, side=side))
            offset += stop - begin
        return offset

    def _copy(self, lo: int, hi: int):
        """Samples [lo, hi) of the oldest-first ordering. Under the lock."""
        times, values = [], []
        offset = 0
        for begin, stop in self._segments():
            length = stop - begin
            a, b = max(lo, offset), min(hi, offset + length)
            if b > a:
                times.append(self.time[begin + a - offset:begin + b - offset])
                values.append(self.values[begin + a - offset:begin + b - offset])
            offset += length
        if not times:
            return np.zeros(0), np.zeros((0, self.n_channels), dtype=self.dtype)
        if len(times) == 1:
            return times[0].copy(), values[0].copy()
        return np.concatenate(times), np.concatenate(values)

    def tail(self, seconds: float):
        """The most recent `seconds` of history."""
        end = self.t_end
        return self.snapshot(end - float(seconds), None)
