"""Named columns over the session's streams: the session's signals.

A stream is one source's samples on its own clock: a buffer, and the
channels in it. A signal is a channel of a stream, or the values of a
recipe the session shows -- computed for whatever window is asked for,
so it follows the data live and over a recording alike.

A channel's full name is `stream:channel`. With one stream its channels
are listed by their own names; with several they are listed by full
name, and a bare channel name still finds the one stream that has it.

A `Window` is what an op, a pane and a query all see: a slice of one
stream's time, its channels, and any shown signal evaluated lazily and
cached for as long as that slice lives. The windows of one `Frame` were
taken together, so a pane drawing two streams draws them as of one moment.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from .buffer import ChannelBuffer
from .data import TimeSeries

__all__ = ["Signal", "SignalTable", "Window", "Frame", "View", "Live", "Stream",
           "SignalError", "SEPARATOR"]

#: Between a stream's name and a channel's in a full name: `left:x`.
SEPARATOR = ":"


class SignalError(Exception):
    """A signal that cannot be computed: unknown name, cycle, wrong shape."""


_SERIALS = itertools.count(1)


@dataclass
class Signal:
    """One column: what it is called, where it comes from, what it means."""

    name: str
    kind: str                      # "channel" or "derived"
    units: str = ""
    description: str = ""
    fn: Callable[["Window"], np.ndarray] | None = None
    source: str = ""               # "graph:<node name>" for a shown recipe
    #: The stream it is of: a channel's own; a derived signal's times.
    #: None for a derived signal means the first stream.
    stream: str | None = None
    #: A channel's name within its stream.
    channel: str = ""
    #: Unique to this definition: a signal defined again is a new one.
    serial: int = field(default_factory=lambda: next(_SERIALS))

    def describe(self) -> dict[str, Any]:
        out = {"name": self.name, "kind": self.kind, "units": self.units}
        if self.stream:
            out["stream"] = self.stream
        if self.description:
            out["description"] = self.description
        if self.source:
            out["source"] = self.source
        return out


@dataclass
class Stream:
    """One source's samples on its own clock."""

    name: str
    buffer: ChannelBuffer
    source: Any = None

    def describe(self) -> dict[str, Any]:
        buffer = self.buffer
        return {
            "name": self.name,
            "channels": list(buffer.channels),
            "units": buffer.units,
            "rate": buffer.rate,
            "dtype": str(buffer.dtype),
            "samples": len(buffer),
            "t_start": buffer.t_start,
            "t_end": buffer.t_end,
            "source": getattr(self.source, "name", ""),
        }


@dataclass(frozen=True)
class Live:
    """Where a whole-buffer window sits in the buffer's stream: samples
    `origin` to `end` of timeline `epoch` of buffer `serial`."""

    serial: int
    epoch: int
    origin: int
    length: int

    @property
    def end(self) -> int:
        """The running number of the last sample; origin - 1 when empty."""
        return self.origin + self.length - 1

    def follows(self, serial: int, epoch: int, end: int) -> bool:
        """Whether this window carries on from one that ended at `end`:
        the same timeline, nothing lost between them, nothing taken back."""
        return (serial == self.serial and epoch == self.epoch
                and self.origin <= end + 1 and end <= self.end)


class Window:
    """A slice of one stream's time and every signal computable over it.

    Values are computed when asked for and kept for the life of the
    window, so a pane, a feature and a query that all want `magnitude`
    for the same window compute it once. A signal of another stream is
    given at this window's times, each value the latest at or before;
    `home(name)` is the window of the signal's own stream.
    """

    def __init__(
        self,
        time: np.ndarray,
        values: np.ndarray,
        table: "SignalTable",
        *,
        rate: float = 0.0,
        units: str = "",
        name: str = "window",
        data_key: tuple | None = None,
        live: "Live | None" = None,
        stream: str | None = None,
        frame: "Frame | None" = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.values = np.asarray(values)
        self.table = table
        self.units = units
        self.name = name
        self._declared_rate = float(rate)
        #: Equal for two windows over the same samples of the same buffer;
        #: None when that cannot be told, as for a slice.
        self.data_key = data_key
        #: Where this window sits in its buffer's stream, when it holds
        #: everything the buffer does; None for a slice or a range.
        self.live = live
        #: The stream whose samples these are.
        self.stream = stream if stream is not None else table.first
        #: The windows taken with this one, of the other streams.
        self.frame = frame
        self._cache: dict[str, np.ndarray] = {}
        self._computing: list[str] = []

    # -- shape ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.time)

    @property
    def empty(self) -> bool:
        return len(self.time) == 0

    @property
    def t_start(self) -> float:
        return float(self.time[0]) if len(self.time) else 0.0

    @property
    def t_end(self) -> float:
        return float(self.time[-1]) if len(self.time) else 0.0

    @property
    def span(self) -> tuple[float, float]:
        return (self.t_start, self.t_end)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def rate(self) -> float:
        if self._declared_rate > 0:
            return self._declared_rate
        if len(self.time) < 2 or self.duration <= 0:
            return 0.0
        return float((len(self.time) - 1) / self.duration)

    @property
    def channels(self) -> list[str]:
        """This stream's channels, by the names the table lists them by."""
        return self.table.channels_of(self.stream)

    @property
    def signals(self) -> list[str]:
        return self.table.names

    # -- other streams -------------------------------------------------

    def of(self, stream: str) -> "Window":
        """The window of `stream` taken with this one."""
        if stream == self.stream:
            return self
        if self.frame is None:
            self.frame = Frame(self.table, None, None)
        return self.frame.window(stream)

    def home(self, name: str) -> "Window":
        """The window of the stream signal `name` is of."""
        return self.of(self.table.stream_of(name))

    def cut(self, t_end: float) -> "Window":
        """This window up to `t_end`: what several streams all hold."""
        n = int(np.searchsorted(self.time, t_end, side="right"))
        if n == len(self.time):
            return self
        key = None if self.data_key is None else (
            *self.data_key, "cut", n)
        live = None if self.live is None else Live(
            self.live.serial, self.live.epoch, self.live.origin, n)
        time, values = self.time[:n], self.values[:n]
        return Window(time, values, self.table, rate=self._declared_rate,
                      units=self.units, name=self.name, data_key=key, live=live,
                      stream=self.stream, frame=self.frame)

    # -- values --------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.table

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        signal = self.table.resolve(name)
        if name in self._computing:
            chain = " -> ".join([*self._computing, name])
            raise SignalError(f"{name} depends on itself: {chain}")

        if signal.kind == "channel":
            if signal.stream == self.stream:
                values = self.values[:, self.table.column(signal)]
            else:
                values = self._held(self.of(signal.stream), signal.name)
        else:
            self._computing.append(name)
            try:
                values = np.asarray(signal.fn(self), dtype=float)
            finally:
                self._computing.pop()
            if values.ndim == 0:
                values = np.full(len(self.time), float(values))
            if values.shape[0] != len(self.time):
                raise SignalError(
                    f"{name} produced {values.shape[0]} values for "
                    f"{len(self.time)} samples"
                )
        self._cache[name] = values
        return values

    def _held(self, other: "Window", name: str) -> np.ndarray:
        values = np.asarray(other[name], dtype=float)
        index = np.searchsorted(other.time, self.time, side="right") - 1
        out = values[np.clip(index, 0, None)] if len(values) else np.full(len(self.time), np.nan)
        out = np.array(out, dtype=float)
        out[index < 0] = np.nan
        return out

    def get(self, name: str, default=None):
        try:
            return self[name]
        except SignalError:
            return default

    # -- conversion ----------------------------------------------------

    def series(self, names=None, *, name: str | None = None) -> TimeSeries:
        """These signals as a TimeSeries, which is what panes draw."""
        names = list(names) if names is not None else self.channels
        columns = [np.asarray(self[n], dtype=float) for n in names]
        values = (
            np.stack(columns, axis=1) if columns
            else np.zeros((len(self.time), 0))
        )
        return TimeSeries(
            name=name or (names[0] if len(names) == 1 else self.name),
            time=self.time,
            values=values,
            channels=list(names),
            units=self.table.units_for(names),
            sample_rate=self.rate or None,
        )

    def to_frame(self, names=None) -> pd.DataFrame:
        """A DataFrame of this window -- for looking at, not for storing.

        Building it copies; the buffer stays NumPy.
        """
        names = list(names) if names is not None else self.signals
        return pd.DataFrame(
            {n: self[n] for n in names},
            index=pd.Index(self.time, name="time"),
        )

    def slice(self, t0: float | None, t1: float | None) -> "Window":
        lo = 0 if t0 is None else int(np.searchsorted(self.time, t0, side="left"))
        hi = (len(self.time) if t1 is None
              else int(np.searchsorted(self.time, t1, side="right")))
        return Window(
            self.time[lo:hi], self.values[lo:hi], self.table,
            rate=self._declared_rate, units=self.units, name=self.name,
            stream=self.stream, frame=Frame(self.table, t0, t1),
        )

    def __repr__(self) -> str:
        return (
            f"<Window {self.stream} {self.t_start:.6g}-{self.t_end:.6g} s, "
            f"{len(self.time)} samples, {len(self.signals)} signals>"
        )


class View:
    """The windows a recipe reads, one per stream it reads.

    Over several streams each window ends where the earliest of them
    does: past that, one stream has said nothing yet, and a value
    computed without it would change once it did.
    """

    def __init__(self, windows: dict[str, Window]) -> None:
        self.sync = frozenset(windows) if len(windows) > 1 else None
        if self.sync is not None:
            if any(w.empty for w in windows.values()):
                end = -np.inf
            else:
                end = min(w.t_end for w in windows.values())
            windows = {s: w.cut(end) for s, w in windows.items()}
        self.windows = windows
        self.live = all(w.live is not None for w in windows.values())
        keys = tuple((s, w.data_key) for s, w in sorted(windows.items()))
        self.key = None if any(k is None for _, k in keys) else (keys, self.sync)

    def of(self, stream: str) -> Window:
        try:
            return self.windows[stream]
        except KeyError:
            raise SignalError(f"stream {stream!r} is not among those read: "
                              f"{', '.join(self.windows)}") from None


class Frame:
    """The windows of every stream over one stretch, taken together."""

    def __init__(self, table: "SignalTable", t0: float | None, t1: float | None) -> None:
        self.table = table
        self.t0, self.t1 = t0, t1
        self.serial = next(_SERIALS)
        self._windows: dict[str, Window] = {}

    def window(self, stream: str | None = None) -> Window:
        stream = self.table.first if stream is None else stream
        found = self._windows.get(stream)
        if found is None:
            found = self._windows[stream] = self.table._take(stream, self.t0, self.t1, self)
        return found


class SignalTable:
    """The signals of one session: the streams' channels and the shown.

    `window()` is how to ask for a stretch of them.
    """

    def __init__(self, buffer: ChannelBuffer | None = None, on_change=None, *,
                 streams: list[Stream] | None = None) -> None:
        self.streams: dict[str, Stream] = {}
        self._signals: dict[str, Signal] = {}
        self._channels: dict[str, Signal] = {}
        self._on_change = on_change
        if streams is not None:
            self.set_streams(streams)
        else:
            self.rebind(buffer)

    # -- the streams' channels -----------------------------------------

    def rebind(self, buffer: ChannelBuffer) -> None:
        """Point at one other buffer, keeping every derived signal.

        A source is swapped, or a recording loaded, without losing the
        recipes the user worked out on the last one.
        """
        name = getattr(buffer, "name", None) or "main"
        self.set_streams([Stream(name, buffer)])

    def set_streams(self, streams: list[Stream]) -> None:
        """These streams and no others, keeping every derived signal."""
        names = [s.name for s in streams]
        if not names:
            raise SignalError("a session needs at least one stream")
        if len(set(names)) != len(names):
            raise SignalError(f"two streams of one name: {', '.join(names)}")
        for name in names:
            if SEPARATOR in name or not name:
                raise SignalError(f"a stream's name may not be empty or hold "
                                  f"{SEPARATOR!r}: {name!r}")
        self.streams = {s.name: s for s in streams}
        self._index()
        self._changed()

    def add_stream(self, stream: Stream) -> None:
        self.set_streams([*self.streams.values(), stream])

    def remove_stream(self, name: str) -> Stream:
        if name not in self.streams:
            raise SignalError(f"no stream {name!r}; have {', '.join(self.streams)}")
        gone = self.streams[name]
        self.set_streams([s for s in self.streams.values() if s.name != name])
        return gone

    def _index(self) -> None:
        derived = {n: s for n, s in self._signals.items() if s.kind != "channel"}
        several = len(self.streams) > 1
        self._signals, self._channels = {}, {}
        for stream in self.streams.values():
            buffer = stream.buffer
            for channel in buffer.channels:
                full = f"{stream.name}{SEPARATOR}{channel}"
                signal = Signal(
                    name=full if several else channel, kind="channel",
                    units=buffer.units, description=f"from {stream.name}",
                    stream=stream.name, channel=channel,
                )
                self._channels[full] = signal
                self._signals[signal.name] = signal
        for name, signal in derived.items():
            if name in self._signals or name in self._channels:
                continue                     # a channel took the name; the channel wins
            if signal.stream is not None and signal.stream not in self.streams:
                signal.stream = None
            self._signals[name] = signal

    @property
    def first(self) -> str:
        """The first stream's name: where a recipe naming no channel reads."""
        return next(iter(self.streams))

    @property
    def buffer(self) -> ChannelBuffer:
        """The first stream's buffer."""
        return self.streams[self.first].buffer

    @property
    def channels(self) -> list[str]:
        """Every stream's channels, as they are listed."""
        return [n for n, s in self._signals.items() if s.kind == "channel"]

    def channels_of(self, stream: str) -> list[str]:
        return [n for n, s in self._signals.items()
                if s.kind == "channel" and s.stream == stream]

    def column(self, signal: Signal) -> int:
        return self.streams[signal.stream].buffer.channels.index(signal.channel)

    @property
    def names(self) -> list[str]:
        return list(self._signals)

    @property
    def derived(self) -> list[str]:
        return [n for n, s in self._signals.items() if s.kind != "channel"]

    def _lookup(self, name: str) -> tuple[Signal | None, list[str]]:
        """(the signal, or None; the names it might have meant)."""
        if not isinstance(name, str):
            return None, []
        found = self._signals.get(name)
        if found is not None:
            return found, []
        found = self._channels.get(name)
        if found is not None:
            return found, []
        if SEPARATOR not in name:
            matches = [s for s in self._channels.values() if s.channel == name]
            if len(matches) == 1:
                return matches[0], []
            return None, [s.name for s in matches]
        return None, []

    def get(self, name: str) -> Signal | None:
        """The signal `name` is: listed, full or a bare channel one stream has."""
        return self._lookup(name)[0]

    def resolve(self, name: str) -> Signal:
        """As `get`, but an unknown or ambiguous name is an error saying why."""
        found, candidates = self._lookup(name)
        if found is not None:
            return found
        if candidates:
            raise SignalError(f"{name!r} is a channel of several streams; "
                              f"say which: {', '.join(candidates)}")
        raise SignalError(f"no signal {name!r}; have {', '.join(self.names)}")

    def ambiguous(self, name: str) -> bool:
        """Whether `name` is a bare channel name several streams have."""
        return bool(self._lookup(name)[1])

    def canonical(self, name: str) -> str:
        """The name `name`'s signal is listed by."""
        return self.resolve(name).name

    def stream_of(self, name: str) -> str:
        """The stream signal `name` is of."""
        signal = self.resolve(name)
        return signal.stream if signal.stream is not None else self.first

    def units_for(self, names) -> str:
        units = {s.units for s in (self.get(n) for n in names) if s is not None}
        return units.pop() if len(units) == 1 else ""

    # -- defining ------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self._signals)

    def __len__(self) -> int:
        return len(self._signals)

    def define(self, name: str, fn, *, units: str = "", description: str = "",
               source: str = "", stream: str | None = None) -> Signal:
        """A derived signal: `fn(window)` gives its values over a window.
        `stream` is the stream whose times it is drawn on."""
        existing = self.get(name)
        if (existing is not None and existing.kind == "channel") or any(
                s.channel == name for s in self._channels.values()):
            raise SignalError(f"{name} is a channel of the source; pick another name")
        if SEPARATOR in name:
            raise SignalError(f"{name!r}: {SEPARATOR!r} is kept for stream:channel")
        if stream is not None and stream not in self.streams:
            raise SignalError(f"no stream {stream!r}; have {', '.join(self.streams)}")
        signal = Signal(name=name, kind="derived", units=units, description=description,
                        fn=fn, source=source, stream=stream)
        self._signals[name] = signal
        self._changed()
        return signal

    def drop(self, name: str) -> None:
        signal = self._signals.get(name)
        if signal is None:
            raise SignalError(f"no signal {name!r}")
        if signal.kind == "channel":
            raise SignalError(f"{name} is a channel of the source and stays")
        del self._signals[name]
        self._changed()

    # -- reading -------------------------------------------------------

    def window(self, t0: float | None = None, t1: float | None = None, *,
               stream: str | None = None) -> Window:
        """Every signal over [t0, t1] of one stream (the first by default);
        with neither end, all its buffer holds -- a live window, whose
        derived signals are extended from the last one's rather than
        computed again."""
        return Frame(self, t0, t1).window(stream)

    def _take(self, stream: str, t0, t1, frame: Frame) -> Window:
        if stream not in self.streams:
            raise SignalError(f"no stream {stream!r}; have {', '.join(self.streams)}")
        buffer = self.streams[stream].buffer
        version = buffer.version
        live = None
        if t0 is None and t1 is None:
            time, values, origin, epoch = buffer.snapshot_all()
            live = Live(buffer.serial, epoch, origin, len(time))
            # Carried results are made of these: no one may change them.
            time.flags.writeable = values.flags.writeable = False
        else:
            time, values = buffer.snapshot(t0, t1)
        key = (buffer.serial, version, len(time),
               float(time[0]) if len(time) else 0.0,
               float(time[-1]) if len(time) else 0.0)
        return Window(
            time, values, self,
            rate=buffer.declared_rate, units=buffer.units,
            name=getattr(buffer, "name", "window"), data_key=key, live=live,
            stream=stream, frame=frame,
        )

    def set_listener(self, callback) -> None:
        """Who to tell when a signal is defined or dropped."""
        self._on_change = callback

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def __repr__(self) -> str:
        return f"<SignalTable {', '.join(self.names)}>"
