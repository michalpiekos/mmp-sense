"""The shared model: one session, three ways of using it.

Everything that is true of an analysis in progress lives here -- the
streams (a source and its buffer each), the signals, the cursor, the
selection, the analysis window and playback -- and the GUI, the Python
console and the MCP server are adapters over it. Whichever of them changes something,
the others hear about it through `listen`.

    from mmp_sense import Session, SyntheticAccelerometer

    session = Session(SyntheticAccelerometer(rate=1000)).start()
    session.poll()
    strength = mmp_sense.signal().highpass(2.0).magnitude().peak("60ms")
    session.values(strength, 4.0, 6.0)     # a pandas Series, by time

Nothing in this file imports Qt, and none of it needs a window.
"""

from __future__ import annotations

import logging
import math
import threading
import time as _time
from dataclasses import asdict, dataclass, field
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.signal import sosfiltfilt

from . import query
from .. import dsp
from .array import Array
from .bands import Band, check_band, ranges as band_ranges
from .buffer import MAX_CAPACITY, ChannelBuffer, Clock
from .cycles import CycleAverage, average_cycles
from .graph import Evaluator, Node
from .graph import signal as _signal
from .graph.evaluate import home_of
from .graph.series import Duration
from .graph.series import seconds as _seconds
from .graph.ops import event_table, ops
from .graph.ops import op as graph_op
from .data import EventSeries
from .plots import Plot, make_plot
from .signals import SEPARATOR, SignalError, SignalTable, Stream, Window

if TYPE_CHECKING:
    from ..sources.base import Source
    from .query import SyncEstimate

__all__ = ["Session", "Context", "Samples", "SignalRef", "EVENTS", "stream_names"]

log = logging.getLogger("mmp_sense.session")

#: What a question is asked of: a channel, a shown name, or a recipe.
SignalRef: TypeAlias = str | Node
#: A shaft or repetition rate: Hz, or the signal that says it.
RateRef: TypeAlias = float | SignalRef
#: Frequency bands: `Band`s, or (low, high) pairs in Hz.
Bands: TypeAlias = Sequence[Band | tuple[float, float]]

#: What a listener can be told about. A listener takes (kind, payload).
EVENTS = (
    "source",       # a source was attached; the channels may have changed
    "data",         # new samples arrived
    "signals",      # a signal was defined or dropped
    "cursor",       # the cursor moved
    "frequency",    # the frequency cursor moved or was cleared
    "selection",    # the selected region changed or was cleared
    "analysis",     # the analysis window changed
    "playback",     # playback started, stopped or reached the end
    "chart",        # something asked for a signal to be charted
    "plots",        # a named (x, y) result was drawn again or removed
    # Samples are no longer being taken, or are being taken again -- and
    # `running: True` means from an empty buffer, so the cursor and the
    # selection went with the samples and are worth reading again.
    "intake",
    "detections",   # an event node was shown or hidden
    "bands",        # a frequency band was added, changed or removed
)


@dataclass(frozen=True)
class Context:
    """What is being looked at, right now.

    The fields an AI tool or a script needs in order to talk about the
    same thing the user is looking at, without being told the timestamp,
    the signal or the window size every time.
    """

    cursor: float
    selection: tuple[float, float] | None
    active_signals: list[str]
    analysis_window: float
    dataset: str
    t_start: float = 0.0
    t_end: float = 0.0
    rate: float = 0.0
    playing: bool = False
    speed: float = 1.0
    following: bool = True
    running: bool = True           # whether samples are still arriving
    frequency_cursor: float | None = None
    #: Each stream: name, channels, units, rate, samples, t_start, t_end.
    streams: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["selection"] = list(self.selection) if self.selection else None
        return out


@dataclass(frozen=True, eq=False)
class Samples:
    """One signal over a stretch of time, as NumPy."""

    name: str
    values: NDArray[np.float64]
    time: NDArray[np.float64]
    rate: float                    # samples per second
    units: str = ""


def _title(x) -> str:
    return x if isinstance(x, str) else x.title


def _segments(name: str) -> list[str]:
    parts = [p for p in str(name).replace(SEPARATOR, "_").split("/") if p]
    return parts or ["stream"]


def stream_names(sources, taken=()) -> list[str]:
    """A short name for each source's stream: the last segment of its
    name (a topic's), with as many segments before it as it takes to
    tell it from the others, then a number."""
    taken = set(taken)
    paths = [_segments(getattr(s, "name", "") or "stream") for s in sources]
    out: list[str] = []
    for i, path in enumerate(paths):
        others = [p for j, p in enumerate(paths) if j != i]
        name = path[-1]
        for depth in range(1, len(path) + 1):
            name = "_".join(path[-depth:])
            if all("_".join(p[-depth:]) != name for p in others) and name not in taken:
                break
        stem, n = name, 1
        while name in taken or name in out:
            n += 1
            name = f"{stem}{n}"
        out.append(name)
    return out


class Session:
    """Streams, what they have produced lately, and what is being asked of them."""

    def __init__(
        self,
        source=None,
        *,
        buffer_s: float = 60.0,
        analysis_window: float = 0.5,
        dataset: str = "live",
        channels=("value",),
        rate: float = 0.0,
        units: str = "",
        clock: Callable[[], float] = _time.monotonic,
        max_samples: int | None = None,
    ) -> None:
        self.buffer_s = float(buffer_s)
        #: The buffer's hard cap. A live buffer keeps the default, which
        #: bounds memory however long a stream runs; a session holding a
        #: whole file raises it to the file's length.
        self.max_samples = int(max_samples) if max_samples else MAX_CAPACITY
        self.dataset = dataset
        self.clock = clock
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, dict], None]] = []
        self._endpoint = None

        buffer = ChannelBuffer(
            channels, span_s=self.buffer_s, rate=rate, units=units,
            max_capacity=self.max_samples,
        )
        self.signals = SignalTable(streams=[Stream(_segments(dataset)[-1], buffer)])
        self.signals.set_listener(self._signals_changed)
        #: Computes graph nodes, and keeps what one tick's panes share.
        self.evaluator = Evaluator()
        self._shown: dict[str, Node] = {}
        self._undrawn: set[str] = set()
        self._groups: dict[str, list[str]] = {}
        #: Microphone arrays by name, for the questions and the direction pane.
        self.arrays: dict[str, Array] = {}

        self._cursor = 0.0
        self._frequency_cursor: float | None = None
        self._selection: tuple[float, float] | None = None
        self._bands: list[Band] = []
        self._plots: dict[str, Plot] = {}
        self._analysis_window = float(analysis_window)
        self._active: list[str] = []
        self._following = True
        self._running = False
        self._playing = False
        self._speed = 1.0
        self._play_until: float | None = None
        self._last_tick: float | None = None

        if source is not None:
            self.attach(source)

    # -- streams -------------------------------------------------------

    @property
    def streams(self) -> dict[str, Stream]:
        """Every stream by name, in the order they were added."""
        return dict(self.signals.streams)

    @property
    def source(self) -> Source | None:
        """The first stream's source."""
        return self.signals.streams[self.signals.first].source

    @source.setter
    def source(self, source: Source) -> None:
        self.signals.streams[self.signals.first].source = source

    @property
    def buffer(self) -> ChannelBuffer:
        """The first stream's buffer."""
        return self.signals.buffer

    def _sources(self) -> list:
        return [s.source for s in self.signals.streams.values() if s.source is not None]

    def _buffer_for(self, source) -> ChannelBuffer:
        return ChannelBuffer(
            source.channels,
            span_s=self.buffer_s,
            rate=getattr(source, "rate", 0.0) or 0.0,
            units=getattr(source, "units", ""),
            max_capacity=self.max_samples,
            dtype=getattr(source, "dtype", float),
        )

    def _dataset(self) -> str:
        sources = [getattr(s, "name", "") for s in self._sources()]
        return " + ".join(n for n in sources if n) or self.dataset

    def attach(self, source: Source, *, keep_signals: bool = True, name: str | None = None) -> "Session":
        """Take samples from `source` from now on, and from it alone.

        The shown recipes are kept: a detector worked out on one stream
        is usually the reason to point mmp-sense at another.

        The source is started here, because what it is called and which
        channels it has are settled only once it has opened -- a live
        stream learns them from whatever it connects to.
        """
        if name is not None and (not name or SEPARATOR in name):
            raise ValueError(f"a stream's name may not be empty or hold {SEPARATOR!r}: {name!r}")
        source.start()
        with self._lock:
            for old in self._sources():
                if old is not source:
                    self._stop(old)
            stream = Stream(name or stream_names([source])[0], self._buffer_for(source), source)
            if not keep_signals:
                self.signals = SignalTable(streams=[stream])
                self.signals.set_listener(self._signals_changed)
                self._shown = {n: node for n, node in self._shown.items()
                               if node.output == "events"}
            else:
                self.signals.set_streams([stream])
            self.dataset = self._dataset()
            self._active = list(self.signals.channels)
            self._cursor = 0.0
            self._selection = None
            self._following = True
            self._running = True
            self.evaluator.clear()
            self._rehome()
        self.notify("source", dataset=self.dataset, channels=list(self.signals.channels))
        return self

    def add(self, source: Source, name: str | None = None) -> str:
        """Take samples from `source` too, as a stream of its own; returns
        the stream's name (`name`, or one made from the source's).

        A session still holding no source's samples is given this one
        instead, as `attach` would. A stopped session takes what the
        source has ready now -- all of a recording -- and no more.
        """
        if not self._sources():
            self.attach(source, name=name)
            return self.signals.first
        taken = [*self.signals.streams, *self.groups, *self.signals.derived]
        if name is not None and (name in taken or SEPARATOR in name):
            raise ValueError(f"{name!r} is taken by a stream, a group or a signal, "
                             f"or holds {SEPARATOR!r}")
        first = self.signals.first
        if len(self.signals.streams) == 1 and first in [*self.groups, *self.signals.derived]:
            raise ValueError(f"with a second stream, the first's name {first!r} names its "
                             "channels; a group or signal has that name -- drop it first")
        source.start()
        clash = [c for c in source.channels if c in self.signals.derived]
        if clash:
            self._stop(source)
            raise ValueError(f"{', '.join(clash)}: a shown signal has the name of a channel "
                             "of the new stream; drop or rename it first")
        with self._lock:
            if name is None:
                name = stream_names([source], taken=taken)[0]
            active = [self._full(a) for a in self._active]
            self.signals.add_stream(Stream(name, self._buffer_for(source), source))
            self.dataset = self._dataset()
            self._active = [self.signals.canonical(a) for a in active if a in self.signals]
            self._active += self.signals.channels_of(name)
            self._rehome()
            if not self._running:
                # Not polled while stopped: what it has ready now -- all of a file -- is taken once.
                time, values = source.read()
                if time is not None and len(time):
                    self.signals.streams[name].buffer.append(time, values)
        self.notify("source", dataset=self.dataset, channels=list(self.signals.channels))
        return name

    def remove(self, name: str) -> None:
        """Stop taking samples from a stream and forget what it held."""
        if len(self.signals.streams) == 1 and name in self.signals.streams:
            raise ValueError("a session keeps at least one stream; attach() another instead")
        with self._lock:
            active = [self._full(a) for a in self._active]
            gone = self.signals.remove_stream(name)
            if gone.source is not None:
                self._stop(gone.source)
            self.dataset = self._dataset()
            self._active = [self.signals.canonical(a) for a in active if a in self.signals]
            self.evaluator.clear()
            self._rehome()
        self.notify("source", dataset=self.dataset, channels=list(self.signals.channels))

    def _full(self, name: str) -> str:
        """A channel's full name, which means it whatever streams come and go."""
        signal = self.signals.get(name)
        if signal is not None and signal.kind == "channel":
            return f"{signal.stream}{SEPARATOR}{signal.channel}"
        return name

    def _rehome(self) -> None:
        """Put every shown recipe's signals on the stream it is now of."""
        for name, node in self._shown.items():
            try:
                home = home_of(node, self.signals) if node.output != "events" else None
            except SignalError:
                home = None
            for column in self._columns_of(name):
                self.signals.get(column).stream = home

    @staticmethod
    def _stop(source) -> None:
        try:
            source.stop()
        except Exception as exc:                             # noqa: BLE001
            log.warning("stopping %s: %s", source, exc)

    def start(self) -> "Session":
        sources = self._sources()
        for source in sources:
            source.start()
        if sources:
            self._running = True
        return self

    def stop(self) -> None:
        """Close the session down: playback, the sources and the endpoint."""
        self.pause()
        for source in self._sources():
            source.stop()
        self._running = False
        self.close_endpoint()

    def __enter__(self) -> "Session":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        """Whether samples are being taken from the sources."""
        return self._running

    def stop_intake(self) -> None:
        """Stop taking samples. Everything already taken stays.

        The sources are stopped as well as the intake, because one left
        running would either spend a core on samples nobody takes or
        pile up a backlog to be thrown away at the next start. What is
        held -- the buffers, the signals, the cursor, the selection -- is
        untouched, so a stopped session is the one to explore: replay,
        the FFT and the statistics all still answer.
        """
        if not self._running:
            return
        with self._lock:
            self._running = False
            sources = self._sources()
        for source in sources:
            self._stop(source)
        self.notify("intake", running=False)

    def start_intake(self) -> None:
        """Take samples again, from empty buffers.

        What the last capture left is dropped -- the samples, the cursor,
        the selection -- because a buffer holding two stretches of time
        with a gap between them is not something the analyses could
        describe honestly. The shown recipes stay: a detector is worth
        more than the samples it was worked out on.

        The sources are restarted before anything is cleared, so one that
        cannot start again leaves the last capture where it was.
        """
        if self._playing:
            self.pause()
        for source in self._sources():
            source.restart()
        with self._lock:
            for stream in self.signals.streams.values():
                stream.buffer.clear()
            self._selection = None
            self._cursor = self.span[1]
            self._following = True
            self._running = True
        self.notify("intake", running=True)

    def poll(self) -> int:
        """Take whatever the sources have ready. Returns samples kept."""
        if not self._running:
            return 0
        kept = 0
        for stream in list(self.signals.streams.values()):
            source = stream.source
            if source is None:
                continue
            time, values = source.read()
            if time is None or not len(time):
                continue
            kept += stream.buffer.append(time, values)
        if not kept:
            return 0
        end = self.span[1]
        if self._following and not self._playing:
            self._cursor = end
        self.notify("data", samples=kept, t_end=end)
        return kept

    # -- what is held --------------------------------------------------

    def window(self, t0: float | None = None, t1: float | None = None, *,
               stream: str | None = None) -> Window:
        """Every signal over [t0, t1] of one stream (the first by default);
        the whole buffer by default. `window.of(name)` is another
        stream's, taken with it."""
        return self.signals.window(t0, t1, stream=stream)

    @property
    def span(self) -> tuple[float, float]:
        """(first, last): the times of the first and last samples held,
        in any stream."""
        held = [s.buffer for s in self.signals.streams.values() if len(s.buffer)]
        if not held:
            return (self.buffer.t_start, self.buffer.t_end)
        return (min(b.t_start for b in held), max(b.t_end for b in held))

    @property
    def channels(self) -> list[str]:
        """Every stream's channels, by name."""
        return list(self.signals.channels)

    @property
    def rate(self) -> float:
        """The first stream's rate."""
        return self.buffer.rate

    # -- charts --------------------------------------------------------

    def chart(self, name: str, view: str | None = None) -> None:
        """Ask for a signal, or a group of them, to be charted.

        A GUI listening to the session opens a pane; with no GUI this
        only marks the signals active, so the same script works headless.
        """
        members = self.members(name)
        if members is None and name not in self.signals:
            raise SignalError(f"no signal or group {name!r}; have "
                              f"{', '.join([*self.groups, *self.signals.names])}")
        for signal in members if members is not None else [self.signals.canonical(name)]:
            if signal not in self._active:
                self._active.append(signal)
        self.notify("chart", signal=name, view=view)

    # -- plots: results that are not signals -----------------------------

    def plot(self, name: str, x: ArrayLike, y: ArrayLike, *, x_label: str = "",
             x_units: str = "", y_label: str = "", y_units: str = "") -> Plot:
        """Draw `y` against `x` under `name`, replacing what that name held.

        A rerun with the same name updates the same chart and registers
        nothing new. Without a window the plot is only kept, in `plots`.
        """
        plot = make_plot(name, x, y, x_label=x_label, x_units=x_units,
                         y_label=y_label, y_units=y_units)
        with self._lock:
            self._plots[name] = plot
        self.notify("plots", name=name)
        return plot

    def remove_plot(self, name: str) -> None:
        """Forget a plot; a window closes its chart."""
        with self._lock:
            if name not in self._plots:
                raise KeyError(f"no plot {name!r}; have {', '.join(self._plots) or 'none'}")
            del self._plots[name]
        self.notify("plots", name=name)

    @property
    def plots(self) -> dict[str, Plot]:
        """Every plot by name."""
        with self._lock:
            return dict(self._plots)

    # -- groups: several signals on one chart ------------------------------

    @property
    def groups(self) -> dict[str, list[str]]:
        """Every group by name, with the signals in it.

        `channels` is every stream's; with several streams each is one
        by its name; a shown node with components is one by its name; the
        rest are made with `group()`.
        """
        out = {"channels": list(self.signals.channels)}
        if len(self.signals.streams) > 1:
            for stream in self.signals.streams:
                out[stream] = self.signals.channels_of(stream)
        for name in self._shown:
            columns = self._columns_of(name)
            if columns != [name] and columns:
                out[name] = columns
        for name, members in self._groups.items():
            out[name] = [m for m in members if m in self.signals]
        return out

    def members(self, name: str) -> list[str] | None:
        """The signals of group `name`, or None if it is not a group."""
        return self.groups.get(name)

    def group(self, name: str, members: list[str] | None) -> None:
        """Name several signals to be drawn on one chart; None removes it."""
        if members is None:
            if self._groups.pop(name, None) is not None:
                self.notify("signals", signals=self.signals.names, active=list(self._active))
            return
        if name in self.signals or name == "channels" or name in self.signals.streams or (
                name in self._shown and name not in self._groups):
            raise ValueError(f"{name!r} is taken by a signal or a shown node; name the group otherwise")
        missing = [m for m in members if m not in self.signals]
        if missing:
            raise SignalError(f"no signal {', '.join(map(repr, missing))}; "
                              f"have {', '.join(self.signals.names)}")
        self._groups[name] = [self.signals.canonical(m) for m in members]
        self.notify("signals", signals=self.signals.names, active=list(self._active))

    def _columns_of(self, name: str) -> list[str]:
        return [n for n in self.signals.names if self.signals.get(n).source == f"graph:{name}"]

    # -- graphs --------------------------------------------------------

    def show(self, *nodes: Node, plot: bool = True) -> list[str]:
        """Put graph nodes where the app can see them; returns their names.

        A node computing values becomes a derived signal of that name --
        drawn by the panes, answered by the queries -- and one computing
        events is drawn over every signal pane. A node without a name is
        given one. Showing a node under a name already shown replaces it,
        which is how a changed parameter reaches the screen.
        """
        names = []
        detections = False
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(f"show() takes graph nodes, not {type(node).__name__}")
            if node.name is None:
                node = node.named(self._fresh_name(node.op))
            name = node.name
            if (name in self.signals and self.signals.get(name).kind == "channel") \
                    or self.signals.ambiguous(name):
                raise ValueError(f"{name} is a channel of the source; name the node otherwise")
            if name in self.signals.streams and len(self.signals.streams) > 1:
                raise ValueError(f"{name} is a stream; name the node otherwise")
            if name in self._groups:
                raise ValueError(f"{name} is a group; name the node otherwise")
            self._forget(name)
            columns = [] if node.output == "events" or node.op == "signal" else self._publish(node)
            self._shown[name] = node
            self._undrawn.discard(name)
            names.append(name)
            if node.output == "events":
                detections = True
            elif node.op == "signal":
                wanted = node.params.get("names") or ()
                if plot and len(wanted) == 1 and wanted[0] in self.signals:
                    self.chart(wanted[0])
            if plot and columns:
                # Components go on one chart together, as the source's do.
                self.chart(name if columns != [name] else columns[0])
        if detections:
            self.notify("detections", names=self.detection_names)
        return names

    def hide(self, name: str) -> None:
        """Stop showing a node: its signal is dropped, its events undrawn."""
        if name not in self._shown:
            raise KeyError(f"nothing shown as {name!r}; shown: {', '.join(self._shown)}")
        was_events = self._shown[name].output == "events"
        self._forget(name)
        del self._shown[name]
        self._undrawn.discard(name)
        if was_events:
            self.notify("detections", names=self.detection_names)

    def set_drawn(self, name: str, drawn: bool = True) -> None:
        """Draw a shown event node's marks, or not; it is still computed.

        Unlike `hide`, the node stays shown: counted, queryable, and
        there for the nodes built on it.
        """
        if name not in self.detection_names:
            raise KeyError(f"no events shown as {name!r}; have "
                           f"{', '.join(self.detection_names) or 'none'}")
        if drawn == (name not in self._undrawn):
            return
        if drawn:
            self._undrawn.discard(name)
        else:
            self._undrawn.add(name)
        self.notify("detections", names=self.detection_names)

    def drawn(self, name: str) -> bool:
        """Whether a shown event node's marks are drawn."""
        return name in self.detection_names and name not in self._undrawn

    @property
    def shown(self) -> dict[str, Node]:
        """Every node shown, by name."""
        return dict(self._shown)

    @property
    def detection_names(self) -> list[str]:
        return [n for n, node in self._shown.items() if node.output == "events"]

    # -- numbers ---------------------------------------------------------
    #
    # Each takes a channel, a shown name or a recipe, and a stretch of
    # time: t0 and t1 if given, else the selection, else everything. A
    # recipe is computed over everything held and then cut, so a filter or
    # a trailing measure has its history at the start of the stretch.

    def _node(self, x) -> Node:
        if isinstance(x, Node):
            return x if x.name else x.named(x.label)
        if not isinstance(x, str):
            raise TypeError(f"a channel or shown name, or a recipe; not {type(x).__name__}")
        if x in self._shown:
            return self._shown[x]
        if x in self.signals:
            return _signal(x, name=x)
        for shown in self._shown.values():         # a named step of a shown recipe
            try:
                return shown.find(x)
            except KeyError:
                pass
        if self.signals.ambiguous(x):
            self.signals.resolve(x)              # says which streams have it
        raise SignalError(f"no channel or shown name {x!r}; have "
                          f"{', '.join([*self.signals.names, *self.detection_names])}")

    def _computed(self, x):
        node = self._node(x)
        return node, self.evaluator.evaluate(node, self.window())

    def values(self, x: SignalRef, t0: float | None = None,
               t1: float | None = None) -> pd.Series | pd.DataFrame:
        """Values against time: a Series indexed by time (seconds), or a
        DataFrame with a column per component -- per frequency, for frames."""
        node, result = self._computed(x)
        if node.output == "events":
            raise TypeError(f"{node.title} gives events; ask session.events() for them")
        a, b = query.stretch(self, t0, t1)
        keep = (result.time >= a) & (result.time <= b)
        values = np.asarray(result.values)[keep]
        if values.dtype.kind in "iu":
            values = values.astype(float)
        return query.frame(result.time[keep], values, name=node.title, coords=result.coords)

    def events(self, x: SignalRef, t0: float | None = None,
               t1: float | None = None) -> pd.DataFrame:
        """The events of an events recipe (or a shown one) starting in the
        stretch: `start`, `end`, `duration` in seconds, and what else the
        recipe tells (a pattern's `times`, `gap`)."""
        node, result = self._computed(x)
        if node.output != "events":
            raise TypeError(f"{node.title} gives values, not events; make events with "
                            "(condition).events()")
        a, b = query.stretch(self, t0, t1)
        table = event_table(result)
        table = table[(table["start_time"] >= a) & (table["start_time"] <= b)]
        out = table.rename(columns={"start_time": "start", "end_time": "end"})
        drop = [c for c in ("label", "confidence", "source", "score", "timestamp")
                if c in out.columns]
        first = ["start", "end", "duration"]
        out = out.drop(columns=drop)
        return out[first + [c for c in out.columns if c not in first]].reset_index(drop=True)

    def stats(self, x: SignalRef, t0: float | None = None,
              t1: float | None = None) -> pd.Series | pd.DataFrame:
        """n, min, max, mean, rms, std and peak_to_peak over the stretch;
        a column per component for several."""
        values = self.values(x, t0, t1)
        if isinstance(values, pd.Series):
            return pd.Series(query.summarise(values.to_numpy(float)), name=values.name)
        return pd.DataFrame({c: query.summarise(values[c].to_numpy(float))
                             for c in values.columns})

    def distribution(self, x: SignalRef, t0: float | None = None,
                     t1: float | None = None, *, bins: int = 64,
                     low: float | None = None, high: float | None = None) -> pd.Series:
        """The density of the stretch's values, indexed by each bin's
        centre: a histogram whose area is 1. `low` and `high` limit the
        values counted and the bins' span; the values' own range if not given."""
        values = self.values(x, t0, t1)
        if isinstance(values, pd.DataFrame):
            raise ValueError(f"{values.columns.name}: a distribution is of one signal; "
                             "pick a component, e.g. recipe['az']")
        edges, density = query.distribution(values.to_numpy(float), bins=bins,
                                            low=low, high=high)
        centres = (edges[:-1] + edges[1:]) / 2.0
        return pd.Series(density, index=pd.Index(centres, name="value"), name=values.name)

    def _one(self, x, t0, t1):
        values = self.values(x, t0, t1)
        if isinstance(values, pd.DataFrame):
            raise ValueError(f"{values.columns.name}: this takes one signal; pick a "
                             "component, e.g. recipe['az']")
        node, result = self._computed(x)
        return values, result.rate or self.rate

    def samples(self, x: SignalRef, t0: float | None = None,
                t1: float | None = None) -> Samples:
        """A signal's values, times and sample rate over the stretch, as NumPy.

        One signal: pick a component of a recipe that has several."""
        values, rate = self._one(x, t0, t1)
        node, result = self._computed(x)
        return Samples(node.title, values.to_numpy(float), values.index.to_numpy(float),
                       float(rate), result.units)

    def _aligned(self, x, t0, t1, align, ref_hz):
        """The stretch of `x`: (values, sample rate, reference rate, name).
        Given `align`, a rate's signal or recipe, the values are resampled
        so the rate reads `ref_hz` all through -- its mean over the stretch
        if not given."""
        values, rate = self._one(x, t0, t1)
        if align is None:
            return values.to_numpy(float), rate, None, values.name
        time = values.index.to_numpy(float)
        if not len(time):
            return np.zeros(0), rate, ref_hz, values.name
        at = self._rate_at(align, time, t0, t1, "align by")
        ref = float(np.mean(at)) if ref_hz is None else float(ref_hz)
        return dsp.order_resample(values.to_numpy(float), rate, at, ref), rate, ref, values.name

    def _rate_at(self, rate, time, t0, t1, purpose):
        """`rate` -- a number, or a signal or recipe of one -- at each of `time`."""
        if isinstance(rate, (int, float, np.integer, np.floating)):
            return np.full(len(time), float(rate))
        turning = self.values(rate, *query.stretch(self, t0, t1))
        if isinstance(turning, pd.DataFrame):
            raise ValueError("a rate is one signal; pick a component")
        turning = turning.dropna()
        if not len(turning):
            raise ValueError(f"{_title(rate)} has no value in the stretch to {purpose}")
        # A rate of frames is stamped at each frame's end and says what its
        # middle was; a stretch may look ahead, so it is read there.
        when = turning.index.to_numpy(float) - self._middle_lag(self._node(rate))
        # Before its first value and after its last, the rate is held.
        return np.interp(time, when, turning.to_numpy(float))

    @staticmethod
    def _middle_lag(node: Node) -> float:
        """How long after the middle of what it measured a recipe stamps
        a value: half of each window along its first inputs."""
        lag = 0.0
        while node.inputs:
            if node.op in ops():
                lag += graph_op(node.op).timing(node.params).length_s / 2.0
            node = node.inputs[0]
        return lag

    def spectrum(self, x: SignalRef, t0: float | None = None,
                 t1: float | None = None, *, taper: str = "hann",
                 pad_to: int | None = None, align: RateRef | None = None,
                 ref_hz: float | None = None
                 ) -> pd.Series:
        """Amplitude against frequency (Hz) over the stretch, its mean
        removed. `taper` fades the ends ("hann"; "boxcar" is none);
        `pad_to` samples of zero-padding draw the bins closer together.
        `align`, a rate's signal or recipe, divides out its drift first:
        lines that move with it stand at what they are when it reads
        `ref_hz` (its mean over the stretch if not given)."""
        values, rate, _ref, name = self._aligned(x, t0, t1, align, ref_hz)
        frequency, amplitude = query.spectrum_table(values, rate, taper=taper, pad_to=pad_to)
        return pd.Series(amplitude, index=pd.Index(frequency, name="frequency_hz"),
                         name=name)

    def psd(self, x: SignalRef, t0: float | None = None,
            t1: float | None = None, *, segment: int = 1024, overlap: float = 0.5,
            taper: str = "hann", align: RateRef | None = None,
            ref_hz: float | None = None) -> pd.Series:
        """Power spectral density over the stretch, in units squared per
        Hz, indexed by frequency: Welch's average of the spectra of
        `segment`-sample pieces overlapping by `overlap`. Noise reads the
        same whatever `segment` is; a tone's power is the sum over its
        bins times their width. `align` as for `spectrum`."""
        values, rate, _ref, name = self._aligned(x, t0, t1, align, ref_hz)
        frequency, density = query.psd_table(values, rate, segment=segment,
                                             overlap=overlap, taper=taper)
        return pd.Series(density, index=pd.Index(frequency, name="frequency_hz"),
                         name=name)

    def orders(self, x: SignalRef, rate: RateRef, t0: float | None = None,
               t1: float | None = None, *, taper: str = "hann",
               pad_to: int | None = None) -> pd.Series:
        """Order tracking: the amplitude spectrum of the stretch with
        `rate` (a signal or recipe, Hz) divided out, indexed by multiples
        of it -- order 28 is the line at 28 times the rate wherever the
        rate went."""
        values, sample_rate, ref, name = self._aligned(x, t0, t1, rate, None)
        frequency, amplitude = query.spectrum_table(values, sample_rate, taper=taper,
                                                    pad_to=pad_to)
        return pd.Series(amplitude, index=pd.Index(frequency / ref, name="order"),
                         name=name)

    def cycle_average(self, x: SignalRef, rate: RateRef, t0: float | None = None,
                      t1: float | None = None, *, lock: int | None = None,
                      per_cycle: int = 720, low_hz: float | None = None,
                      high_hz: float | None = None, lock_hz: float = 4.0) -> CycleAverage:
        """The stretch's cycles averaged -- the time-synchronous average:
        each cycle of a source turning at `rate` (a number, or a signal or
        recipe of one, Hz) resampled to `per_cycle` points, and their
        mean. What repeats once a cycle stays; what does not, wind or
        voices, averages away as the square root of the cycles.

        `lock`, an order, follows the phase of the line at `lock` times
        the rate (its wander below `lock_hz`, which keep under half the
        rate), so the rate's small errors do not smear the average over
        many cycles: lock on the strongest line. The signal is
        band-passed to `low_hz`..`high_hz` first; the top is held under
        half of `per_cycle` times the slowest rate, so nothing folds when
        it is resampled."""
        values, sample_rate = self._one(x, t0, t1)
        time = values.index.to_numpy(float)
        if len(time) < 2:
            raise ValueError("the stretch holds no samples to average")
        at = self._rate_at(rate, time, t0, t1, "average by")
        top = 0.45 * per_cycle * float(np.min(at))
        high = top if high_hz is None else min(float(high_hz), top)
        high = high if high < 0.49 * sample_rate else None
        samples = np.nan_to_num(values.to_numpy(float))
        if low_hz is not None and high is not None:
            sos = dsp.design("bandpass", sample_rate, low_hz=low_hz, high_hz=high)
        elif low_hz is not None or high is not None:
            sos = dsp.design("highpass" if high is None else "lowpass", sample_rate,
                             cutoff_hz=low_hz if high is None else high)
        else:
            sos = None
        if sos is not None:
            samples = sosfiltfilt(sos, samples)
        cycles = dsp.cycle_count(values.to_numpy(float), sample_rate, at, lock=lock,
                                 lock_hz=lock_hz)
        if lock is not None and np.any(np.diff(cycles) <= 0):
            raise ValueError(f"the line at order {lock} was lost in the stretch (its phase "
                             "jumped back): lock on a stronger line, take a shorter "
                             "stretch, or leave lock off")
        rows = dsp.cut_cycles(samples, cycles, per_cycle)
        return average_cycles(rows, float(np.median(at)))

    def repetition_hz(self, x: SignalRef, t0: float | None = None,
                      t1: float | None = None, *, min_hz: float, max_hz: float,
                      low_hz: float | None = None, high_hz: float | None = None,
                      floor_hz: float = 60.0, guard_db: float = dsp.GUARD_DB,
                      align: RateRef | None = None) -> float:
        """How often the stretch repeats, between `min_hz` and `max_hz`,
        from the autocorrelation of its lines -- as `frames.repetition_hz`
        takes it of a frame -- checked against half and double the rate.
        NaN where no comb stands. `align` divides out a drift first, to
        the rate's mean."""
        values, rate, _ref, _name = self._aligned(x, t0, t1, align, None)
        frequency, amplitude = query.spectrum_table(values, rate)
        return float(dsp.repetition_rate(frequency, amplitude ** 2, min_hz, max_hz,
                                         low_hz=low_hz, high_hz=high_hz, floor_hz=floor_hz,
                                         guard_db=guard_db)[0])

    def lines(self, x: SignalRef, t0: float | None = None,
              t1: float | None = None, *, low_hz: float = 0.0, high_hz: float | None = None,
              prominence: float = 8.0, near: Sequence[float] | float | None = None,
              within: float = 1.5,
              resolution_hz: float = 1.5, segment: int | None = None,
              pad_to: int | None = None, floor_hz: float = 60.0, align: RateRef | None = None,
              ref_hz: float | None = None) -> pd.DataFrame:
        """The lines of the stretch, a row each: `hz` (refined between
        bins), `prominence_db` over their neighbours, `over_floor_db`
        over the median of the `floor_hz` around, and `level_db`.

        Listed, the peaks between `low_hz` and `high_hz` standing
        `prominence` dB out, from Welch's PSD -- pieces of `segment`
        samples, by default as many as give bins `resolution_hz` apart,
        so noise's bumps average away. Measured, given `near` (Hz), the
        strongest bin within `within` Hz of each, a row each in their
        order, from the whole stretch's spectrum (`pad_to` samples) --
        the finest it can be. `align` and `ref_hz` as for `spectrum`."""
        if near is None:
            if segment is None:
                rate = self.rate_of(x)
                segment = 2 ** int(np.round(np.log2(max(4.0, rate / resolution_hz))))
            spectrum = self.psd(x, t0, t1, segment=segment, align=align, ref_hz=ref_hz)
            power = spectrum.to_numpy(float)
        else:
            spectrum = self.spectrum(x, t0, t1, pad_to=pad_to, align=align, ref_hz=ref_hz)
            power = spectrum.to_numpy(float) ** 2
        frequency = spectrum.index.to_numpy(float)
        top = frequency[-1] if len(frequency) else 0.0
        table = dsp.spectrum_lines(frequency, power, low_hz, top if high_hz is None else high_hz,
                                   prominence=prominence, near=near, within=within,
                                   floor_hz=floor_hz)
        columns = ["hz", "prominence_db", "over_floor_db", "level_db"]
        if near is not None:
            columns = ["near", *columns]
        return pd.DataFrame(table, columns=columns)

    def spectrum_summary(self, x: SignalRef, t0: float | None = None,
                         t1: float | None = None, *, bands: Bands = (),
                         taper: str = "hann") -> pd.Series:
        """The spectrum of the stretch in a few numbers: samples,
        resolution_hz, dominant_hz, centroid_hz, spread_hz, flatness,
        rolloff_hz, and for each (low, high) in `bands` its power and its
        share of all power."""
        values, rate = self._one(x, t0, t1)
        frequency, amplitude = query.spectrum_table(values.to_numpy(float), rate, taper=taper)
        return pd.Series(query.shape_of(frequency, amplitude, len(values), taper, bands),
                         name=values.name)

    def audio(self, x: SignalRef, t0: float | None = None,
              t1: float | None = None, *, rate: int = 48000,
              speed: float = 1.0, bands: Bands | None = None,
              amp: float | Literal["auto"] | None = None) -> np.ndarray:
        """The stretch of one signal as sound: float32 samples within
        -1..1 at `rate` per second, `speed` times as fast and as high.
        16-bit samples keep their level; anything outside -1..1 is
        scaled to fit; a gap is silence. `bands`, (low, high) Hz pairs
        or `Band`s, keeps only those frequencies, scaled to be heard;
        `session.heard_bands` are the ticked ones. `amp` raises a weak
        signal by that many dB (0 to 60), or "auto" to a comfortable
        level, a limiter holding loud moments under 0.9; bands are then
        not raised on their own."""
        a, b = query.stretch(self, t0, t1)
        kept = band_ranges(bands) if bands is not None else None
        signal = self.signals.get(x) if isinstance(x, str) else None
        if signal is not None and signal.kind == "channel":
            window = self.window(a, b).home(x)
            return query.as_audio(window[x], window.rate, rate, speed=speed, bands=kept,
                                  amp=amp)
        values, source_rate = self._one(x, a, b)
        return query.as_audio(values.to_numpy(float), source_rate, rate, speed=speed,
                              bands=kept, amp=amp)

    # -- two signals ---------------------------------------------------

    def _pair(self, x, y, t0, t1):
        """x over the stretch, y at x's times, and x's rate."""
        a, b = query.stretch(self, t0, t1)
        xs, rate = self._one(x, a, b)
        margin = 2.0 / max(self.rate_of(y), 1e-9)     # a sample beyond either end
        ys = self.values(y, a - margin, b + margin)
        if isinstance(ys, pd.DataFrame):
            raise ValueError(f"{ys.columns.name}: a correlation is of one signal; pick a "
                             "component, e.g. recipe['az']")
        time, xv, yv = query.paired(xs, ys)
        if len(time) < 2:
            raise query.NoOverlap("the two signals have too little time in common there")
        return time, xv, yv, rate

    def correlation(self, x: SignalRef, y: SignalRef, t0: float | None = None,
                    t1: float | None = None, *, max_lag: Duration | None = None,
                    method: str = "phat") -> pd.Series:
        """How alike `y` is to `x` shifted by each lag, over the stretch:
        a Series indexed by lag (s), peaking at the lag `y` hears things
        after `x`. "phat" (GCC-PHAT) is sharp whatever the sound;
        "plain" is the normalised correlation, -1..1. `max_lag` in
        seconds or as text ("50ms"); all of it by default."""
        time, xv, yv, rate = self._pair(x, y, t0, t1)
        lags, values = dsp.cross_correlation(xv, yv, rate, max_lag=_seconds(max_lag),
                                             method=method)
        return pd.Series(values, index=pd.Index(lags, name="lag_s"),
                         name=f"{_title(y)} vs {_title(x)}")

    def delay(self, x: SignalRef, y: SignalRef, t0: float | None = None,
              t1: float | None = None, *, max_lag: Duration | None = None,
              method: str = "phat") -> pd.Series:
        """When `y` hears what `x` does, over the stretch: `delay` (s;
        positive when `y` is later, between samples), `peak` (the
        correlation there) and `snr` (the peak over the rest)."""
        series = self.correlation(x, y, t0, t1, max_lag=max_lag, method=method)
        lag, peak, snr = dsp.peak_lag(series.index.to_numpy(float), series.to_numpy(float))
        return pd.Series({"delay": lag, "peak": peak, "snr": snr}, name=series.name)

    def coherence(self, x: SignalRef, y: SignalRef, t0: float | None = None,
                  t1: float | None = None, *, segment: int = 256) -> pd.Series:
        """How much of `y` follows from `x` at each frequency, 0..1, over
        the stretch (Welch, over segments of `segment` samples)."""
        time, xv, yv, rate = self._pair(x, y, t0, t1)
        frequency, values = dsp.coherence(xv, yv, rate, segment=segment)
        return pd.Series(values, index=pd.Index(frequency, name="frequency_hz"),
                         name=f"{_title(y)} vs {_title(x)}")

    # -- arrays --------------------------------------------------------

    def add_array(self, name: str, array: Array) -> None:
        """Name an array, so the direction pane can offer it."""
        missing = [n for n in array.names if n not in self.signals]
        if missing:
            raise SignalError(f"no signal {', '.join(map(repr, missing))} for the array")
        self.arrays[name] = array
        self.notify("signals", signals=self.signals.names, active=list(self._active))

    def _array(self, array) -> Array:
        if isinstance(array, Array):
            return array
        try:
            return self.arrays[array]
        except KeyError:
            raise KeyError(f"no array {array!r}; have {', '.join(self.arrays) or 'none'}") \
                from None

    def _array_values(self, array: Array, t0, t1):
        """Every microphone over the stretch, on the first one's times, at
        the instants all have a value; and the first one's rate."""
        a, b = query.stretch(self, t0, t1)
        first, rate = self._one(array.names[0], a, b)
        time = first.index.to_numpy(float)
        columns = [first.to_numpy(float)]
        for name in array.names[1:]:
            margin = 2.0 / max(self.rate_of(name), 1e-9)
            other = self.values(name, a - margin, b + margin)
            if isinstance(other, pd.DataFrame):
                raise ValueError(f"{name}: one signal per microphone")
            columns.append(query.at_times(other, time))
        values = np.stack(columns, axis=1)
        keep = np.isfinite(values).all(axis=1)
        if keep.sum() < 4:
            raise query.NoOverlap("the microphones have too little time in common there")
        return values[keep], rate

    def _pair_delays(self, array: Array, values, rate, max_lag):
        reach = array.reach + 2.0 / rate if max_lag is None else _seconds(max_lag)
        delays = []
        for i, j in array.pairs:
            lags, correlation = dsp.cross_correlation(values[:, i], values[:, j], rate,
                                                      max_lag=reach)
            delays.append(dsp.peak_lag(lags, correlation)[0])
        return np.array(delays)

    def direction(self, array: str | Array, t0: float | None = None,
                  t1: float | None = None, *, max_lag: Duration | None = None) -> pd.Series:
        """Where a far sound came from, over the stretch: `azimuth_deg`
        (counter-clockwise from +x), `elevation_deg` (from the x-y plane;
        in 2-D how far off it, sign unknown), the unit vector, and the
        fit's `residual_s`, from the delay between every pair of the
        array's microphones. On one line: `axis_angle_deg` only."""
        array = self._array(array)
        values, rate = self._array_values(array, t0, t1)
        found = dsp.far_field(array.points, array.pairs,
                              self._pair_delays(array, values, rate, max_lag), array.speed)
        vector = found.pop("vector", np.full(array.dims, np.nan))
        for axis, value in zip("xyz", vector):
            found[f"u{axis}"] = float(value)
        return pd.Series(found, name="direction")

    def locate(self, array: str | Array, t0: float | None = None,
               t1: float | None = None, *, max_lag: Duration | None = None) -> pd.Series:
        """Where a near sound came from, over the stretch: `x`, `y` (and
        `z`) in the array's metres and the fit's `residual_s`, from the
        delay between every pair. Needs a microphone more than the
        dimensions."""
        array = self._array(array)
        values, rate = self._array_values(array, t0, t1)
        found = dsp.near_field(array.points, array.pairs,
                               self._pair_delays(array, values, rate, max_lag), array.speed)
        out = {axis: float(v) for axis, v in zip("xyz", found["position"])}
        out.update(residual_s=found["residual_s"], pairs=found["pairs"])
        return pd.Series(out, name="position")

    def direction_map(self, array: str | Array, t0: float | None = None,
                      t1: float | None = None, *, step_deg: float = 2.0,
                      elevation_deg: float = 0.0) -> pd.Series:
        """How strongly the array hears the stretch from each azimuth
        (steered response power, SRP-PHAT): 1 where every pair agrees,
        indexed by azimuth in degrees."""
        array = self._array(array)
        values, rate = self._array_values(array, t0, t1)
        azimuths = np.arange(0.0, 360.0, float(step_deg))
        power = dsp.srp_map(values, rate, array.points, array.speed, azimuths, elevation_deg)
        return pd.Series(power, index=pd.Index(azimuths, name="azimuth_deg"), name="power")

    # -- clocks --------------------------------------------------------

    def _recorded(self, key: str):
        for source in self._sources():
            series = getattr(source, "series", None)
            if series is not None and key in series.metadata:
                return series.metadata[key]
        return None

    def clocks(self) -> pd.DataFrame:
        """The clocks the recording declared (a session exported by srv-senselab): id,
        name, kind, hz, bits, whether it is the timeline, source and domain. Empty for
        other files."""
        clocks = self._recorded("clocks") or []
        return pd.DataFrame(clocks, columns=["id", "name", "kind", "hz", "bits", "timeline",
                                             "source", "domain"])

    def clock_observations(self) -> pd.DataFrame:
        """Every clock reading the recording holds, raw: `time` (session s), `kind`
        (device, pong, arrival), `clock`, `value` (that clock's ticks), `uncertainty_ns`,
        and `pps` and the gateway's UTC fields where the observation has them."""
        table = self._recorded("clock_observations")
        if table is None:
            return pd.DataFrame(columns=["time", "kind", "clock", "value", "uncertainty_ns"])
        return table.copy()

    def clock_of(self, stream: str) -> Clock | None:
        """The correction of a stream's clock, or None."""
        return self._stream(stream).buffer.clock

    def set_clock(self, stream: str, offset: float = 0.0, drift_ppm: float = 0.0,
                  at: float = 0.0) -> None:
        """Correct a stream's timestamps, held and to come: one stamped `t`
        happened at `t - offset - drift * (t - at)`. With no correction,
        they are put back as the source gave them."""
        buffer = self._stream(stream).buffer
        clock = None if offset == 0 and drift_ppm == 0 else Clock(
            float(offset), float(drift_ppm) * 1e-6, float(at))
        buffer.set_clock(clock)
        self.evaluator.clear()
        self._cursor = self._clamp(self._cursor)
        self.notify("data", samples=0, t_end=self.span[1])

    def estimate_sync(self, x: SignalRef, y: SignalRef, t0: float | None = None,
                      t1: float | None = None, *, window: Duration = "1s",
                      max_lag: Duration = "100ms",
                      method: str = "phat", min_snr: float = 5.0) -> SyncEstimate:
        """How late the clock of `y`'s stream is against `x`'s, from the
        delay over each `window` of the stretch -- claps, a played chirp,
        any sound both heard: offset at the windows' mean time, drift,
        the fit's residual and the windows (`time`, `delay`, `peak`,
        `snr`, `used`). Weak windows and outliers are left out."""
        a, b = query.stretch(self, t0, t1)
        width = _seconds(window)
        if width is None or width <= 0:
            raise ValueError("the window must be a positive duration")
        dsp.check_method(method)
        _seconds(max_lag)
        for signal in (x, y):
            self._one(signal, a, b)                 # one signal each, or say so now
        rows = []
        start = a
        # Half-overlapping, so a clap cut by one window's end is whole in the next.
        while start + width <= b + 1e-12:
            try:
                found = self.delay(x, y, start, start + width, max_lag=max_lag, method=method)
                rows.append({"time": start + width / 2, **found.to_dict()})
            except query.NoOverlap:
                pass
            start += width / 2
        windows = pd.DataFrame(rows, columns=["time", "delay", "peak", "snr"])
        floor = 0.25 / max(self.rate_of(y), 1e-9)
        return query.fit_sync(windows, min_snr=min_snr, floor=floor, width=width)

    def synchronize(self, x: SignalRef, y: SignalRef, t0: float | None = None,
                    t1: float | None = None, *, window: Duration = "1s",
                    max_lag: Duration = "100ms",
                    method: str = "phat", min_snr: float = 5.0,
                    min_windows: int = 3) -> SyncEstimate:
        """Estimate how late `y`'s stream's clock is against `x`'s
        (`estimate_sync`) and correct it, so the two agree on when things
        happened. Refused, leaving the clock as it was, when fewer than
        `min_windows` windows agree or the fit is off by more than a
        sample of the slower of the two."""
        stream = self.signals.stream_of(y) if isinstance(y, str) else home_of(y, self.signals)
        if stream == (self.signals.stream_of(x) if isinstance(x, str)
                      else home_of(x, self.signals)):
            raise ValueError("both signals are of one stream: one clock, nothing to align")
        estimate = self.estimate_sync(x, y, t0, t1, window=window, max_lag=max_lag,
                                      method=method, min_snr=min_snr)
        sample = 1.0 / max(min(self.rate_of(x), self.rate_of(y)), 1e-9)
        if estimate.apart < min_windows or not np.isfinite(estimate.residual) \
                or estimate.residual > sample:
            raise ValueError(f"not applied: {estimate.apart} windows apart agree (want "
                             f"{min_windows}), residual {estimate.residual * 1e6:.1f} us "
                             f"(want under {sample * 1e6:.1f}); {estimate!r}")
        step = Clock(estimate.offset, estimate.drift, estimate.at)
        current = self.clock_of(stream)
        clock = step if current is None else current.then(step)
        self.set_clock(stream, clock.offset, clock.drift * 1e6, clock.at)
        return estimate

    def _stream(self, name: str):
        try:
            return self.signals.streams[name]
        except KeyError:
            raise KeyError(f"no stream {name!r}; have {', '.join(self.signals.streams)}") from None

    def rate_of(self, x: SignalRef) -> float:
        """Samples per second of a signal, or of a recipe's result."""
        node, result = self._computed(x)
        return float(result.rate or self.rate)

    def _fresh_name(self, stem: str) -> str:
        name, suffix = stem, 1
        while name in self._shown or name in self.signals:
            suffix += 1
            name = f"{stem}{suffix}"
        return name

    def _publish(self, node: Node) -> list[str]:
        """Define the node's values as signals: one, or one per component."""
        evaluator = self.evaluator
        description = node.label

        def component(index):
            def compute(window):
                result = evaluator.evaluate(node, window)
                if len(result.shape) > (0 if index is None else 1):
                    raise ValueError(
                        f"{node.name}: {result.shape} per instant cannot be drawn "
                        "against time; pick a component with node[...]"
                    )
                values = evaluator.held(node, window)
                return values if index is None else values[:, index]
            return compute

        units = ""
        channels: list[str] = []
        try:
            home = home_of(node, self.signals)
        except SignalError:
            home = None                 # the pane that draws it will say what fails
        try:
            probe = evaluator.evaluate(node, self.window(stream=home))
            units = probe.units
            channels = probe.channels if len(probe.shape) == 1 else []
        except Exception:                                    # noqa: BLE001
            pass                    # the pane that draws it will say what fails
        if channels:
            names = [f"{node.name}.{c}" for c in channels]
            for i, column in enumerate(names):
                self.signals.define(column, component(i), units=units,
                                    description=f"{description} [{channels[i]}]", stream=home)
                self.signals.get(column).source = f"graph:{node.name}"
            return names
        self.signals.define(node.name, component(None), units=units, description=description,
                            stream=home)
        self.signals.get(node.name).source = f"graph:{node.name}"
        return [node.name]

    def _forget(self, name: str) -> None:
        for column in [n for n in self.signals.names
                       if (self.signals.get(n).source == f"graph:{name}")]:
            self.signals.drop(column)
            if column in self._active:
                self._active.remove(column)

    # -- cursor, selection, analysis window ------------------------------

    @property
    def cursor(self) -> float:
        return self._cursor

    @cursor.setter
    def cursor(self, t: float) -> None:
        self.set_cursor(t)

    def set_cursor(self, t: float, *, follow: bool = False) -> None:
        """Move the cursor. Doing so stops it following the newest sample."""
        t = self._clamp(float(t))
        if not follow:
            self._following = False
        if t == self._cursor:
            return
        self._cursor = t
        self.notify("cursor", cursor=t)

    def step_cursor(self, steps: float = 1.0, *, unit: str = "sample") -> float:
        """Move the cursor `steps` of `unit`: sample (of the fastest
        stream), window or second."""
        rate = max(s.buffer.rate for s in self.signals.streams.values())
        if unit == "sample":
            delta = steps / rate if rate else steps * 0.001
        elif unit == "window":
            delta = steps * self._analysis_window
        elif unit == "second":
            delta = steps
        else:
            raise ValueError(f"unknown unit {unit!r}: sample, window or second")
        self.set_cursor(self._cursor + delta)
        return self._cursor

    def follow_live(self) -> None:
        """Let the cursor stick to the newest sample again."""
        self._following = True
        self.set_cursor(self.span[1], follow=True)

    @property
    def following(self) -> bool:
        return self._following

    @property
    def frequency_cursor(self) -> float | None:
        """Where the frequency charts' cursor is, in Hz; None when it is not placed."""
        return self._frequency_cursor

    @frequency_cursor.setter
    def frequency_cursor(self, hz: float | None) -> None:
        self.set_frequency_cursor(hz)

    def set_frequency_cursor(self, hz: float | None) -> None:
        """Place the frequency cursor, independent of the time cursor; None clears it."""
        if hz is not None:
            hz = float(hz)
            if not math.isfinite(hz):
                raise ValueError(f"frequency cursor must be finite, not {hz}")
            hz = max(0.0, hz)
        if hz == self._frequency_cursor:
            return
        self._frequency_cursor = hz
        self.notify("frequency", frequency_cursor=hz)

    @property
    def selection(self) -> tuple[float, float] | None:
        return self._selection

    @selection.setter
    def selection(self, span: tuple[float, float] | None) -> None:
        self.set_selection(span)

    def set_selection(self, span: tuple[float, float] | None) -> None:
        if span is not None:
            t0, t1 = float(span[0]), float(span[1])
            span = (t0, t1) if t0 <= t1 else (t1, t0)
        if span == self._selection:
            return
        self._selection = span
        self.notify("selection", selection=span)

    def clear_selection(self) -> None:
        self.set_selection(None)

    # -- frequency bands -------------------------------------------------

    @property
    def bands(self) -> list[Band]:
        """The frequency bands, in the order they were added."""
        return list(self._bands)

    @property
    def heard_bands(self) -> list[tuple[float, float]]:
        """(low, high) of the ticked bands: what Listen plays through."""
        return [b.range for b in self._bands if b.on]

    def band(self, name: str) -> Band:
        for b in self._bands:
            if b.name == name:
                return b
        raise KeyError(f"no band {name!r}; there are {[b.name for b in self._bands]}")

    def add_band(self, low_hz: float, high_hz: float, name: str | None = None, *,
                 on: bool = True) -> str:
        """Add the band from `low_hz` to `high_hz`; returns its name,
        "band 1", "band 2"... unless one is given."""
        low, high = check_band(low_hz, high_hz)
        taken = {b.name for b in self._bands}
        if name is None:
            n = 1
            while f"band {n}" in taken:
                n += 1
            name = f"band {n}"
        elif not str(name).strip():
            raise ValueError("a band needs a name")
        elif name in taken:
            raise ValueError(f"there is a band {name!r} already")
        self._bands.append(Band(str(name), low, high, bool(on)))
        self._bands_changed()
        return str(name)

    def edit_band(self, name: str, *, low_hz: float | None = None,
                  high_hz: float | None = None, on: bool | None = None,
                  rename: str | None = None) -> Band:
        """Change what is given of a band; returns it as it now is."""
        old = self.band(name)
        low, high = check_band(old.low_hz if low_hz is None else low_hz,
                               old.high_hz if high_hz is None else high_hz)
        new_name = old.name if rename is None else str(rename)
        if not new_name.strip():
            raise ValueError("a band needs a name")
        if new_name != old.name and any(b.name == new_name for b in self._bands):
            raise ValueError(f"there is a band {new_name!r} already")
        new = Band(new_name, low, high, old.on if on is None else bool(on))
        if new != old:
            self._bands[self._bands.index(old)] = new
            self._bands_changed()
        return new

    def set_band_on(self, name: str, on: bool = True) -> None:
        self.edit_band(name, on=on)

    def remove_band(self, name: str) -> None:
        self._bands.remove(self.band(name))
        self._bands_changed()

    def clear_bands(self) -> None:
        if self._bands:
            self._bands.clear()
            self._bands_changed()

    def _bands_changed(self) -> None:
        self.notify("bands", bands=[b.name for b in self._bands])

    @property
    def analysis_window(self) -> float:
        return self._analysis_window

    @analysis_window.setter
    def analysis_window(self, seconds: float) -> None:
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("the analysis window must be positive")
        if seconds == self._analysis_window:
            return
        self._analysis_window = seconds
        self.notify("analysis", analysis_window=seconds)

    @property
    def active_signals(self) -> list[str]:
        return list(self._active)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def speed(self) -> float:
        return self._speed

    def play(self, start: float | None = None, *, speed: float = 1.0,
             until: float | None = None) -> None:
        """Replay from `start` (the cursor, or the selection, by default).

        Replay moves the cursor through history at `speed` times real
        time; the samples themselves are not moved or copied.
        """
        if speed <= 0:
            raise ValueError("speed must be positive")
        if start is None and self._selection is not None:
            start, until = self._selection[0], self._selection[1]
        if start is None:
            start = self._cursor
            if start >= self.span[1]:
                start = self.span[0]
        self._speed = float(speed)
        self._play_until = until
        self._playing = True
        self._following = False
        self._last_tick = self.clock()
        self._cursor = self._clamp(float(start))
        self.notify("playback", playing=True, speed=self._speed, cursor=self._cursor)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._last_tick = None
        self.notify("playback", playing=False, cursor=self._cursor)

    def toggle_play(self) -> None:
        self.pause() if self._playing else self.play()

    def tick(self, now: float | None = None) -> float:
        """Advance playback to `now`. The GUI's timer calls this; so can
        a script, with a clock of its own."""
        if not self._playing:
            return self._cursor
        now = self.clock() if now is None else float(now)
        last = self._last_tick if self._last_tick is not None else now
        self._last_tick = now
        end = self._play_until if self._play_until is not None else self.span[1]
        target = self._cursor + (now - last) * self._speed
        if target >= end:
            self._cursor = self._clamp(end)
            self.notify("cursor", cursor=self._cursor)
            self._playing = False
            self._last_tick = None
            self.notify("playback", playing=False, cursor=self._cursor, finished=True)
            return self._cursor
        self._cursor = self._clamp(target)
        self.notify("cursor", cursor=self._cursor)
        return self._cursor

    def _clamp(self, t: float) -> float:
        t0, t1 = self.span
        if t1 <= t0:
            return t1
        return min(max(t, t0), t1)

    # -- listeners -----------------------------------------------------

    def listen(self, callback: Callable[[str, dict], None]) -> Callable[[], None]:
        """Hear about every change. Returns a function that stops it."""
        self._listeners.append(callback)
        return lambda: self._listeners.remove(callback) if callback in self._listeners else None

    def notify(self, kind: str, **payload) -> None:
        if kind not in EVENTS:
            raise ValueError(f"unknown event {kind!r}; one of {', '.join(EVENTS)}")
        for listener in list(self._listeners):
            try:
                listener(kind, payload)
            except Exception:                                # noqa: BLE001
                # A broken listener must not take the session with it;
                # a pane that raises should not stop the data arriving.
                log.exception("listener %r failed on %s", listener, kind)

    def _signals_changed(self) -> None:
        self.notify("signals", signals=self.signals.names, active=list(self._active))

    # -- what is being looked at ----------------------------------------

    @property
    def context(self) -> Context:
        t0, t1 = self.span
        return Context(
            cursor=self._cursor,
            selection=self._selection,
            active_signals=list(self._active),
            analysis_window=self._analysis_window,
            dataset=self.dataset,
            t_start=t0,
            t_end=t1,
            rate=self.buffer.rate,
            playing=self._playing,
            speed=self._speed,
            following=self._following,
            running=self._running,
            frequency_cursor=self._frequency_cursor,
            streams=[s.describe() for s in self.signals.streams.values()],
        )

    # -- serving -------------------------------------------------------

    def serve(self, path: str | None = None) -> Path:
        """Answer the MCP tools on a local socket.

        Local only, read-only, and off until asked for. The path is
        returned so a tool can be pointed at it.
        """
        from ..mcp.endpoint import Endpoint

        if self._endpoint is not None:
            return self._endpoint.path
        self._endpoint = Endpoint(self, path)
        self._endpoint.start()
        return self._endpoint.path

    def close_endpoint(self) -> None:
        if self._endpoint is not None:
            self._endpoint.stop()
            self._endpoint = None

    @property
    def endpoint_path(self) -> Path | None:
        return self._endpoint.path if self._endpoint is not None else None

    def __repr__(self) -> str:
        t0, t1 = self.span
        samples = sum(len(s.buffer) for s in self.signals.streams.values())
        return (
            f"<Session {self.dataset}: {samples} samples "
            f"{t0:.6g}-{t1:.6g} s, signals {', '.join(self.signals.names)}>"
        )
