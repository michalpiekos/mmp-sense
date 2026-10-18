"""Running a graph over some data, keeping what did not change.

One evaluator serves the window, the console and a backtest alike. It
remembers each node's result by the node's key and the data's key, so
two panes and an overlay that all want `energy` this tick compute it
once, and a threshold changed on stopped data recomputes the comparison
and what depends on it -- not the filter and the windows beneath it.

Over a live window -- all a buffer holds, as a tick draws it -- a result
is not computed again but carried on to the new samples, by the op's
`incremental` rule; see incremental.py.

A node is of the stream its first input is of, down to the signal it
starts from, and is computed on that stream's times. A node reading
several streams sees each up to where the earliest of them ends (a
`View`), so its values are final when given, live as over a recording.
"""

from __future__ import annotations

import itertools
import threading
import time as _time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..data import EventSeries
from ..signals import SignalError, View
from .incremental import Part, Stale, held
from .node import Node
from .ops import REGISTRY
from .series import Series, hold

__all__ = ["Evaluator", "GraphError", "RunResult", "run", "streams_of", "home_of"]


_GENERATIONS = itertools.count(1)

#: A live result not asked for over this many new windows is forgotten.
KEEP_WINDOWS = 100


@dataclass
class _Live:
    part: Any
    state: Any
    serial: int
    epoch: int
    end: int
    data_key: tuple
    used: int


class GraphError(Exception):
    """A node that could not be computed, named along with the reason."""


def _leaf_streams(node: Node, table) -> set[str]:
    names = node.params.get("names") or ()
    if not names:
        return {table.first}
    return {table.stream_of(n) for n in names}


def streams_of(node: Node, table) -> frozenset[str]:
    """Every stream a node reads."""
    out: set[str] = set()
    for step in node.walk():
        if step.op == "signal":
            out |= _leaf_streams(step, table)
    return frozenset(out)


def home_of(node: Node, table) -> str:
    """The stream a node is of: its first input's, down to a signal."""
    while node.inputs:
        node = node.inputs[0]
    found = _leaf_streams(node, table)
    if len(found) > 1:
        raise SignalError(f"{node.title} reads several streams ({', '.join(sorted(found))}); "
                          "a signal() is of one -- combine streams with arithmetic")
    return found.pop()


@dataclass
class OpContext:
    """What an op is handed besides its inputs' results."""

    node: Node
    window: Any
    evaluator: "Evaluator"

    @property
    def name(self) -> str:
        return self.node.name or self.node.op

    @property
    def label(self) -> str:
        return self.node.title

    @property
    def inputs(self) -> tuple[Node, ...]:
        return self.node.inputs


class Evaluator:
    """Computes nodes over windows, remembering recent results."""

    def __init__(self, keep: int = 4) -> None:
        self.keep = int(keep)
        self._stores: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.RLock()
        #: How many times each node key has been computed.
        self.computed: Counter = Counter()
        #: Seconds each node key took, the last time it was computed.
        self.seconds: dict[tuple, float] = {}
        self._live: dict[tuple, _Live] = {}
        self._windows = 0
        self._last_window = None

    def count(self, node: Node) -> int:
        return self.computed[node.key]

    def clear(self) -> None:
        with self._lock:
            self._stores.clear()
            self._live.clear()

    def _store(self, view) -> dict:
        key = view.key
        if key is None:
            stores = getattr(view, "_graph_results", None)
            if stores is None:
                stores = view._graph_results = {}
            return stores.setdefault(REGISTRY["version"], {})
        key = (key, REGISTRY["version"])
        store = self._stores.get(key)
        if store is None:
            store = self._stores[key] = {}
            while len(self._stores) > self.keep:
                self._stores.popitem(last=False)
        else:
            self._stores.move_to_end(key)
        return store

    def view(self, node: Node, window) -> View:
        """The windows `node` reads, taken with `window`, which stands for
        its own stream."""
        streams = streams_of(node, window.table) or frozenset({window.stream})
        views = window.__dict__.setdefault("_views", {})
        found = views.get(streams)
        if found is None:
            found = views[streams] = View({s: window.of(s) for s in streams})
        return found

    def evaluate(self, node: Node, window):
        """The node's result over the window: a Series or an EventSeries."""
        with self._lock:
            view = self.view(node, window)
            if view.live:
                return self._carry(node, view, {})[0].result
            return self._evaluate(node, view, self._store(view), {})[0]

    def held(self, node: Node, window) -> np.ndarray:
        """The node's values at each of the window's times, as floats: the
        latest at or before each, as `hold` has it. Past where a node
        reading several streams has values, NaN."""
        with self._lock:
            live = getattr(window, "live", None)
            view = self.view(node, window)
            if live is None or not view.live:
                home = home_of(node, window.table)
                return _held_whole(self.evaluate(node, window), window.time,
                                   _limit(view, home, window.stream))
            part, key = self._carry(node, view, {})
            result = part.result
            numbers = part.pos
            limit = _limit(view, part.stream, window.stream)
            if numbers is None:                 # its old values may change
                return _held_whole(result, window.time, limit)
            if (view.sync is None and part.stream == window.stream
                    and len(numbers) == live.length
                    and (not len(numbers) or (numbers[0] == live.origin and numbers[-1] == live.end))
                    and result.values.dtype == float):
                return result.values
            key = ("held", key, window.stream)
            stamp = (window.data_key, view.key, part.generation)
            entry = self._live.get(key)
            if entry is not None and entry.data_key == stamp:
                entry.used = self._windows
                return entry.part
            values = None
            if (entry is not None and entry.data_key[2] == part.generation
                    and live.follows(entry.serial, entry.epoch, entry.end)):
                try:
                    values, state = held(result, window.time, entry.state, live, limit)
                except Stale:
                    values = None
            if values is None:
                values, state = held(result, window.time, None, live, limit)
            self._live[key] = _Live(values, state, live.serial, live.epoch, live.end,
                                    stamp, self._windows)
            return values

    def apply(self, ctx: "OpContext", inputs, params: dict):
        """The op of `ctx.node` over these inputs, as a rule asks for it."""
        return ctx.node.spec.fn(ctx, *inputs, **params)

    def _seen(self, view) -> None:
        # One tick's views share their windows' frame; it is what counts as a window.
        first = next(iter(view.windows.values()))
        mark = first.frame.serial if first.frame is not None else view.key
        if mark == self._last_window:
            return
        self._last_window = mark
        self._windows += 1
        for key in [k for k, e in self._live.items() if e.used < self._windows - KEEP_WINDOWS]:
            del self._live[key]

    def _carry(self, node: Node, view: View, keys: dict):
        """(Part, key): the node's result over a live view, carried on
        from its last if that can be done, else computed afresh."""
        known = keys.get(node.key)
        if known is not None:
            return known
        children = [self._carry(child, view, keys) for child in node.inputs]
        stream = children[0][0].stream if children else home_of(node, _table(view))
        window = view.of(stream)
        spec = node.spec
        params = dict(node.params)
        extra = spec.depends(window, **params) if spec.depends else None
        key = (node.key, extra, tuple(k for _, k in children), REGISTRY["version"], view.sync)
        self._seen(view)
        entry = self._live.get(key)
        if entry is not None and entry.data_key == view.key:
            entry.used = self._windows
            keys[node.key] = (entry.part, key)
            return keys[node.key]
        live = window.live
        ctx = OpContext(node, window, self)
        inputs = [part for part, _ in children]
        rule = spec.incremental
        started = _time.perf_counter()
        try:
            part = None
            if rule is None:
                part, state = Part(self.apply(ctx, [p.result for p in inputs], params)), None
            elif entry is not None and live.follows(entry.serial, entry.epoch, entry.end):
                try:
                    part, state = rule.extend(ctx, inputs, params, live,
                                              entry.part, entry.state, entry.end)
                    part.generation = entry.part.generation
                except Stale:
                    part = None
            if part is None:
                part, state = rule.fresh(ctx, inputs, params, live)
                part.generation = next(_GENERATIONS)
            part.result = _stamp(node, part.result)
            part.stream = stream
        except GraphError:
            self._live.pop(key, None)
            raise
        except Exception as exc:                              # noqa: BLE001
            self._live.pop(key, None)
            raise GraphError(f"{node.title}: {exc}") from exc
        self.seconds[node.key] = _time.perf_counter() - started
        self.computed[node.key] += 1
        self._live[key] = _Live(part, state, live.serial, live.epoch, live.end,
                                view.key, self._windows)
        keys[node.key] = (part, key)
        return keys[node.key]

    def _evaluate(self, node: Node, view: View, store: dict, keys: dict):
        """(result, key): the key is the node's with whatever its op and
        its inputs' ops depend on besides, so a signal defined again is
        computed again and nothing else is."""
        known = keys.get(node.key)
        if known is not None:
            return store[known[0]], known[0], known[1]
        evaluated = [self._evaluate(child, view, store, keys) for child in node.inputs]
        stream = evaluated[0][2] if evaluated else home_of(node, _table(view))
        window = view.of(stream)
        spec = node.spec
        extra = spec.depends(window, **dict(node.params)) if spec.depends else None
        key = (node.key, extra, tuple(k for _, k, _ in evaluated), view.sync)
        keys[node.key] = (key, stream)
        found = store.get(key)
        if found is not None:
            return found, key, stream
        started = _time.perf_counter()
        try:
            result = spec.fn(OpContext(node, window, self),
                             *[r for r, _, _ in evaluated], **dict(node.params))
        except GraphError:
            raise
        except Exception as exc:                              # noqa: BLE001
            raise GraphError(f"{node.title}: {exc}") from exc
        self.seconds[node.key] = _time.perf_counter() - started
        self.computed[node.key] += 1
        result = _stamp(node, result)
        store[key] = result
        return result, key, stream

    def counts(self) -> Counter:
        with self._lock:
            return Counter(self.computed)


def _table(view: View):
    return next(iter(view.windows.values())).table


def _limit(view: View, home: str | None, target: str) -> float | None:
    """Up to when a result of stream `home`, read through `view`, is final
    at times of stream `target`: everywhere on its own stream's times
    alone, else up to the newest sample it was computed from."""
    if view.sync is None and home == target:
        return None
    window = view.windows.get(home)
    if window is None or window.empty:
        return -np.inf
    return window.t_end


def _held_whole(result, time, limit: float | None) -> np.ndarray:
    values = np.asarray(hold(result, time), dtype=float)
    if limit is not None:
        values = np.array(values, dtype=float)
        values[np.asarray(time) > limit] = np.nan
    return values


def _stamp(node: Node, result):
    wanted = Series if node.output == "series" else EventSeries
    if not isinstance(result, wanted):
        raise GraphError(
            f"{node.title}: op {node.op!r} returned {type(result).__name__}, "
            f"not the {wanted.__name__} it declares"
        )
    if isinstance(result, Series):
        return replace(result, name=node.title, timing=node.timing)
    return replace(result, name=node.title)


@dataclass
class RunResult:
    """A graph run over some data, with every step of it kept."""

    node: Node
    output: Any
    intermediates: dict[str, Any] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    execution_metadata: dict = field(default_factory=dict)

    @property
    def events(self) -> EventSeries:
        if isinstance(self.output, EventSeries):
            return self.output
        return EventSeries(name=self.node.title)

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def timestamps(self) -> np.ndarray:
        table = self.events.table
        column = "timestamp" if "timestamp" in table.columns else "start_time"
        return table[column].to_numpy(dtype=float)

    def __getitem__(self, which):
        """An intermediate result, by node name or id, or by the node."""
        if isinstance(which, Node):
            which = self.node.idents()[which.key]
        return self.intermediates[which]

    def summary(self) -> str:
        meta = self.execution_metadata
        nodes = meta.get("nodes", {})
        computed = sum(1 for n in nodes.values() if n.get("computed"))
        return (f"{self.node.title}: {self.count} events over "
                f"{meta.get('t_start', 0):.6g}-{meta.get('t_end', 0):.6g} s "
                f"({meta.get('samples', 0)} samples; {computed} of {len(nodes)} "
                f"nodes computed, {1000 * meta.get('seconds', 0):.1f} ms)")


def run(node: Node, data, evaluator: Evaluator | None = None) -> RunResult:
    """Evaluate `node` over `data`: a Window, or anything with `.window()`
    (a Session, a SignalTable), keeping every intermediate result."""
    window = data if hasattr(data, "time") and hasattr(data, "table") else data.window()
    evaluator = evaluator or Evaluator()
    before = evaluator.counts()
    started = _time.perf_counter()
    output = evaluator.evaluate(node, window)
    elapsed = _time.perf_counter() - started

    ids = node.idents()
    after = evaluator.counts()
    intermediates, nodes = {}, {}
    for step in node.walk():
        ident = ids[step.key]
        intermediates[ident] = evaluator.evaluate(step, window)
        nodes[ident] = {
            "op": step.op,
            "label": step.label,
            "output": step.output,
            "computed": after[step.key] > before[step.key],
            "seconds": evaluator.seconds.get(step.key, 0.0),
            **step.timing.describe(),
        }
    return RunResult(
        node=node,
        output=output,
        intermediates=intermediates,
        parameters=node.to_dict(),
        execution_metadata={
            "samples": len(window),
            "rate": window.rate,
            "t_start": window.t_start,
            "t_end": window.t_end,
            "seconds": elapsed,
            "latency_s": node.latency,
            "causal": node.causal,
            "nodes": nodes,
        },
    )
