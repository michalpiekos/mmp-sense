"""A node: one step of an analysis, and everything it depends on.

A node is a declaration, not a computation: an op's name, its
parameters, its inputs and optionally a name of its own. Nothing is
computed until an `Evaluator` is handed a node and some data, so the
same graph runs over a live buffer, a recording and a test's array.

    accel = signal("ax", "ay", "az")
    strength = accel.highpass(2.0).magnitude().peak("60ms")
    knock = (strength > 0.7).events(refractory="145ms", name="knock")
    triple = knock.repeated(3, gap=("140ms", "450ms"), name="triple")

Nodes are immutable; changing a parameter makes a new graph
(`with_params`), and equal declarations have equal keys, which is what
lets an evaluator keep what did not change.
"""

from __future__ import annotations

import hashlib
import inspect
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np

from .ops import op as _op
from .series import Duration, Timing, format_seconds, seconds

__all__ = ["Node", "signal", "beam"]


def _freeze(value) -> Any:
    """A hashable stand-in for a parameter, equal when the values are."""
    if isinstance(value, np.ndarray):
        digest = hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest()
        return ("ndarray", value.shape, str(value.dtype), digest)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    try:
        hash(value)
    except TypeError:
        return ("object", id(value))
    return value


class Node:
    """One op applied to other nodes. See the module docstring."""

    __slots__ = ("op", "inputs", "params", "name", "_key")

    def __init__(self, op: str, inputs=(), params: dict | None = None,
                 name: str | None = None) -> None:
        spec = _op(op)
        params = dict(params or {})
        accepted = _accepts(spec.fn)
        unknown = set(params) - accepted if accepted is not None else ()
        if unknown:
            raise TypeError(
                f"{op}: no parameter(s) {', '.join(sorted(unknown))}; "
                f"takes {', '.join(sorted(accepted)) or 'none'}"
            )
        for key in spec.durations:
            if key in params:
                params[key] = seconds(params[key])
        for node in inputs:
            if not isinstance(node, Node):
                raise TypeError(f"{op}: inputs must be nodes, got {type(node).__name__}")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "params", MappingProxyType(params))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_key", None)

    def __setattr__(self, *_):
        raise AttributeError("a node is immutable; with_params() makes a changed copy")

    # -- identity ------------------------------------------------------

    @property
    def key(self) -> tuple:
        """Equal for two nodes that declare the same computation."""
        if self._key is None:
            key = (self.op, _freeze(dict(self.params)), self.name,
                   tuple(n.key for n in self.inputs))
            object.__setattr__(self, "_key", key)
        return self._key

    def __hash__(self) -> int:
        return hash(self.key)

    def same_as(self, other: "Node") -> bool:
        return isinstance(other, Node) and self.key == other.key

    def __bool__(self):
        raise TypeError(
            "a node is not true or false until it is evaluated; for a range "
            "use node.between(low, high), and & | ~ rather than and/or/not"
        )

    # -- what it is ----------------------------------------------------

    @property
    def spec(self):
        return _op(self.op)

    @property
    def output(self) -> str:
        """"series" or "events"."""
        return self.spec.output

    @property
    def label(self) -> str:
        """The op and its parameters, as graph() prints them."""
        return self.spec.text(dict(self.params))

    @property
    def title(self) -> str:
        return self.name or self.label

    @property
    def timing(self) -> Timing:
        """This node's timing, its inputs' included."""
        own = self.spec.timing(dict(self.params))
        inputs = [n.timing for n in self.inputs]
        return Timing(
            causal=own.causal and all(t.causal for t in inputs),
            stateful=own.stateful or any(t.stateful for t in inputs),
            windowed=own.windowed or any(t.windowed for t in inputs),
            length_s=own.length_s,
            delay_s=own.delay_s + max((t.delay_s for t in inputs), default=0.0),
            lookahead_s=own.lookahead_s + max((t.lookahead_s for t in inputs), default=0.0),
        )

    @property
    def latency(self) -> float:
        """Seconds from an instant to when this node can say anything
        about it, the slowest path through the graph included."""
        return self.timing.delay_s

    @property
    def causal(self) -> bool:
        return self.timing.causal

    @property
    def stateful(self) -> bool:
        return self.timing.stateful

    @property
    def windowed(self) -> bool:
        return self.timing.windowed

    def walk(self) -> Iterator["Node"]:
        """Every distinct node of the graph, inputs before what uses them."""
        seen: set = set()

        def visit(node):
            if node.key in seen:
                return
            for child in node.inputs:
                yield from visit(child)
            seen.add(node.key)
            yield node

        yield from visit(self)

    def find(self, name: str) -> "Node":
        for node in self.walk():
            if node.name == name:
                return node
        raise KeyError(f"no node named {name!r} in {self.title}")

    def graph(self) -> str:
        """The graph as a tree of text, this node at the top."""
        lines: list[str] = []
        named_seen: set = set()

        def text(node):
            if node.name and node.name != node.label:
                return f"{node.name}: {node.label}"
            return node.label

        def visit(node, prefix, last, top):
            joint = "" if top else ("└── " if last else "├── ")
            repeated = node.name is not None and node.key in named_seen
            lines.append(prefix + joint + text(node) + (" (above)" if repeated else ""))
            if repeated:
                return
            if node.name is not None:
                named_seen.add(node.key)
            child_prefix = prefix + ("" if top else ("    " if last else "│   "))
            for i, child in enumerate(node.inputs):
                visit(child, child_prefix, i == len(node.inputs) - 1, False)

        visit(self, "", True, True)
        return "\n".join(lines)

    # -- changing it ---------------------------------------------------

    def named(self, name: str) -> "Node":
        return Node(self.op, self.inputs, dict(self.params), name)

    def with_params(self, target: str | None = None, **params) -> "Node":
        """A copy of the graph with the named node's parameters changed.

        With no target, the change is to this node. Every occurrence of
        the named node is changed -- a knock used twice stays one knock.
        """
        if target is None:
            return Node(self.op, self.inputs, {**self.params, **params}, self.name)
        found = False

        def rewrite(node, memo):
            nonlocal found
            if node.key in memo:
                return memo[node.key]
            inputs = tuple(rewrite(child, memo) for child in node.inputs)
            if node.name == target:
                found = True
                new = Node(node.op, inputs, {**node.params, **params}, node.name)
            elif all(a is b for a, b in zip(inputs, node.inputs)):
                new = node
            else:
                new = Node(node.op, inputs, dict(node.params), node.name)
            memo[node.key] = new
            return new

        result = rewrite(self, {})
        if not found:
            raise KeyError(f"no node named {target!r} in {self.title}")
        return result

    # -- serialisation -------------------------------------------------

    def idents(self) -> dict[tuple, str]:
        """A short unique id per node of the graph: its name, or n<i>."""
        steps = list(self.walk())
        taken = {n.name for n in steps if n.name}
        ids: dict[tuple, str] = {}
        counter = 0
        for node in steps:
            if node.name and node.name not in ids.values():
                ids[node.key] = node.name
                continue
            while f"n{counter}" in taken:
                counter += 1
            ids[node.key] = f"n{counter}"
            taken.add(ids[node.key])
        return ids

    def to_dict(self) -> dict:
        """The graph as plain data: each node once, inputs by id."""
        ids = self.idents()
        nodes: dict[str, dict] = {}
        for node in self.walk():
            entry: dict[str, Any] = {"op": node.op}
            if node.inputs:
                entry["inputs"] = [ids[child.key] for child in node.inputs]
            if node.params:
                entry["params"] = {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in node.params.items()}
            if node.name:
                entry["name"] = node.name
            nodes[ids[node.key]] = entry
        return {"output": ids[self.key], "nodes": nodes}

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        built: dict[str, Node] = {}

        def build(ident):
            if ident not in built:
                entry = data["nodes"][ident]
                params = {k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in entry.get("params", {}).items()}
                built[ident] = cls(entry["op"], [build(i) for i in entry.get("inputs", [])],
                                   params, entry.get("name"))
            return built[ident]

        return build(data["output"])

    # -- building ------------------------------------------------------

    def apply(self, op: str, *others: "Node", name: str | None = None, **params) -> "Node":
        """This node, and any others, through a registered op."""
        return Node(op, (self, *others), params, name)

    # filters: the same number of values out as in

    def highpass(self, cutoff_hz: float, order: int = 1, *, name=None) -> "Node":
        """Keep what changes faster than `cutoff_hz`; lose the slow part.
        Order 1 is a first-order filter; more is a Butterworth that steep."""
        extra = {} if order == 1 else {"order": order}
        return self.apply("highpass", cutoff_hz=float(cutoff_hz), name=name, **extra)

    def lowpass(self, cutoff_hz: float, order: int = 1, *, name=None) -> "Node":
        """Keep what changes slower than `cutoff_hz`; lose the fast part.
        Order 1 is a first-order filter; more is a Butterworth that steep."""
        extra = {} if order == 1 else {"order": order}
        return self.apply("lowpass", cutoff_hz=float(cutoff_hz), name=name, **extra)

    def bandpass(self, low_hz: float, high_hz: float, order: int = 4, *, name=None) -> "Node":
        """Keep what lies between two frequencies (Butterworth)."""
        extra = {} if order == 4 else {"order": order}
        return self.apply("bandpass", low_hz=float(low_hz), high_hz=float(high_hz),
                          name=name, **extra)

    def bandstop(self, low_hz: float, high_hz: float, order: int = 4, *, name=None) -> "Node":
        """Lose what lies between two frequencies (Butterworth)."""
        extra = {} if order == 4 else {"order": order}
        return self.apply("bandstop", low_hz=float(low_hz), high_hz=float(high_hz),
                          name=name, **extra)

    def notch(self, freq_hz: float, q: float = 30.0, *, name=None) -> "Node":
        """Lose one frequency -- mains hum, a motor -- over `freq_hz / q`."""
        extra = {} if float(q) == 30.0 else {"q": float(q)}
        return self.apply("notch", freq_hz=float(freq_hz), name=name, **extra)

    def response(self, rate: float, frequency=None):
        """The frequency response of this recipe's filters, from its
        signal along its first inputs, at `rate`: a DataFrame by
        frequency (Hz) of `gain_db`, `phase_deg` and `group_delay_s`."""
        import pandas as pd

        from ... import dsp
        from .ops import filter_sections

        rate = float(rate)
        if frequency is None:
            frequency = np.linspace(0.0, rate / 2.0, 1025)[1:]
        gain, phase, delay = dsp.filter_response(filter_sections(self, rate), rate, frequency)
        with np.errstate(divide="ignore"):
            gain_db = 20.0 * np.log10(gain)
        return pd.DataFrame({"gain_db": gain_db, "phase_deg": np.degrees(phase),
                             "group_delay_s": delay},
                            index=pd.Index(np.asarray(frequency, float), name="frequency_hz"))

    def envelope(self, min_hz: float = 20.0, *, name=None) -> "Node":
        """How loud, instant by instant: the size of the analytic signal,
        through a causal Hilbert transformer good from `min_hz` up. It
        lags the signal by about 2 / min_hz seconds. A constant offset
        is part of the signal: high-pass it away first."""
        extra = {} if float(min_hz) == 20.0 else {"min_hz": float(min_hz)}
        return self.apply("envelope", name=name, **extra)

    def fill(self, longest: Duration | None = None, *, name=None) -> "Node":
        """Each gap (NaN) filled with the last value before it, for at most
        `longest` (for ever if not given) -- a rate measured in most frames
        then holds through the frames it was not."""
        return self.apply("fill", longest=longest, name=name)

    def abs(self, *, name=None) -> "Node":
        """Each value's size: the signal rectified."""
        return self.apply("abs", name=name)

    def magnitude(self, *, name=None) -> "Node":
        """The size of a vector: sqrt(x^2 + y^2 + ...) per instant."""
        return self.apply("magnitude", name=name)

    # trailing measures: each value is about the last `length`

    def peak(self, length: Duration, *, name=None) -> "Node":
        """The largest absolute value in the last `length`."""
        return self.apply("peak", length=length, name=name)

    def energy(self, length: Duration, *, name=None) -> "Node":
        """The sum of squares over the last `length`."""
        return self.apply("energy", length=length, name=name)

    def rms(self, length: Duration, *, name=None) -> "Node":
        """The root mean square over the last `length`."""
        return self.apply("rms", length=length, name=name)

    def zero_crossing_rate(self, length: Duration, *, name=None) -> "Node":
        """Sign changes per second over the last `length`."""
        return self.apply("zero_crossing_rate", length=length, name=name)

    # frequency: frames first, then a measure per frame

    def frames(self, length: Duration, hop: Duration | None = None, *,
               name=None) -> "Node":
        """An amplitude spectrum of the last `length`, every `hop`
        (a quarter of `length` if not given)."""
        return self.apply("frames", length=length, hop=hop, name=name)

    def delay(self, other: "Node", length: Duration, hop: Duration | None = None, *,
              max_lag: Duration | None = None, method: str = "phat", name=None) -> "Node":
        """How much later `other` hears what this does, in seconds, over
        each frame of the last `length`, every `hop` (a quarter of
        `length`); positive when `other` is later. `max_lag` bounds the
        search (half of `length`); "phat" (GCC-PHAT) or "plain"."""
        return self.apply("delay", other, length=length, hop=hop, max_lag=max_lag,
                          method=method, name=name)

    def band_power(self, low_hz: float, high_hz: float, *, name=None) -> "Node":
        """Of frames: the power between two frequencies (units squared)."""
        return self.apply("band_power", low_hz=float(low_hz), high_hz=float(high_hz),
                          name=name)

    def band_share(self, low_hz: float, high_hz: float, *, name=None) -> "Node":
        """Of frames: the share of all power between two frequencies, 0..1."""
        return self.apply("band_share", low_hz=float(low_hz), high_hz=float(high_hz),
                          name=name)

    def dominant_hz(self, *, min_hz: float = 0.0, name=None) -> "Node":
        """Of frames: the frequency of the strongest bin at or above `min_hz`."""
        return self.apply("dominant_hz", min_hz=float(min_hz), name=name)

    def centroid_hz(self, *, name=None) -> "Node":
        """Of frames: the power's centre of gravity."""
        return self.apply("centroid_hz", name=name)

    def spread_hz(self, *, name=None) -> "Node":
        """Of frames: how far the power lies from the centroid."""
        return self.apply("spread_hz", name=name)

    def flatness(self, *, name=None) -> "Node":
        """Of frames: 1 for power spread evenly, near 0 for a few bins."""
        return self.apply("flatness", name=name)

    def rolloff_hz(self, fraction: float = 0.85, *, name=None) -> "Node":
        """Of frames: the frequency below which `fraction` of the power lies."""
        return self.apply("rolloff_hz", fraction=float(fraction), name=name)

    def flux(self, *, normalize: bool = False, name=None) -> "Node":
        """Of frames: how much the spectrum grew since the previous frame.
        With `normalize`, how much its shape changed, whatever the loudness."""
        return self.apply("flux", normalize=bool(normalize), name=name)

    def line_power(self, low_hz: float, high_hz: float, *, above: float = 10.0,
                   floor_hz: float = 60.0, name=None) -> "Node":
        """Of frames: the power in lines between two frequencies -- bins
        standing more than `above` times (10 dB) over the median of those
        in the `floor_hz` around them, counted above that floor (units
        squared)."""
        return self.apply("line_power", low_hz=float(low_hz), high_hz=float(high_hz),
                          above=float(above), floor_hz=float(floor_hz), name=name)

    def line_share(self, low_hz: float, high_hz: float, *, above: float = 10.0,
                   floor_hz: float = 60.0, name=None) -> "Node":
        """Of frames: the share of the band that is lines, 0..1 -- each bin
        measured against its own floor first, so a sloping floor counts
        alike all through the band."""
        return self.apply("line_share", low_hz=float(low_hz), high_hz=float(high_hz),
                          above=float(above), floor_hz=float(floor_hz), name=name)

    def repetition_hz(self, min_hz: float, max_hz: float, *, low_hz: float | None = None,
                      high_hz: float | None = None, floor_hz: float = 60.0,
                      guard_db: float = 3.0, name=None) -> "Node":
        """Of frames: how often the sound repeats, between `min_hz` and
        `max_hz` -- the autocorrelation of the lines between `low_hz`
        (`min_hz`) and `high_hz` (40 times `max_hz`), checked against
        half and double the rate. NaN where no comb stands there."""
        extra = {k: v for k, v in (("low_hz", low_hz), ("high_hz", high_hz))
                 if v is not None}
        return self.apply("repetition_hz", min_hz=float(min_hz), max_hz=float(max_hz),
                          floor_hz=float(floor_hz), guard_db=float(guard_db), name=name,
                          **{k: float(v) for k, v in extra.items()})

    def peak_hz(self, low_hz: float, high_hz: float, *, near=None,
                within: float | None = None, name=None) -> "Node":
        """Of frames: the precise frequency of the strongest bin between
        two frequencies -- only within `within` Hz (two bins) of `near`
        if given, a number or a recipe such as `rate * 28`, so a line
        is followed while it glides."""
        extra = {} if within is None else {"within": float(within)}
        if isinstance(near, Node):
            return self.apply("peak_hz", near, low_hz=float(low_hz), high_hz=float(high_hz),
                              name=name, **extra)
        if near is not None:
            extra["near"] = float(near)
        return self.apply("peak_hz", low_hz=float(low_hz), high_hz=float(high_hz),
                          name=name, **extra)

    def align(self, rate: "Node", ref_hz: float, *, name=None) -> "Node":
        """Of frames: frames again, each spectrum's frequencies scaled by
        `ref_hz / rate` at its time, so lines that move with `rate` stand
        still -- `steady` and `line_share` then keep a comb that glides."""
        return self.apply("align", rate, ref_hz=float(ref_hz), name=name)

    def steady(self, length: Duration, *, name=None) -> "Node":
        """Of frames: frames again, each bin its smallest over the last
        `length` -- what sounded all along; what came and went is gone."""
        return self.apply("steady", length=length, name=name)

    # conditions and events

    def between(self, low: float, high: float, *, name=None) -> "Node":
        """True where low <= value <= high."""
        return self.apply("between", low=float(low), high=float(high), name=name)

    def events(self, *, refractory: Duration = 0.0, min_duration: Duration = 0.0,
               name=None) -> "Node":
        """Of a condition: an event for each stretch where it is true.
        One starting within `refractory` of the last is dropped; one
        shorter than `min_duration` too."""
        return self.apply("events", refractory=refractory, min_duration=min_duration,
                          name=name)

    def followed_by(self, other: "Node", *, gap: tuple = (0.0, None),
                    name=None) -> "Node":
        """Of events: this event, then one of `other` from gap[0] to gap[1]
        later (gap[1] None: any time later)."""
        low, high = _gap(gap)
        return self.apply("followed_by", other, min_delay=low, max_delay=high, name=name)

    def repeated(self, n: int, *, gap: tuple, name=None) -> "Node":
        """Of events: `n` of them in a row, each from gap[0] to gap[1]
        after the one before."""
        if int(n) < 2:
            raise ValueError("repeated needs n of 2 or more")
        low, high = _gap(gap)
        return self.apply("repeated", n=int(n), min_delay=low, max_delay=high, name=name)

    def lasting(self, *, min: Duration | None = None, max: Duration | None = None,
                name=None) -> "Node":
        """Of events: only those lasting from `min` to `max`."""
        return self.apply("lasting", min=min, max=max, name=name)

    def __getitem__(self, key) -> "Node":
        return self.apply("select", key=key)

    def _binary(self, op: str, kind: str, other) -> "Node":
        if isinstance(other, Node):
            return Node(op, (self, other), {"op": kind})
        return Node(op, (self,), {"op": kind, "value": float(other)})

    def __gt__(self, other):
        return self._binary("compare", "gt", other)

    def __ge__(self, other):
        return self._binary("compare", "ge", other)

    def __lt__(self, other):
        return self._binary("compare", "lt", other)

    def __le__(self, other):
        return self._binary("compare", "le", other)

    def __add__(self, other):
        return self._binary("arith", "add", other)

    def __sub__(self, other):
        return self._binary("arith", "sub", other)

    def __mul__(self, other):
        return self._binary("arith", "mul", other)

    def __truediv__(self, other):
        return self._binary("arith", "div", other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __and__(self, other: "Node") -> "Node":
        return Node("and", (self, other))

    def __or__(self, other: "Node") -> "Node":
        return Node("or", (self, other))

    def __invert__(self) -> "Node":
        return Node("not", (self,))

    def __repr__(self) -> str:
        extra = f", latency {format_seconds(self.latency)}" if self.latency else ""
        return f"<Node {self.title} ({self.output}{extra})>"


def _gap(gap) -> tuple:
    """(min, max) of a gap between events; max None for unbounded."""
    try:
        if isinstance(gap, str):
            raise TypeError
        low, high = gap
    except (TypeError, ValueError):
        raise TypeError("gap is (min, max), e.g. gap=('140ms', '450ms')") from None
    return low or 0.0, high


def _accepts(fn) -> set[str] | None:
    """The parameters an op's function takes; None when it takes any."""
    parameters = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return None
    return {p.name for p in parameters.values()
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "ctx"}


def signal(*names: str, name: str | None = None) -> Node:
    """The start of every recipe: channels by name (every channel when
    none are given), or signals already shown in the session.
    """
    return Node("signal", (), {"names": tuple(names)} if names else {}, name)


def beam(array, azimuth_deg: float, elevation_deg: float = 0.0, *,
         name: str | None = None) -> Node:
    """What `array` hears from one direction: its microphones' signals,
    each held back so a sound from there lines up in all, averaged
    (delay and sum). Its microphones must be channels of one stream.
    Azimuth is degrees counter-clockwise from +x, elevation from the
    x-y plane; it lags by the array's widest delay, and half a
    fractional-delay filter more."""
    params = {"positions": array.frozen(), "speed": array.speed,
              "azimuth_deg": float(azimuth_deg)}
    if elevation_deg:
        params["elevation_deg"] = float(elevation_deg)
    return Node("beam", (signal(*array.names),), params, name)
