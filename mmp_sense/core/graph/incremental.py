"""Extending a node's result by what new samples change, not computing it again.

A live window holds everything the buffer does, and the next one holds
the same samples less some at the front and plus some at the end. For a
causal op, the values it already gave for the samples both hold do not
change, so the evaluator keeps them and asks the op's `Rule` only for
the values of the new samples, and drops the ones that scrolled out.

What is kept for a live window is what one computation over the whole
stream would have given -- since the result was first computed, or last
had to be computed afresh -- over the samples the window holds: a filter
carries on through the scroll instead of starting from rest at the
window's first sample.

A series result carries the running sample number (`pos`) of each of
its values, so a consumer can find what is new to it; an events result
carries a `frontier`, before which none of its rows will change again.
An op without a rule is computed over the whole window, as a non-live
window always is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data import EventSeries
from .series import Series, hold

__all__ = ["Stale", "Tape", "Part", "Rule", "Lookback", "pointwise", "Filter",
           "Frontier", "restrict", "held"]


class Stale(Exception):
    """A result that cannot be carried on: it is computed afresh."""


class Tape:
    """An array appended to at the end and dropped from at the front.

    A view once handed out never changes: appending writes only past its
    end, and growing copies to a new array.
    """

    def __init__(self, values) -> None:
        values = np.asarray(values)
        self._data = np.empty((max(64, 2 * len(values)), *values.shape[1:]),
                              dtype=values.dtype)
        self._data[:len(values)] = values
        self._start, self._stop = 0, len(values)

    def __len__(self) -> int:
        return self._stop - self._start

    def drop(self, n: int) -> None:
        self._start = min(self._stop, self._start + max(0, int(n)))

    def append(self, values) -> None:
        values = np.asarray(values)
        if values.dtype != self._data.dtype or values.shape[1:] != self._data.shape[1:]:
            raise Stale("a value of another type or shape")
        n = len(values)
        if self._stop + n > len(self._data):
            size = len(self)
            data = np.empty((max(64, 2 * (size + n)), *self._data.shape[1:]),
                            dtype=self._data.dtype)
            data[:size] = self._data[self._start:self._stop]
            self._data, self._start, self._stop = data, 0, size
        self._data[self._stop:self._stop + n] = values
        self._stop += n

    def view(self) -> np.ndarray:
        out = self._data[self._start:self._stop]
        out.flags.writeable = False
        return out


@dataclass
class Part:
    """A node's result over a live window, and what its consumers need to
    carry on from it."""

    result: Any
    #: Series: the running sample number of each value; None if unknown.
    pos: np.ndarray | None = None
    #: Events: rows starting before it are final.
    frontier: float = -np.inf
    #: The stream whose samples `pos` numbers.
    stream: str | None = None
    #: Bumped each time the result is computed afresh rather than carried
    #: on, so what was made from an older one can tell.
    generation: int = 0


class Rule:
    """How an op's result is computed afresh and carried on.

    Both take the op's context, its inputs as Parts, its params and the
    window's `Live`; `extend` also the node's previous Part, the state
    `fresh` or the last `extend` returned, and the running number of the
    last sample it saw. Each returns (Part, state), or `extend` raises
    `Stale` to have the result computed afresh.
    """

    def fresh(self, ctx, inputs, params, live):
        raise NotImplementedError

    def extend(self, ctx, inputs, params, live, previous, state, end):
        raise Stale("no rule to carry this on")


def _new_from(pos, end: int) -> int:
    """The index of the first value past sample `end`."""
    return int(np.searchsorted(pos, end, side="right"))


def _series_tail(series: Series, start: int) -> Series:
    return replace(series, time=series.time[start:], values=series.values[start:])


def _aligned_from(other: Part, first: Part, start: int) -> Series:
    """Of another input, what `hold` needs to align it onto the first
    input's values from `start` on: on the same samples, the same stretch;
    else from its last value at or before them."""
    series = other.result
    if (other.stream == first.stream and len(other.pos) == len(first.pos) and len(first.pos)
            and other.pos[0] == first.pos[0] and other.pos[-1] == first.pos[-1]):
        return _series_tail(series, start)
    t = float(first.result.time[start])
    return _series_tail(series, max(0, int(np.searchsorted(series.time, t, side="right")) - 1))


def _carried(inputs) -> bool:
    """Whether every input is a series whose old values stay put."""
    return all(isinstance(p.result, Series) and p.pos is not None for p in inputs)


class Lookback(Rule):
    """An op whose value at an instant depends on its first input's last
    `samples` values and on nothing earlier -- a pointwise op (0), or a
    trailing window.

    Its result is on its first input's times. Any other input is aligned
    onto those by `hold`, so it is handed from its last value at or
    before the first value needed. `samples` is a number or
    `fn(ctx, series, params)` of the first input.
    """

    def __init__(self, samples: int | Callable = 0) -> None:
        self.samples = samples

    def _lookback(self, ctx, series, params) -> int:
        samples = self.samples
        return int(samples(ctx, series, params) if callable(samples) else samples)

    def fresh(self, ctx, inputs, params, live):
        first = inputs[0]
        out = ctx.evaluator.apply(ctx, [p.result for p in inputs], params)
        if not _carried(inputs) or len(out.time) != len(first.pos):
            return Part(out, None), None
        tape = Tape(out.values)
        return Part(replace(out, values=tape.view()), first.pos), tape

    def extend(self, ctx, inputs, params, live, previous, tape, end):
        first = inputs[0]
        if tape is None or not _carried(inputs) or previous.pos is None:
            raise Stale("an input whose old values may change")
        tape.drop(np.searchsorted(previous.pos, live.origin, side="left"))
        series = first.result
        new = _new_from(first.pos, end)
        template = previous.result
        if new < len(first.pos):
            start = new - self._lookback(ctx, series, params)
            if start < 0:
                raise Stale("not enough history held")
            head = _series_tail(series, start)
            others = [_aligned_from(p, first, start) for p in inputs[1:]]
            out = ctx.evaluator.apply(ctx, [head, *others], params)
            if len(out.time) != len(head.time):
                raise Stale("not on the input's times")
            tape.append(out.values[new - start:])
            template = out
        if len(tape) != len(first.pos):
            raise Stale("out of step with the input")
        return Part(replace(template, time=series.time, values=tape.view()),
                    first.pos), tape


#: The rule of an op that looks at one instant at a time.
pointwise = Lookback(0)


class Filter(Rule):
    """A recursive filter along the first input: `filter(values, rate,
    **params, state=...)` returns (values, state), and the state carries
    it from one block to the next."""

    def __init__(self, filter: Callable) -> None:
        self.filter = filter

    def _run(self, series, params, state):
        return self.filter(np.asarray(series.values, dtype=float), series.rate,
                           state=state, **params)

    @staticmethod
    def _out(series, values):
        # One value from several channels (a beam) has no channels.
        if np.ndim(values) < np.ndim(series.values):
            return series.with_values(values, coords={})
        return series.with_values(values)

    def fresh(self, ctx, inputs, params, live):
        first = inputs[0]
        values, state = self._run(first.result, params, None)
        if first.pos is None:
            return Part(self._out(first.result, values), None), None
        tape = Tape(values)
        return Part(self._out(first.result, tape.view()), first.pos), (tape, state)

    def extend(self, ctx, inputs, params, live, previous, carried, end):
        first = inputs[0]
        if carried is None or first.pos is None or previous.pos is None:
            raise Stale("no positions to carry on from")
        tape, state = carried
        tape.drop(np.searchsorted(previous.pos, live.origin, side="left"))
        new = _new_from(first.pos, end)
        if new < len(first.pos):
            values, state = self._run(_series_tail(first.result, new), params, state)
            tape.append(values)
        if len(tape) != len(first.pos):
            raise Stale("out of step with the input")
        return Part(self._out(first.result, tape.view()), first.pos), (tape, state)


def restrict(events: EventSeries, since: float) -> EventSeries:
    """The rows starting at or after `since`."""
    table = events.table
    if not len(table) or not np.isfinite(since):
        return events
    return replace(events, table=table[_starts(table) >= since].reset_index(drop=True))


def _starts(table: pd.DataFrame) -> np.ndarray:
    column = "timestamp" if "timestamp" in table.columns else "start_time"
    return table[column].to_numpy(dtype=float)


class Frontier(Rule):
    """An op of events whose rows starting at an instant depend only on
    the input rows at or after it (given the final rows before it), and
    on none more than `horizon(params)` seconds after it.

    Its rows are final before the inputs' frontier less the horizon;
    those from there on are computed again from the input rows from
    there on. `run` is `fn(ctx, inputs, params, kept)`, `kept` being the
    final rows kept; by default the op itself, which ignores them.
    """

    def __init__(self, horizon: Callable[[dict], float], run: Callable | None = None,
                 needs_history: bool = False) -> None:
        self.horizon = horizon
        self.run = run
        #: Whether rows computed again need input rows from before the
        #: window when the final part lies there.
        self.needs_history = needs_history

    def _frontier(self, inputs, params) -> float:
        horizon = float(self.horizon(params))
        if not np.isfinite(horizon):
            return -np.inf
        return min(p.frontier for p in inputs) - horizon

    def _compute(self, ctx, inputs, params, kept):
        if self.run is not None:
            return self.run(ctx, inputs, params, kept)
        return ctx.evaluator.apply(ctx, inputs, params)

    def fresh(self, ctx, inputs, params, live):
        out = self._compute(ctx, [p.result for p in inputs], params, None)
        return Part(out, frontier=self._frontier(inputs, params)), out.table

    def extend(self, ctx, inputs, params, live, previous, table, end):
        since = previous.frontier
        if table is None or not np.isfinite(since):
            raise Stale("nothing final yet")
        start = float(ctx.window.t_start)
        if self.needs_history and since < start:
            raise Stale("what is not final has scrolled out")
        # Final rows from before the window are kept while a row computed
        # again may still depend on them: one horizon.
        starts = _starts(table)
        kept = table[(starts < since) & (starts >= start - float(self.horizon(params)))]
        out = self._compute(ctx, [restrict(p.result, since) for p in inputs], params, kept)
        fresh = out.table[_starts(out.table) >= since] if len(out.table) else out.table
        pieces = [t for t in (kept, fresh) if len(t)]
        table = (pd.concat(pieces, ignore_index=True) if pieces
                 else out.table.iloc[:0]).reset_index(drop=True)
        shown = table[_starts(table) >= start].reset_index(drop=True) if len(table) else table
        return Part(replace(out, table=shown),
                    frontier=self._frontier(inputs, params)), table


def held(result: Series, time: np.ndarray, previous, live, limit: float | None = None):
    """`result` held onto each of `time` as floats, carried on from
    `previous`, the state the last call returned.

    Values at times up to `limit` (all, if None) are final and kept;
    later ones are NaN, to be given once the result reaches them.
    """
    n = len(time)
    final = n if limit is None else int(np.searchsorted(time, limit, side="right"))
    start, tape = 0, None
    if previous is not None:
        tape, done = previous
        tape.drop(live.origin - (done - len(tape)))
        start = done - live.origin
        if start < 0 or start > final or len(tape) != start:
            raise Stale("a gap")
    fresh = np.asarray(hold(result, time[start:final]), dtype=float)
    if tape is None:
        tape = Tape(fresh)
    else:
        tape.append(fresh)
    out = tape.view()
    if final < n:
        out = np.concatenate([out, np.full(n - final, np.nan)])
    return out, (tape, live.origin + final)
