"""The semantic tools, named once.

The MCP server and the local endpoint both work from this table, so a
tool call and a Python call are the same question asked twice. Every
tool is read-only: it looks at what the session holds and computes from
it, and none of them moves the cursor, changes a signal or runs code the
caller supplies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = ["TOOLS", "ToolSpec", "call", "describe_tools", "jsonable"]


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    params: dict[str, str] = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": dict(self.params),
        }


def jsonable(value):
    """Plain JSON: NumPy scalars become numbers, NaN becomes null."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


TOOLS: dict[str, ToolSpec] = {}


def _tool(name: str, description: str, **params):
    def wrap(fn):
        TOOLS[name] = ToolSpec(name, description, fn, params)
        return fn

    return wrap


def _stretch(session, t0, t1) -> list:
    from ..core.query import stretch

    return list(stretch(session, t0, t1))


@_tool("get_current_context",
       "What the user is looking at: cursor, frequency cursor, selection, the time held, the "
       "streams, the shown signals and detectors, and whether samples are "
       "still arriving.")
def _context(session):
    return session.context.to_dict() | {"detectors": session.detection_names}


@_tool("list_signals", "Every signal: the streams' channels and what the user "
       "has shown, with units and the stream each is of. With several streams a "
       "channel is named stream:channel.")
def _signals(session):
    return {"dataset": session.dataset,
            "streams": [s.describe() for s in session.streams.values()],
            "signals": [session.signals.get(n).describe() for n in session.signals.names],
            "detectors": session.detection_names}


@_tool("get_statistics", "n, min, max, mean, rms, std and peak_to_peak of one "
       "signal. Without t0/t1: the user's selection, else everything held.",
       signal="signal name", t0="start in seconds", t1="end in seconds")
def _statistics(session, signal: str, t0=None, t1=None):
    stats = session.stats(signal, t0, t1)
    return {"signal": signal, "stretch": _stretch(session, t0, t1),
            "stats": stats.to_dict()}


@_tool("get_values",
       "The shape of a signal: a min/max envelope of at most `points` buckets, "
       "so a transient narrower than a bucket still shows.",
       signal="signal name", t0="start in seconds", t1="end in seconds",
       points="how many buckets")
def _values(session, signal: str, t0=None, t1=None, points=200):
    values = session.values(signal, t0, t1)
    if not isinstance(values, pd.Series):
        raise ValueError(f"{signal} has components {list(values.columns)}; ask for one")
    x, time = values.to_numpy(float), values.index.to_numpy(float)
    edges = np.linspace(0, len(x), max(1, min(int(points), len(x))) + 1).astype(int)
    buckets = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
    return {"signal": signal, "stretch": _stretch(session, t0, t1),
            "samples_per_point": len(x) / max(1, len(buckets)),
            "t": [time[(lo + hi - 1) // 2] for lo, hi in buckets],
            "min": [np.nanmin(x[lo:hi]) if np.isfinite(x[lo:hi]).any() else None
                    for lo, hi in buckets],
            "max": [np.nanmax(x[lo:hi]) if np.isfinite(x[lo:hi]).any() else None
                    for lo, hi in buckets]}


@_tool("get_events", "The events of a shown detector: start, end, duration.",
       detector="detector name", t0="start in seconds", t1="end in seconds",
       limit="how many at most")
def _events(session, detector: str, t0=None, t1=None, limit=200):
    table = session.events(detector, t0, t1)
    return {"detector": detector, "stretch": _stretch(session, t0, t1),
            "count": len(table), "events": table.head(int(limit)).to_dict("records")}


@_tool("get_spectrum_summary",
       "The spectrum of a stretch in a few numbers: resolution, dominant "
       "frequency, centroid, spread, flatness, rolloff, and the power and "
       "share of each band asked for.",
       signal="signal name", t0="start in seconds", t1="end in seconds",
       bands="[[low_hz, high_hz], ...]")
def _spectrum(session, signal: str, t0=None, t1=None, bands=None):
    summary = session.spectrum_summary(signal, t0, t1,
                                       bands=[tuple(b) for b in (bands or [])])
    return {"signal": signal, "stretch": _stretch(session, t0, t1),
            "summary": summary.to_dict()}


@_tool("get_delay",
       "How much later signal `y` hears what signal `x` does, over a stretch: "
       "delay in seconds (positive when y is later), the correlation peak and "
       "its snr. Without t0/t1: the user's selection, else everything held.",
       x="signal name", y="signal name", t0="start in seconds", t1="end in seconds",
       max_lag="longest lag to look for, in seconds", method="phat or plain")
def _delay(session, x: str, y: str, t0=None, t1=None, max_lag=None, method="phat"):
    found = session.delay(x, y, t0, t1, max_lag=max_lag, method=method)
    return {"x": x, "y": y, "stretch": _stretch(session, t0, t1), **found.to_dict()}


@_tool("get_clocks", "The clocks the recording declared (a session exported by "
       "srv-senselab): id, name, kind, hz, bits, timeline, source, domain.")
def _clocks(session):
    return {"clocks": jsonable(session.clocks().to_dict("records"))}


@_tool("get_clock_observations",
       "The recording's clock readings, raw, a row per reading: time (session s), kind "
       "(device, pong, arrival), clock, value (its ticks), uncertainty_ns, pps and the "
       "gateway's UTC fields. Optionally only one `clock`, and at most `limit` rows.",
       clock="a clock's name", limit="most rows to return")
def _clock_observations(session, clock=None, limit=1000):
    table = session.clock_observations()
    if clock is not None:
        table = table[table["clock"] == clock]
    rows = table.head(int(limit)).astype(object).where(table.head(int(limit)).notna(), None)
    return {"rows": len(table), "observations": jsonable(rows.to_dict("records"))}


def describe_tools() -> list[dict]:
    return [spec.describe() for spec in TOOLS.values()]


def call(session, name: str, args: dict | None = None) -> dict:
    """Run one tool against a session and return a JSON-safe result."""
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"no tool {name!r}; have {', '.join(TOOLS)}")
    return jsonable(spec.fn(session, **(args or {})))
