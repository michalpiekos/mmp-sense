"""What a node computes: values against time, with their timing."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ..data import TimeSeries

__all__ = ["Series", "Timing", "seconds", "hold"]

_DURATION = re.compile(r"^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(us|ms|s|min)?\s*$")
_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, None: 1.0}


#: A duration: seconds, or text such as "80ms" or "1.5s".
Duration = float | str


def seconds(value: Duration | None) -> float | None:
    """A duration in seconds, from a number or text such as "80ms"."""
    if value is None:
        return None
    if isinstance(value, str):
        match = _DURATION.match(value)
        if not match:
            raise ValueError(f"not a duration: {value!r}; try 0.08 or '80ms'")
        return float(match.group(1)) * _SCALE[match.group(2)]
    return float(value)


def format_seconds(value: float) -> str:
    if value is None or math.isinf(value):
        return "unbounded" if value is not None else "-"
    if value and abs(value) < 1.0:
        return f"{value * 1000:g}ms"
    return f"{value:g}s"


@dataclass(frozen=True)
class Timing:
    """How a node's values relate to the samples they came from.

    `length_s` is how much input one value summarises, `delay_s` how long
    after an instant the result about it is known (a trailing window of
    W knows about an onset only W later at worst), and `lookahead_s` how
    much of the future a value uses -- zero for anything causal.
    """

    causal: bool = True
    stateful: bool = False
    windowed: bool = False
    length_s: float = 0.0
    delay_s: float = 0.0
    lookahead_s: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "causal": self.causal,
            "stateful": self.stateful,
            "windowed": self.windowed,
            "length_s": self.length_s,
            "delay_s": self.delay_s,
            "lookahead_s": self.lookahead_s,
        }


@dataclass
class Series:
    """Values against time, of any shape per timestamp.

    `values` is (n, ...): a scalar per instant is (n,), a vector (n, k),
    a spectrum (n, bins). `coords` names what the trailing axes mean --
    {"channel": [...]} or {"frequency": array} -- so a consumer can tell a
    spectrum from three accelerometer axes. The time of a windowed value
    is the time of the last sample in its window.
    """

    time: np.ndarray
    values: np.ndarray
    rate: float = 0.0
    units: str = ""
    name: str = ""
    coords: dict[str, Any] = field(default_factory=dict)
    timing: Timing = field(default_factory=Timing)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.values = np.asarray(self.values)
        if self.values.shape[:1] != self.time.shape:
            raise ValueError(
                f"{self.name or 'series'}: {self.values.shape[0]} values for "
                f"{len(self.time)} timestamps"
            )

    def __len__(self) -> int:
        return len(self.time)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one value: () for a scalar."""
        return tuple(self.values.shape[1:])

    @property
    def is_bool(self) -> bool:
        return self.values.dtype == bool

    @property
    def channels(self) -> list[str]:
        return [str(c) for c in self.coords.get("channel", [])]

    def aligned(self, time: np.ndarray) -> np.ndarray:
        """These values at each of `time`, by the causal rule `hold`."""
        return hold(self, time)

    def to_timeseries(self) -> TimeSeries:
        """As a TimeSeries, for a scalar or a vector series."""
        if len(self.shape) > 1:
            raise ValueError(f"{self.name}: {self.shape} per instant is not a time series")
        channels = self.channels or ([self.name or "value"] if not self.shape
                                     else [f"{self.name}[{i}]" for i in range(self.shape[0])])
        return TimeSeries(
            name=self.name or "series", time=self.time,
            values=self.values.astype(float), channels=channels,
            units=self.units, sample_rate=self.rate or None,
        )

    def to_frame(self) -> pd.DataFrame:
        return self.to_timeseries().to_dataframe()

    def with_values(self, values, **changes) -> "Series":
        return replace(self, values=values, **changes)

    def __repr__(self) -> str:
        span = f"{self.time[0]:.6g}-{self.time[-1]:.6g} s" if len(self.time) else "empty"
        return (f"<Series {self.name or '?'} {span}, {len(self.time)} x "
                f"{self.shape or 'scalar'} {self.values.dtype}>")


def hold(series: Series, time: np.ndarray) -> np.ndarray:
    """The series' values at `time`, each the latest at or before it.

    Causal alignment: nothing is interpolated from a value not yet known.
    Before the first value there is none, so a float reads NaN and a
    boolean false.
    """
    time = np.asarray(time, dtype=float)
    if len(series.time) == len(time) and np.array_equal(series.time, time):
        return series.values
    idx = np.searchsorted(series.time, time, side="right") - 1
    missing = idx < 0
    if len(series.time) == 0:
        shape = (len(time), *series.shape)
        return (np.zeros(shape, dtype=bool) if series.is_bool
                else np.full(shape, np.nan))
    out = series.values[np.clip(idx, 0, None)]
    if missing.any():
        out = out.copy()
        if series.is_bool:
            out[missing] = False
        else:
            out = out.astype(float)
            out[missing] = np.nan
    return out
