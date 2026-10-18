"""Reading and writing time series as delimited text: CSV, TSV and the like.

The separator is sniffed from the first lines (with `;`, a decimal
comma is understood), and the time column is found by its name:
milliseconds, microseconds or nanoseconds in `t_ms`, `t_us` or `t_ns`
(the last is what srv-senselab's clip export writes); seconds in
`time`, `time_s`, `t` or `timestamp` -- unless their values are too
large to be, when they are an epoch's milliseconds, microseconds or
nanoseconds -- or dates (ISO 8601). A file with no time column needs its
sample rate. Rows are put in time order, and a row with no time is
dropped.
"""

from __future__ import annotations

import csv as _csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.data import TimeSeries

__all__ = ["read_timeseries", "write_timeseries", "TIME_COLUMNS", "TEXT_SUFFIXES"]

#: Suffixes opened as delimited text.
TEXT_SUFFIXES = frozenset({".csv", ".tsv", ".tab", ".txt", ".dat"})

#: Time column names, and the seconds in one of their units.
TIME_COLUMNS = {
    "time": 1.0, "time_s": 1.0, "t": 1.0, "t_s": 1.0, "timestamp": 1.0,
    "time_ms": 1e-3, "t_ms": 1e-3,
    "time_us": 1e-6, "t_us": 1e-6,
    "time_ns": 1e-9, "t_ns": 1e-9,
}

#: Past this many seconds a clock is an epoch's, counted from its first
#: sample so float64 keeps its resolution.
_EPOCH_S = 1e6

#: A "seconds" column past these is an epoch in ns, us or ms (1e11 s is
#: three thousand years from now).
_TOO_LARGE = ((1e17, 1e-9), (1e14, 1e-6), (1e11, 1e-3))


def read_timeseries(
    path: str | Path,
    *,
    time_column: str | None = None,
    channels: Iterable[str] | None = None,
    rate: float | None = None,
    sep: str | None = None,
    **kwargs,
) -> TimeSeries:
    """A TimeSeries from delimited text with a time column and a column
    per channel; or, given `rate`, with no time column at all."""
    path = Path(path)
    try:
        found, decimal = _sniff(path)
        sep = sep or found
        frame = pd.read_csv(path, sep=sep, decimal=decimal,
                            engine="python" if sep == r"\s+" else "c")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: not a table that can be read ({exc})") from None
    frame.columns = [str(c).strip() for c in frame.columns]
    if time_column is not None and time_column not in frame.columns:
        raise KeyError(
            f"{path}: no column {time_column!r}; have {', '.join(frame.columns)}"
        )
    if time_column is None:
        time_column = next((c for c in frame.columns if c.lower() in TIME_COLUMNS), None)
    if time_column is None and not rate:
        raise KeyError(
            f"{path}: no time column ({', '.join(sorted(TIME_COLUMNS))}); "
            f"have {', '.join(frame.columns)}; give the sample rate instead"
        )
    if channels is not None:
        frame = frame[[*([time_column] if time_column else []), *channels]]

    metadata = {"source": str(path)}
    if time_column is None:
        time = np.arange(len(frame)) / float(rate)
        sample_rate = float(rate)
    else:
        unit = TIME_COLUMNS.get(time_column.lower(), 1.0)
        try:
            time, epoch = _seconds(frame[time_column], unit)
        except ValueError:
            raise ValueError(f"{path}: the time column {time_column!r} holds neither "
                             "numbers nor dates") from None
        if epoch is not None:
            metadata["epoch"] = epoch
        frame = frame.drop(columns=[time_column])
        held = ~np.isnan(time)
        order = np.argsort(time[held], kind="stable")
        time = time[held][order]
        frame = frame.loc[held].iloc[order]
        sample_rate = None
    values = frame.apply(pd.to_numeric, errors="coerce")
    if channels is None:
        # Labels and notes are not channels.
        values = values.loc[:, [c for c in values.columns
                                if values[c].notna().any() or frame[c].isna().all()]]
    if values.shape[1] == 0 and len(values):
        raise ValueError(f"{path}: no numeric columns besides time")
    kwargs.setdefault("name", path.stem)
    return TimeSeries(
        time=time,
        values=values.to_numpy(dtype=float),
        channels=[str(c) for c in values.columns],
        sample_rate=sample_rate,
        metadata={**metadata, **kwargs.pop("metadata", {})},
        **kwargs,
    )


def _seconds(column: pd.Series, unit: float) -> tuple[np.ndarray, int | float | None]:
    if column.dtype.kind not in "iufb":
        stamps = pd.to_datetime(column, utc=True, errors="raise", format="ISO8601")
        nanoseconds = stamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        valid = ~stamps.isna().to_numpy()
        first = int(nanoseconds[valid][0]) if valid.any() else 0
        seconds = (nanoseconds - first) / 1e9
        seconds[~valid] = np.nan
        return seconds, first
    raw = column.to_numpy()
    finite = raw[np.isfinite(raw.astype(float))] if len(raw) else raw
    if unit == 1.0 and len(finite):
        unit = next((u for limit, u in _TOO_LARGE if abs(float(finite[0])) > limit), 1.0)
    if raw.dtype.kind in "iu":
        first = int(raw[0]) if len(raw) else 0
        if abs(first) * unit > _EPOCH_S:
            return (raw - first).astype(float) * unit, first
        return raw.astype(float) * unit, None
    seconds = raw.astype(float) * unit
    if len(finite) and abs(float(finite[0])) * unit > _EPOCH_S:
        return seconds - float(finite[0]) * unit, float(finite[0])
    return seconds, None


def _sniff(path: Path) -> tuple[str, str]:
    """The separator, and the decimal mark."""
    with open(path, newline="", encoding="utf-8", errors="replace") as file:
        sample = file.read(16384)
    lines = sample.splitlines()[:20]
    head = "\n".join(lines)
    try:
        found = _csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except _csv.Error:
        found = None
    if found:
        comma = found == ";" and re.search(r"\d,\d", "\n".join(lines[1:])) is not None
        return found, "," if comma else "."
    if lines and len(lines[0].split()) > 1:
        return r"\s+", "."
    return ",", "."


def write_timeseries(series: TimeSeries, path: str | Path) -> Path:
    """Write it back out, time first. Provenance is not written: CSV has
    nowhere honest to put it, and a recipe records it properly."""
    path = Path(path)
    series.to_dataframe().to_csv(path, index_label="time")
    return path
