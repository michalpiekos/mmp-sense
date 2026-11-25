"""A session folder, as mmp-stream's receiver writes it and srv-senselab exports it:

    <session>/session.json     the session: its clocks, streams and files
    <session>/mics.wav         a pcm stream's frames as they arrived (mics-001.wav ... past 4 GiB)
    <session>/timing.jsonl     timing points, clock observations, events, records
    <session>/chunk_*.mcap     a phone's sensors

The folder opens as the session it is: a series per WAV stream, named after it, with
its channels' names (tetrahedral's mics are S, N, W, E), on the session's time -- frames
placed by their timing points, so lost frames stay a gap in time -- and every plottable
MCAP topic. The declared clocks and every clock observation ride along in each series'
metadata, read by `Session.clocks()` and `Session.clock_observations()`.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.data import TimeSeries
from .audio import _read_wav
from .mcap import plottable_topics, read_mcap_topics, topics_in

__all__ = ["is_session_folder", "read_session_folder", "SESSION_FILE", "TIMING_FILE"]

SESSION_FILE = "session.json"
TIMING_FILE = "timing.jsonl"
CLOCK_COLUMNS = ["id", "name", "kind", "hz", "bits", "timeline", "source", "domain"]


def is_session_folder(path) -> bool:
    path = Path(path)
    if not path.is_dir() or not (path / SESSION_FILE).is_file():
        return False
    try:
        return json.loads((path / SESSION_FILE).read_text()).get("format") == "mmp-session"
    except (OSError, ValueError, AttributeError):
        return False


def _lines(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break                                       # torn by a crash
            try:
                out.append(json.loads(line))
            except ValueError:
                break
    return out


def _points(lines: list[dict], t0: int, hz: float) -> dict[str, list[tuple[int, float, bool]]]:
    """Per WAV file, its timing points: (frame in the file, session seconds, after a loss)."""
    points: dict[str, list[tuple[int, float, bool]]] = {}
    last: dict[str, dict] = {}
    for line in lines:
        if line.get("topic") != "audio":
            continue
        file = line["file"]
        prev = last.get(file)
        lost = prev is not None and line["device_frame"] - prev["device_frame"] != line["frame"] - prev["frame"]
        points.setdefault(file, []).append((int(line["frame"]), (line["t"] - t0) / hz, lost))
        last[file] = line
    for v in points.values():
        v.sort()
    return points


def _frame_times(first_frame: int, frames: int, rate: float, start_s: float, points) -> np.ndarray:
    """Session time of each frame: interpolated between the timing points around it, so time keeps
    increasing although a device's points jitter by a few ms; at the rate from the last point
    before a loss or the end; without points, linear from `start_s`."""
    k = (first_frame + np.arange(frames)).astype(float)
    if not points:
        return start_s + (k - first_frame) / rate
    at = np.array([p[0] for p in points], dtype=float)
    t = np.array([p[1] for p in points], dtype=float)
    lost = np.array([p[2] for p in points], dtype=bool)
    i = np.searchsorted(at, k, side="right") - 1
    before = i < 0
    i = np.clip(i, 0, len(at) - 1)
    out = t[i] + (k - at[i]) / rate
    j = np.minimum(i + 1, len(at) - 1)
    run = (at[j] > at[i]) & ~lost[j]
    out[run] = t[i[run]] + (k[run] - at[i[run]]) * (t[j[run]] - t[i[run]]) / (at[j[run]] - at[i[run]])
    out[before] = (start_s + (k - first_frame) / rate)[before]
    return out


def _read_counts(path: Path, decl: dict | None) -> tuple[np.ndarray, float]:
    """A WAV's frames: full scale (-1..1) for sound, else counts times each channel's scale."""
    sound = decl is None or decl.get("modality", "audio") in ("audio", "geophone")
    if sound:
        values, native = _read_wav(path)
    else:
        from scipy.io import wavfile

        native, data = wavfile.read(path)
        values = np.asarray(data, dtype=float)
        if decl.get("format") == "s24":
            values = values / 256.0                         # scipy gives 24-bit in an int32's top
        values = values.reshape(len(values), -1)
        scales = [c.get("scale") for c in decl.get("channels") or []]
        if any(s is not None for s in scales) and len(scales) == values.shape[1]:
            values = values * np.array([s if s is not None else 1.0 for s in scales])
    return np.asarray(values, dtype=float).reshape(len(values), -1), float(native)


def _file_order(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.*?)(?:-(\d{3}))?\.wav", name)
    return (m.group(1), int(m.group(2) or 0)) if m else (name, 0)


def _stream(name: str, files: list[Path], decl: dict | None, points) -> TimeSeries:
    """A WAV stream's files, in order, as one series."""
    times, rows, rate, start = [], [], None, 0.0
    for path in sorted(files, key=lambda p: _file_order(p.name)):
        values, rate = _read_counts(path, decl)
        t = _frame_times(0, len(values), rate, start, points.get(path.name, []))
        if len(t):
            start = float(t[-1]) + 1.0 / rate
        times.append(t)
        rows.append(values)
    time, values = np.concatenate(times), np.concatenate(rows)
    count = values.shape[1]
    sound = decl is None or decl.get("modality", "audio") in ("audio", "geophone")
    names = [str(c.get("name")) for c in (decl or {}).get("channels") or []]
    if len(names) != count or len(set(names)) != count:
        names = ["audio"] if count == 1 and sound else [f"ch{i}" for i in range(count)]
    units = {c.get("unit") for c in (decl or {}).get("channels") or []}
    unit = "" if sound or len(units) != 1 else (units.pop() or "")
    return TimeSeries(name=name, time=time, values=values, channels=names, units=unit, sample_rate=rate,
                      metadata={"source": str(files[0].parent), "stream": name, "files": [p.name for p in files]})


def _observations(lines: list[dict], timeline: str, t0: int, hz: float) -> pd.DataFrame:
    """Every clock reading, raw: a row per reading of each observation, at the session time of
    its timeline reading (an arrival: of the data that arrived)."""
    rows = []
    for o in lines:
        if o.get("topic") != "clock":
            continue
        readings = dict(o.get("readings") or {})
        if o.get("kind") == "arrival":
            readings = {timeline: [o["t"], 0]}
        at = readings.get(timeline)
        base = {"time": (at[0] - t0) / hz if at else np.nan, "kind": o.get("kind"), "pps": o.get("pps"),
                "receiver_utc_ns": o.get("receiver_utc_ns"), "sent_utc_ns": o.get("sent_utc_ns"),
                "received_utc_ns": o.get("received_utc_ns"), "ping_id": o.get("id")}
        for clock, (value, unc) in readings.items():
            rows.append(base | {"clock": clock, "value": value, "uncertainty_ns": unc})
    columns = ["time", "kind", "clock", "value", "uncertainty_ns", "pps", "receiver_utc_ns",
               "sent_utc_ns", "received_utc_ns", "ping_id"]
    table = pd.DataFrame(rows, columns=columns, dtype=object)
    for c in ("value", "uncertainty_ns", "receiver_utc_ns", "sent_utc_ns", "received_utc_ns", "ping_id"):
        table[c] = pd.array(list(table[c]), dtype="Int64")  # ns and ticks need all 64 bits, not a float's
    table["time"] = table["time"].astype(float)
    return table.sort_values("time", kind="stable").reset_index(drop=True)


def read_session_folder(folder) -> list[TimeSeries]:
    """The session a folder holds: a series per WAV stream and per plottable MCAP topic, on
    the session's time (seconds from its start)."""
    folder = Path(folder)
    m = json.loads((folder / SESSION_FILE).read_text())
    clocks = list(m.get("clocks") or [])
    timeline = next((c for c in clocks if c.get("timeline")), None)
    tl = m.get("timeline") or {}
    hz = float(tl.get("hz") or (timeline or {}).get("hz") or 1e9)
    t0 = tl.get("t0")
    lines = _lines(folder / TIMING_FILE)
    if t0 is None:
        t0 = min((l["t"] for l in lines if l.get("topic") == "audio"), default=0)
    points = _points(lines, t0, hz)
    decls = {s["name"]: s for s in m.get("streams") or [] if isinstance(s, dict) and "name" in s}
    wavs: dict[str, list[Path]] = {}
    mcaps: list[Path] = []
    for f in m.get("files") or []:
        path = folder / f["name"]
        if not path.is_file():
            continue
        if f.get("media_type") == "audio/wav" or path.suffix.lower() == ".wav":
            stream = f.get("stream") or _file_order(f["name"])[0]
            wavs.setdefault(stream, []).append(path)
        elif f.get("media_type") == "application/x-mcap" or path.suffix.lower() == ".mcap":
            mcaps.append(path)
    series = [_stream(name, files, decls.get(name), points) for name, files in wavs.items()]
    topics: dict[str, TimeSeries] = {}
    for path in sorted(mcaps):
        data = path.read_bytes()
        catalog = topics_in(data)
        wanted = plottable_topics(data, catalog)
        for one in read_mcap_topics(data, wanted, catalog) if wanted else []:
            epoch = one.metadata.get("epoch_ns")
            if epoch is not None:
                one.time = one.time + epoch / 1e9           # a phone's MCAP times are session ns
                one.metadata["epoch_ns"] = 0
            first = topics.get(one.name)
            if first is None or first.channels != one.channels:
                topics.setdefault(one.name, one)
                continue
            time = np.concatenate([first.time, one.time])
            order = np.argsort(time, kind="stable")
            first.time, first.values = time[order], np.concatenate([first.values, one.values])[order]
    series.extend(topics.values())
    if not series:
        raise ValueError(f"{folder}: no WAV stream or plottable MCAP topic in it")
    started = m.get("started_utc_ns")
    started_text = (datetime.datetime.fromtimestamp(started / 1e9, datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%S.%fZ") if isinstance(started, int) else None)
    shared = {"session_id": m.get("id"), "started_at_utc": started_text,
              "clocks": [{k: c.get(k) for k in CLOCK_COLUMNS} for c in clocks],
              "clock_observations": _observations(lines, timeline["name"], t0, hz) if timeline else None,
              "session_title": m.get("title"), "session_tags": list(m.get("tags") or []),
              "session_notes": m.get("notes") or ""}
    for one in series:
        one.metadata.update(shared)
    return series
