"""A recording as srv-senselab exports it: `sample export "<sql>" --out DIR --raw`
writes `DIR/manifest.json` and a folder per capture session,

    DIR/<session>/capture_session.json    the session: sensors, declared clocks
    DIR/<session>/samples/<sample>.wav    a stream's sound or counts
    DIR/<session>/samples/<sample>.mcap   timing points, clock observations, sensors

and the folder opens as the session it is: a series per WAV stream, named
after it, with its channels' names (tetrahedral's mics are S, N, W, E), on
the session's time -- frames placed by their `/audio/timing` points, so
lost frames stay a gap in time -- and an MCAP's plottable topics. The declared
clocks and every `/clock/observation` ride along in each series'
metadata, read by `Session.clocks()` and `Session.clock_observations()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.data import TimeSeries
from .audio import _read_wav
from .mcap import _collect, plottable_topics, read_mcap_topics, topics_in

__all__ = ["is_session_folder", "read_session_folder", "SESSION_FILE"]

SESSION_FILE = "capture_session.json"
TIMING_TOPIC = "/audio/timing"
CLOCK_TOPIC = "/clock/observation"


def is_session_folder(path) -> bool:
    return Path(path).is_dir() and (Path(path) / SESSION_FILE).is_file()


def _declarations(folder: Path, session_id: str) -> list[dict]:
    """Each sample's path and declaration, from the export's manifest."""
    manifest = folder.parent / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"{folder}: no manifest.json beside it; open the folder "
                         "`sample export ... --raw` wrote, as it wrote it")
    for entry in json.loads(manifest.read_text()):
        if str(entry.get("capture_session_id")) == session_id:
            return [s for s in entry.get("samples", []) if isinstance(s.get("sample"), dict)]
    raise ValueError(f"{manifest}: no capture session {session_id}")


def _timing(mcaps: list[Path]) -> dict[str, list[tuple[int, float]]]:
    """Per WAV sample id, its timing points: (first frame, session seconds)."""
    points: dict[str, list[tuple[int, float]]] = {}
    for path in mcaps:
        for _t, payload in _collect(path, [TIMING_TOPIC])[TIMING_TOPIC]:
            try:
                p = json.loads(payload)
                points.setdefault(str(p["audio_sample_id"]).lower(), []).append(
                    (int(p["first_frame"]), float(p["session_ns"]) / 1e9))
            except (ValueError, KeyError, TypeError):
                continue
    for v in points.values():
        v.sort()
    return points


def _frame_times(first_frame: int, frames: int, rate: float, start_s: float, points) -> np.ndarray:
    """Session time of each frame: the latest timing point at or before it, plus the frames
    since at the rate; without points, linear from the sample's start."""
    k = first_frame + np.arange(frames)
    if not points:
        return start_s + (k - first_frame) / rate
    at = np.array([p[0] for p in points])
    t = np.array([p[1] for p in points])
    i = np.searchsorted(at, k, side="right") - 1
    before = i < 0
    i = np.clip(i, 0, len(at) - 1)
    out = t[i] + (k - at[i]) / rate
    out[before] = (start_s + (k - first_frame) / rate)[before]
    return out


def _stream(name: str, samples: list[tuple[dict, Path]], points) -> TimeSeries:
    """A WAV stream's samples, in frame order, as one series."""
    samples.sort(key=lambda s: int(s[0].get("encoding", {}).get("first_frame", 0)))
    times, rows = [], []
    rate = None
    first = samples[0][0]
    enc = first.get("encoding", {})
    sound = first.get("modality") in ("audio", "geophone")
    for decl, path in samples:
        e = decl.get("encoding", {})
        if sound:
            values, native = _read_wav(path)
        else:
            from scipy.io import wavfile

            native, data = wavfile.read(path)
            values = np.asarray(data, dtype=float)
            if e.get("sample_format") == "s24":
                values = values / 256.0                     # scipy gives 24-bit in an int32's top
            scale = e.get("scale")
            if scale is not None:
                values = values * np.asarray(scale, dtype=float)
        values = np.asarray(values, dtype=float).reshape(len(values), -1)
        rate = float(native)
        start_s = float(decl.get("capture", {}).get("start_offset_ns", 0)) / 1e9
        times.append(_frame_times(int(e.get("first_frame", 0)), len(values), rate, start_s,
                                  points.get(str(decl.get("sample_id")).lower(), [])))
        rows.append(values)
    # Lost frames are a jump in time, at the time the next frame's timing point gives.
    time, values = np.concatenate(times), np.concatenate(rows)
    count = values.shape[1]
    names = [str(n) for n in enc.get("channel_names") or []]
    if len(names) != count or len(set(names)) != count:
        names = ["audio"] if count == 1 and sound else [f"ch{i}" for i in range(count)]
    units = enc.get("units") or []
    unit = "" if sound else (units[0] if units and len(set(units)) == 1 and enc.get("scale") is None else "")
    return TimeSeries(name=name, time=time, values=values, channels=names, units=unit, sample_rate=rate,
                      metadata={"source": str(samples[0][1].parent.parent), "stream": name,
                                "samples": [str(d.get("sample_id")) for d, _ in samples]})


def _observations(mcaps: list[Path]) -> pd.DataFrame:
    """Every clock reading, raw: a row per reading of each observation."""
    rows = []
    for path in mcaps:
        for publish_ns, payload in _collect(path, [CLOCK_TOPIC])[CLOCK_TOPIC]:
            try:
                o = json.loads(payload)
            except ValueError:
                continue
            base = {"time": publish_ns / 1e9, "kind": o.get("kind"), "pps": o.get("pps"),
                    "gateway_utc_ns": o.get("gateway_utc_ns"),
                    "gateway_send_utc_ns": o.get("gateway_send_utc_ns"),
                    "gateway_recv_utc_ns": o.get("gateway_recv_utc_ns"), "ping_id": o.get("ping_id"),
                    "stream": o.get("stream")}
            for r in o.get("readings") or []:
                rows.append(base | {"clock": r.get("clock"), "value": r.get("value"),
                                    "uncertainty_ns": r.get("uncertainty_ns")})
    columns = ["time", "kind", "clock", "value", "uncertainty_ns", "pps", "gateway_utc_ns",
               "gateway_send_utc_ns", "gateway_recv_utc_ns", "ping_id", "stream"]
    table = pd.DataFrame(rows, columns=columns, dtype=object)
    for c in ("value", "gateway_utc_ns", "gateway_send_utc_ns", "gateway_recv_utc_ns", "ping_id"):
        table[c] = pd.array(list(table[c]), dtype="Int64")  # ns and ticks need all 64 bits, not a float's
    table["time"] = table["time"].astype(float)
    return table.sort_values("time", kind="stable").reset_index(drop=True)


def read_session_folder(folder) -> list[TimeSeries]:
    """The session an export folder holds: a series per WAV stream and per plottable MCAP
    topic, on the session's time (seconds from its start)."""
    folder = Path(folder)
    session = json.loads((folder / SESSION_FILE).read_text())
    session_id = str(session.get("id") or session.get("capture_session_id"))
    entries = _declarations(folder, session_id)
    wavs: dict[str, list[tuple[dict, Path]]] = {}
    mcaps: list[Path] = []
    for entry in entries:
        decl, path = entry["sample"], folder / entry["path"]
        enc = decl.get("encoding", {})
        if not path.is_file():
            continue
        if enc.get("container") == "wav":
            stream = enc.get("stream") or decl.get("sensor_key") or decl.get("modality") or "audio"
            wavs.setdefault(str(stream), []).append((decl, path))
        elif enc.get("container") == "mcap":
            mcaps.append(path)
    points = _timing(mcaps)
    series = [_stream(name, samples, points) for name, samples in wavs.items()]
    for path in mcaps:
        data = path.read_bytes()
        catalog = topics_in(data)
        wanted = plottable_topics(data, catalog)
        for one in read_mcap_topics(data, wanted, catalog) if wanted else []:
            epoch = one.metadata.get("epoch_ns")
            if epoch is not None:
                one.time = one.time + epoch / 1e9           # MCAP times are session ns
                one.metadata["epoch_ns"] = 0
            series.append(one)
    if not series:
        raise ValueError(f"{folder}: no WAV stream or plottable MCAP topic in it")
    clocks = list((session.get("capture_config") or {}).get("clocks") or [])
    observations = _observations(mcaps)
    shared = {"capture_session_id": session_id, "started_at_utc": session.get("started_at"),
              "clocks": clocks, "clock_observations": observations}
    for one in series:
        one.metadata.update(shared)
    return series
