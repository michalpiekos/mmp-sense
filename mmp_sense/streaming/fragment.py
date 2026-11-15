"""A selection of the live streams, kept as files.

Each stream's frames in the selection are written as a WAV in the node's
own sample format, byte for byte, with what a workspace needs to say what
they are: the channel names, the node and its session, and the clock the
start was taken from.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.audio import write_wav
from .live import LiveStreams
from .relay import encode

__all__ = ["Fragment", "cut", "fragments_dir"]


@dataclass
class Fragment:
    path: Path
    #: What the workspace keeps with it (Recording.meta).
    meta: dict
    started_at_utc: str
    duration_s: float
    title: str


def fragments_dir() -> Path:
    """Where cut fragments wait to be saved into a workspace."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "mmp-sense" / "fragments"


def _utc_text(seconds: float) -> str:
    when = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def cut(live: LiveStreams, t0: float, t1: float, folder: Path | None = None) -> list[Fragment]:
    """The frames of every live stream between t0 and t1 on the shared axis."""
    if t1 <= t0:
        raise ValueError("the selection is empty")
    if live.epoch is None:
        raise ValueError("the streams are not on a UTC axis yet")
    folder = Path(folder) if folder is not None else fragments_dir()
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, entry in live.streams.items():
        stream = live.session.streams.get(name)
        if stream is None:
            continue
        time, values = stream.buffer.snapshot(t0, t1)
        if not len(time):
            continue
        source = entry.source
        # Frames lost in transit are a gap: each run either side of one is a file of its own.
        breaks = np.flatnonzero(np.diff(time) > 1.5 / source.rate) + 1
        for run_time, run_values in zip(np.split(time, breaks), np.split(values, breaks)):
            out.append(_write(live, name, entry, run_time, run_values, folder))
    if not out:
        raise ValueError("no live stream holds anything in the selection")
    return out


def _write(live, name, entry, time, values, folder) -> Fragment:
    source = entry.source
    decl = source.pcm.decl
    start = live.epoch + float(time[0])
    stamp = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = folder / f"{name}-{stamp}.wav"
    n = 1
    while path.exists():
        n += 1
        path = folder / f"{name}-{stamp}-{n}.wav"
    write_wav(path, encode(source.counts(values), source.format), rate=int(source.rate),
              channels=len(source.channels), fmt=source.format, valid_bits=decl.get("valid_bits"))
    utc = entry.relayed.utc_offset()
    if utc is None:
        clock = {"source": "unknown", "from": "none"}
    else:
        clock = {"source": "ntp" if utc.source == "gateway" else utc.source,
                 "uncertainty_ns": None if utc.uncertainty_s is None else int(round(utc.uncertainty_s * 1e9)),
                 "from": "gateway" if utc.source == "gateway" else "node"}
        if utc.domain and not entry.trim and not entry.by_hand:
            clock["domain"] = utc.domain               # a start moved by hand is no longer that clock's
    clock["trim_s"] = entry.trim
    clock["by_hand"] = entry.by_hand
    meta = {
        "channel_names": list(source.channels),
        "sample_format": source.format,
        "valid_bits": decl.get("valid_bits"),
        "node": entry.relayed.device,
        "relayed_session": entry.relayed.id,
        "stream": decl["name"],
        "clock": clock,
    }
    scales = [c.get("scale") for c in decl["channels"]]
    if any(scales):
        meta["scale"] = scales
        meta["units"] = [c.get("unit") for c in decl["channels"]]
    return Fragment(path, meta, _utc_text(start), len(time) / source.rate,
                    f"{entry.relayed.node} {decl['name']} {_utc_text(start)[11:19]}")
