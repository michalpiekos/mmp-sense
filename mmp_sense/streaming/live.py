"""Relayed streams in a session, on one time axis.

Each stream keeps its node's own times. Its clock correction (the session's
`set_clock`) puts it on a shared axis: seconds from `epoch`, a whole UTC
second near the first stream's start. The correction follows what the stream
says about UTC (`RelayStream.utc_offset`) and a trim set by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.buffer import MAX_CAPACITY
from .relay import RelaySource, RelayStream

__all__ = ["LiveStreams", "Live", "HISTORY_S"]

#: Seconds of each stream a session keeps, unless told otherwise, and at most.
HISTORY_S = 120.0
MAX_HISTORY_S = 600.0
#: A correction is redone when the estimate moves this much, or a quarter of its uncertainty.
RESET_S = 0.002


@dataclass
class Live:
    """One stream of the session that a relay feeds."""

    name: str
    source: RelaySource
    #: Seconds added by hand to where the stream's own UTC puts it.
    trim: float = 0.0
    applied: float | None = None
    #: The correction set here; another one on the stream means it was set by hand, and is kept.
    clock: object = None
    by_hand: bool = False

    @property
    def relayed(self) -> RelayStream:
        return self.source.relayed


def _name(relayed: RelayStream, source: RelaySource, several: bool, taken) -> str:
    stem = f"{relayed.node}-{relayed.id[:4]}" + (f"-{source.name}" if several else "")
    stem = "_".join(stem.replace(":", "_").replace("/", "_").split())
    name, n = stem, 1
    while name in taken:
        n += 1
        name = f"{stem}{n}"
    return name


class LiveStreams:
    """The relayed streams a session holds, and their alignment."""

    def __init__(self, session) -> None:
        self.session = session
        self.streams: dict[str, Live] = {}
        #: UTC seconds of the shared axis's zero; set by the first stream that knows UTC.
        self.epoch: float | None = None

    def open(self, relay: str, listings: list[dict], *,
             history_s: float = HISTORY_S, replace: bool = True) -> list[str]:
        """Watch these relayed sessions: a stream each of their PCM streams.
        `replace` drops what the session held before (a file, other streams)."""
        sources = []
        for listing in listings:
            relayed = RelayStream(relay, listing)
            sources += [(relayed, s, len(relayed.pcm) > 1) for s in relayed.sources()]
        if not sources:
            raise ValueError("no PCM stream among those to watch")
        session = self.session
        history_s = min(float(history_s), MAX_HISTORY_S)
        rate = max(s.rate for _, s, _ in sources)
        if replace or not self.streams:
            self.close_all()
            session.buffer_s = history_s
            session.max_samples = max(MAX_CAPACITY, int(math.ceil(rate * history_s * 1.1)))
            if session.playing:
                session.pause()
        else:
            session.max_samples = max(session.max_samples, int(math.ceil(rate * session.buffer_s * 1.1)))
        names = []
        taken = set(session.streams) if self.streams else set()
        for i, (relayed, source, several) in enumerate(sources):
            name = _name(relayed, source, several, taken | set(names))
            if i == 0 and not self.streams:
                session.attach(source, name=name, keep_signals=False)
            else:
                session.add(source, name)
            self.streams[name] = Live(name, source)
            names.append(name)
        return names

    def align(self) -> bool:
        """Correct each stream onto the shared axis from what it says about
        UTC; True when a correction changed."""
        changed = False
        for live in self.streams.values():
            stream = self.session.streams.get(live.name)
            if stream is None:
                continue
            if live.applied is not None and stream.buffer.clock is not live.clock:
                if stream.buffer.clock is None:
                    live.applied = None                # a correction taken off: aligned again
                else:
                    live.by_hand = True                # session.synchronize or set_clock moved it
                    continue
            elif live.by_hand and stream.buffer.clock is None:
                live.by_hand, live.applied = False, None
            utc = live.relayed.utc_offset()
            if utc is None or live.source.pcm.anchor is None:
                continue
            if self.epoch is None:
                self.epoch = float(math.floor(live.source.pcm.anchor[1] + utc.offset))
            offset = self.epoch - utc.offset - live.trim
            moved = max(RESET_S, (utc.uncertainty_s or 0.0) / 4)
            if live.applied is None or abs(offset - live.applied) >= moved:
                self.session.set_clock(live.name, offset)
                live.clock = stream.buffer.clock
                live.applied = offset
                changed = True
        return changed

    def set_trim(self, name: str, seconds: float) -> None:
        live = self.streams[name]
        live.trim = float(seconds)
        live.applied = None
        live.by_hand = False
        live.clock = None
        self.align()

    def utc_of(self, t: float) -> float | None:
        """The UTC seconds of a time on the shared axis."""
        return None if self.epoch is None else self.epoch + t

    def close(self, name: str) -> None:
        """Stop watching one stream; what it held goes with it, unless it is the last."""
        live = self.streams.pop(name)
        live.source.closed = True
        live.source.stop()
        if name in self.session.streams and len(self.session.streams) > 1:
            self.session.remove(name)

    def close_all(self) -> None:
        """Stop watching every stream, keeping nothing of the alignment."""
        for live in self.streams.values():
            live.source.closed = True
            live.source.stop()
        self.streams.clear()
        self.epoch = None
