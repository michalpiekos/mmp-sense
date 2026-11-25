"""Live streams from an mmp-stream receiver's relay (stream format 3; mmp-stream's README, "Relay").

    relayed = list_streams("192.168.1.2:8091")             # what is open now
    stream = RelayStream("192.168.1.2:8091", relayed[0])
    sources = stream.sources()                             # a Source per PCM stream

One HTTP response carries a whole relayed session; a reader thread splits it
into the sources of its PCM streams. A sample's time is the node's timeline
clock in seconds, counted by the sample clock's frame counter from the first
block, so a lost block is a gap and blocks never overlap. How the timeline
relates to UTC is kept apart (`RelayStream.utc_offset`): the session's clock
correction puts it on one axis with other streams.
"""

from __future__ import annotations

import http.client
import json
import socket
import statistics
import struct
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..sources.base import EMPTY, Source

__all__ = ["RelayError", "RelaySource", "RelayStream", "Utc", "list_streams", "relay_url",
           "PCM_FORMATS"]

HEADER = struct.Struct("<BBHI")
#: Bytes per sample, and the full scale an integer count is divided by.
PCM_FORMATS = {"u8": (1, 128.0), "s16": (2, 32768.0), "s24": (3, 8388608.0),
               "s32": (4, 2147483648.0), "f32": (4, 1.0), "f64": (8, 1.0)}
#: Observations the UTC estimate is the median of.
OBSERVATIONS = 30
#: A source hands nothing over until it knows UTC, or this long after its first block.
HOLD_S = 3.0
#: Connecting, and waiting for the next message: beyond the receiver's own 15 s without data.
TIMEOUT_S = 30.0
#: Blocks the timeline is fitted against the frame counter over, and how often it is refitted.
FIT_BLOCKS = 6000
FIT_EVERY = 100
FIT_MIN = 100
#: How far, in samples, one block may move against the last to follow a refit.
SLEW = 0.25
RECEIVER_DOMAIN = "ntp:mmp-stream-receiver"


class RelayError(Exception):
    """A relay that cannot be reached, or refuses."""


def relay_url(relay: str, path: str = "") -> str:
    relay = relay.strip().rstrip("/")
    if "://" not in relay:
        relay = "http://" + relay
    return relay + path


def _request(relay: str, path: str, timeout: float | None):
    try:
        return urllib.request.urlopen(relay_url(relay, path), timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read() or b"{}").get("message", "")
        except ValueError:
            message = ""
        raise RelayError(f"{exc.code}: {message or exc.reason}") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RelayError(f"{relay} cannot be reached ({reason})") from None


def list_streams(relay: str, timeout: float = 5.0) -> list[dict]:
    """The streams open on a relay now, as its `GET /live` lists them."""
    with _request(relay, "/live", timeout) as res:
        return json.loads(res.read())["streams"]


@dataclass
class Utc:
    """How a stream's timeline relates to UTC: UTC seconds = timeline seconds + `offset`."""

    offset: float
    #: How sure, in seconds; None when not known (a UTC timeline declares none).
    uncertainty_s: float | None
    #: The clock it came from: ntp, gnss, gnss_pps, phone or unknown, or "receiver".
    source: str
    #: The clock's domain, if it declares one; the receiver's is ntp:mmp-stream-receiver.
    domain: str | None = None


@dataclass
class _Pcm:
    decl: dict
    width: int
    full_scale: float
    channels: int
    scale: np.ndarray | None
    queue: deque = field(default_factory=deque)
    #: The first block's (frame, timeline seconds).
    anchor: tuple[int, float] | None = None
    next_frame: int | None = None
    frames: int = 0
    lost_frames: int = 0
    #: Recent blocks' (frame, timeline seconds), and the line through them:
    #: timeline seconds = t + per_frame * (frame - frame0).
    points: deque = field(default_factory=lambda: deque(maxlen=FIT_BLOCKS))
    fit: tuple[int, float, float] | None = None
    blocks: int = 0
    last_time: float | None = None

    def time_of(self, frame: int) -> float:
        if self.fit is not None:
            frame0, t, per_frame = self.fit
            return t + per_frame * (frame - frame0)
        f0, t0 = self.anchor
        return t0 + (frame - f0) / float(self.decl["rate_hz"])

    def refit(self) -> None:
        """The timeline against the frame counter over the recent blocks, so a
        sample clock running fast or slow does not carry its error along."""
        if len(self.points) < FIT_MIN:
            return
        frames = np.array([p[0] for p in self.points], dtype=np.float64)
        times = np.array([p[1] for p in self.points])
        frame0 = int(frames[-1])
        per_frame, t = np.polyfit(frames - frame0, times, 1)
        nominal = 1.0 / float(self.decl["rate_hz"])
        if abs(per_frame / nominal - 1.0) < 0.01:          # an absurd fit keeps the last one
            self.fit = (frame0, float(t), float(per_frame))


def _decode(frames: bytes, fmt: str, channels: int) -> np.ndarray:
    """Interleaved frames as an (n, channels) array of counts (floats as they are)."""
    if fmt == "u8":
        counts = np.frombuffer(frames, np.uint8).astype(np.int32) - 128
    elif fmt == "s16":
        counts = np.frombuffer(frames, "<i2")
    elif fmt == "s24":
        b = np.frombuffer(frames, np.uint8).reshape(-1, 3).astype(np.int32)
        counts = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        counts = np.where(counts & 0x800000, counts - 0x1000000, counts)
    elif fmt == "s32":
        counts = np.frombuffer(frames, "<i4")
    elif fmt == "f32":
        counts = np.frombuffer(frames, "<f4")
    else:
        counts = np.frombuffer(frames, "<f8")
    return counts.reshape(-1, channels)


def encode(counts: np.ndarray, fmt: str) -> bytes:
    """Counts back into the stream's own sample format, byte for byte."""
    if fmt == "u8":
        return (np.asarray(counts, np.int32) + 128).astype(np.uint8).tobytes()
    if fmt == "s16":
        return np.asarray(counts).astype("<i2").tobytes()
    if fmt == "s24":
        c = np.asarray(counts, np.int64).reshape(-1) & 0xFFFFFF
        return np.stack([c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF], axis=1).astype(np.uint8).tobytes()
    if fmt == "s32":
        return np.asarray(counts).astype("<i4").tobytes()
    if fmt == "f32":
        return np.asarray(counts).astype("<f4").tobytes()
    return np.asarray(counts).astype("<f8").tobytes()


class RelayStream:
    """One relayed session: its declaration, its PCM streams and how it is timed."""

    def __init__(self, relay: str, listing: dict) -> None:
        self.relay = relay
        self.id = listing["session_id"]
        self.listing = listing
        self.device = listing.get("device") or {}
        self.clocks = {c["id"]: c for c in listing.get("clocks") or []}
        self.timeline = next((c for c in self.clocks.values() if c.get("timeline")), None)
        if self.timeline is None:
            raise ValueError(f"relayed session {self.id} declares no timeline clock")
        self.pcm: dict[int, _Pcm] = {}
        for s in listing.get("streams") or []:
            if s.get("kind") != "pcm" or s.get("format") not in PCM_FORMATS:
                continue
            width, full = PCM_FORMATS[s["format"]]
            scales = [c.get("scale") for c in s["channels"]]
            scale = np.array([x if x is not None else 1.0 for x in scales]) if any(scales) else None
            self.pcm[s["id"]] = _Pcm(s, width, full, len(s["channels"]), scale)
        #: (utc seconds - timeline seconds, uncertainty s): per UTC clock of the node, and the receiver's.
        self._device: dict[int, deque] = {}
        self._receiver: deque = deque(maxlen=OBSERVATIONS)
        #: The clock the relay adds for its arrival times; the relayed hello names it.
        self.receiver_clock: int | None = None
        #: connecting, live, or ended; `end` says why it ended.
        self.state = "connecting"
        self.end = ""
        self.relay_lost = 0
        self._lock = threading.Lock()
        #: Held while a message is taken, so a restart sees none half done.
        self.messages_lock = threading.RLock()
        self._users: set = set()
        self._thread: threading.Thread | None = None
        self._response = None
        #: Each connection's number; a reader whose number is not current stops touching anything.
        self._generation = 0

    @property
    def node(self) -> str:
        return self.listing.get("node") or self.device.get("node") or self.device.get("model") or "stream"

    def sources(self) -> list["RelaySource"]:
        return [RelaySource(self, sid) for sid in self.pcm]

    # -- the connection -------------------------------------------------------

    def _acquire(self, user) -> None:
        with self._lock:
            self._users.add(user)
            if self._thread is None and self.state != "ended":
                self._generation += 1
                self._thread = threading.Thread(target=self._run, args=(self._generation,),
                                                name=f"relay-{self.id[:8]}", daemon=True)
                self._thread.start()

    def _release(self, user) -> None:
        with self.messages_lock, self._lock:              # a message being taken finishes first
            self._users.discard(user)
            if self._users or self._thread is None:
                return
            self._generation += 1
            response, self._thread, self._response = self._response, None, None
            if self.state != "ended":
                self.state = "stopped"
        if response is not None:
            try:
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)  # wakes the blocked read
            except (AttributeError, OSError):
                pass

    def _current(self, generation: int) -> bool:
        return self._generation == generation

    def _finish(self, end: str, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and not self._current(generation):
                return
            if self.state != "ended":
                self.state, self.end = "ended", end

    def _run(self, generation: int) -> None:
        try:
            response = _request(self.relay, f"/live/{self.id}", TIMEOUT_S)
        except RelayError as exc:
            self._finish(str(exc), generation)
            return
        with self._lock:
            if not self._current(generation):
                response.close()
                return
            self._response = response
        end = "link_lost"
        try:
            with response:
                while self._current(generation):
                    head = response.read(HEADER.size)
                    if len(head) < HEADER.size:
                        break
                    kind, flags, stream, length = HEADER.unpack(head)
                    body = response.read(length) if length else b""
                    if len(body) < length:
                        break
                    with self.messages_lock:
                        if not self._current(generation):
                            return
                        if self.on_message(chr(kind), flags, stream, body):
                            return
        except (OSError, http.client.HTTPException):
            pass
        except Exception as exc:                              # noqa: BLE001
            end = f"protocol_error: {exc}"
        self._finish(end, generation)

    # -- messages ---------------------------------------------------------------

    def on_message(self, kind: str, flags: int, stream: int, body: bytes) -> bool:
        """Takes one relayed message; True once the stream has ended."""
        if kind == "H":
            hello = json.loads(body)
            self.receiver_clock = (hello.get("relay") or {}).get("receiver_clock", self.receiver_clock)
            self.clocks.update({c["id"]: c for c in hello.get("clocks") or []})
            with self._lock:
                self.state = "live"
        elif kind == "D":
            self._data(stream, body)
        elif kind == "C":
            self._observation(body[1:])
        elif kind == "B":
            i = 0
            while i + HEADER.size <= len(body):
                k, f, s, n = HEADER.unpack_from(body, i)
                i += HEADER.size
                if self.on_message(chr(k), f, s, body[i:i + n]):
                    return True
                i += n
        elif kind == "N":
            event = json.loads(body[8:] or b"{}")
            if not isinstance(event, dict):
                return False
            if event.get("kind") == "relay_lost":
                self.relay_lost += int(event.get("messages", 0))
            elif event.get("kind") == "stream_end":
                self._finish(event.get("end", "clean"))
        elif kind == "X":
            self._finish(self.end or "clean")
            return True
        return False

    def _data(self, stream: int, body: bytes) -> None:
        pcm = self.pcm.get(stream)
        if pcm is None or len(body) < 16:
            return
        first, t = struct.unpack_from("<Qq", body)
        counts = _decode(body[16:], pcm.decl["format"], pcm.channels)
        if not len(counts):
            return
        t_s = t / float(self.timeline["hz"])
        if pcm.anchor is None:
            pcm.anchor = (first, t_s)
        contiguous = pcm.next_frame is not None and first == pcm.next_frame
        if pcm.next_frame is not None:
            if first + len(counts) <= pcm.next_frame:
                return                                 # a repeat
            if first < pcm.next_frame:
                counts = counts[pcm.next_frame - first:]
                t_s += (pcm.next_frame - first) / float(pcm.decl["rate_hz"])
                first = pcm.next_frame
                contiguous = True
            pcm.lost_frames += first - pcm.next_frame
        pcm.next_frame = first + len(counts)
        pcm.frames += len(counts)
        pcm.points.append((first, t_s))
        pcm.blocks += 1
        if pcm.fit is None and len(pcm.points) >= FIT_MIN or pcm.blocks % FIT_EVERY == 0:
            pcm.refit()
        start = pcm.time_of(first)
        per_frame = pcm.fit[2] if pcm.fit is not None else 1.0 / float(pcm.decl["rate_hz"])
        if pcm.last_time is not None:
            expected = pcm.last_time + per_frame
            if contiguous:
                # A refit is followed a little each block: no overlap, and no jump that looks like a loss.
                start = min(max(start, expected - SLEW * per_frame), expected + SLEW * per_frame)
            elif start < expected:
                start = expected
        times = start + per_frame * np.arange(len(counts))
        pcm.last_time = float(times[-1])
        pcm.queue.append((times, counts))

    def _observation(self, raw: bytes) -> None:
        if not raw:
            return
        n = raw[0]
        readings = [struct.unpack_from("<BqI", raw, 1 + 13 * i) for i in range(n) if 1 + 13 * (i + 1) <= len(raw)]
        timeline = next(((v, u) for c, v, u in readings if c == self.timeline["id"]), None)
        if timeline is None:
            return
        t_s = timeline[0] / float(self.timeline["hz"])
        for clock, value, unc in readings:
            decl = self.clocks.get(clock)
            if decl is None or decl.get("kind") != "utc" or clock == self.timeline["id"]:
                continue
            entry = (value / float(decl["hz"]) - t_s, max(unc, timeline[1]) / 1e9)
            if clock == self.receiver_clock:
                self._receiver.append(entry)
            else:
                self._device.setdefault(clock, deque(maxlen=OBSERVATIONS)).append(entry)

    def utc_offset(self) -> Utc | None:
        """How the timeline relates to UTC: itself, when it is a UTC clock; else
        the node's surest UTC clock; else the receiver's arrival times; None
        before any is seen."""
        tl = self.timeline
        if tl.get("kind") == "utc":
            return Utc(0.0, None, tl.get("source") or "unknown", tl.get("domain"))
        best = None
        for clock, seen in list(self._device.items()):
            if seen:
                entries = list(seen)
                utc = Utc(statistics.median(e[0] for e in entries), statistics.median(e[1] for e in entries),
                          self.clocks[clock].get("source") or "unknown", self.clocks[clock].get("domain"))
                if best is None or utc.uncertainty_s < best.uncertainty_s:  # node clocks always state one
                    best = utc
        if best is not None:
            return best
        if self._receiver:
            entries = list(self._receiver)
            return Utc(statistics.median(e[0] for e in entries), statistics.median(e[1] for e in entries),
                       "receiver", RECEIVER_DOMAIN)
        return None


class RelaySource(Source):
    """One PCM stream of a relayed session, as a session's live source.

    Values are full-scale (-1..1) for counts without a scale, or counts
    times each channel's scale. They are held as float32 when that keeps
    every count exactly, which `counts()` relies on to give them back.
    """

    kind = "live"

    def __init__(self, relayed: RelayStream, stream_id: int) -> None:
        self.relayed = relayed
        self.stream_id = stream_id
        self.pcm = relayed.pcm[stream_id]
        decl = self.pcm.decl
        self.name = decl["name"]
        self.channels = [c["name"] for c in decl["channels"]]
        units = {c.get("unit") for c in decl["channels"] if c.get("unit")}
        self.units = units.pop() if len(units) == 1 and self.pcm.scale is not None else ""
        self.rate = float(decl["rate_hz"])
        self.format = decl["format"]
        bits = decl.get("valid_bits") or self.pcm.width * 8
        exact = self.format in ("u8", "s16", "s24", "f32") or (self.format == "s32" and bits <= 24)
        self.dtype = np.dtype(np.float32 if exact and self.pcm.scale is None else np.float64)
        self._started = False
        #: Closed for good: Start intake does not open it again.
        self.closed = False

    def start(self) -> "RelaySource":
        if not self._started and not self.closed:
            self._started = True
            self.relayed._acquire(self)
        return self

    def stop(self) -> None:
        if self._started:
            self._started = False
            self.relayed._release(self)

    def restart(self) -> "RelaySource":
        """From now on: nothing taken before, and the stretch stopped is not counted lost."""
        with self.relayed.messages_lock:
            self.pcm.queue.clear()
            self.pcm.next_frame = None
        self.start()
        return self

    def _ready(self) -> bool:
        if self.relayed.utc_offset() is not None:
            return True
        queue = self.pcm.queue
        return bool(queue) and float(queue[-1][0][-1] - queue[0][0][0]) >= HOLD_S

    def read(self):
        queue = self.pcm.queue
        if not queue or not self._ready():
            return EMPTY
        parts = []
        while queue:
            parts.append(queue.popleft())
        times = np.concatenate([p[0] for p in parts])
        counts = np.concatenate([p[1] for p in parts])
        return times, self.values(counts)

    def values(self, counts: np.ndarray) -> np.ndarray:
        if self.pcm.scale is not None:
            return (np.asarray(counts, np.float64) * self.pcm.scale).astype(self.dtype)
        if self.format in ("f32", "f64"):
            return np.asarray(counts, self.dtype)
        return (np.asarray(counts, np.float64) / self.pcm.full_scale).astype(self.dtype)

    def counts(self, values: np.ndarray) -> np.ndarray:
        """The node's own samples back from what `read` gave."""
        values = np.asarray(values, np.float64)
        if self.format in ("f32", "f64"):
            return values / self.pcm.scale if self.pcm.scale is not None else values
        if self.pcm.scale is not None:
            return np.rint(values / self.pcm.scale)
        return np.rint(values * self.pcm.full_scale)

    @property
    def state(self) -> str:
        return self.relayed.state

    def describe(self) -> dict:
        info = super().describe()
        info.update(relay=self.relayed.relay, session_id=self.relayed.id,
                    node=self.relayed.node, state=self.relayed.state, end=self.relayed.end,
                    lost_frames=self.pcm.lost_frames)
        return info
