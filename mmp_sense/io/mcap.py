"""MCAP recordings with JSON messages, read with the official `mcap` library.

Two conventions are understood:

- one channel per topic, messages of one sample each, and a JSON Schema
  whose `properties` name the components in order and carry the unit;
- srv-senselab's (and the mmp-sample phone app's): `dev.sensetool.*.v1`
  schemas, `/sensor/accelerometer|gyroscope|magnetometer` messages of
  `{elapsed_ns, accuracy, values: [x, y, z]}` in SI units and
  `/location/gnss` fixes, timed by session-relative nanoseconds.

A file the phone wrote has no summary section, and one it never closed
has no footer either; both are read by scanning their records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader

from ..core.data import TimeSeries

__all__ = ["read_mcap", "read_mcap_topics", "topics_in", "plottable_topics",
           "RecordedTopic", "MAGIC"]

MAGIC = b"\x89MCAP0\r\n"

_SENSOR_SAMPLE = "dev.sensetool.SensorSample.v1"
_LOCATION_SAMPLE = "dev.sensetool.LocationSample.v1"
_SENSOR_UNITS = {
    "/sensor/accelerometer": "m/s^2",
    "/sensor/gyroscope": "rad/s",
    "/sensor/magnetometer": "uT",
}
_LOCATION_FIELDS = ["latitude", "longitude", "altitude_m", "accuracy_m", "speed_mps",
                    "bearing_deg"]
_NOT_NUMERIC = {"string", "boolean", "object", "array", "null"}


@dataclass
class RecordedTopic:
    """One channel of a recording, before it is turned into a series."""

    topic: str
    message_count: int
    components: list[str]
    unit: str
    #: Its messages hold numbers to plot, rather than events or bookkeeping.
    plottable: bool = True
    #: The schema's name, which picks how a message becomes a sample.
    schema: str = ""


def _parsed(schema) -> dict:
    if schema is None or not schema.data:
        return {}
    try:
        parsed = json.loads(schema.data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _schema_fields(parsed: dict) -> tuple[list[str], str]:
    """Component names in order, and the unit, from a generated schema.

    JSON Schema has no ordering of its own, but `properties` is written
    in component order and Python keeps that order when it parses the
    object, which is what makes the columns come back as x, y, z rather
    than alphabetically.
    """
    properties = parsed.get("properties")
    if not isinstance(properties, dict):
        return [], ""
    names = [name for name, spec in properties.items() if not _non_numeric(spec)]
    unit = parsed.get("unit", "")
    if not unit:
        # Each component may carry the unit; they share one.
        for name in names:
            spec = properties[name]
            if isinstance(spec, dict) and spec.get("unit"):
                unit = spec["unit"]
                break
    return names, (unit if isinstance(unit, str) else "")


def _non_numeric(spec) -> bool:
    if not isinstance(spec, dict):
        return False
    if "const" in spec:
        return True
    kind = spec.get("type")
    kinds = set(kind) if isinstance(kind, list) else {kind}
    return bool(kinds) and kinds <= _NOT_NUMERIC


def _describe(topic: str, schema, count: int, encoding: str = "json") -> RecordedTopic:
    name = schema.name if schema is not None else ""
    if encoding.lower() != "json":
        # Protobuf, ROS messages, CDR: no numbers this reader can see.
        return RecordedTopic(topic, count, [], "", plottable=False, schema=name)
    if name == _SENSOR_SAMPLE:
        return RecordedTopic(topic, count, ["x", "y", "z"], _SENSOR_UNITS.get(topic, ""),
                             schema=name)
    if name == _LOCATION_SAMPLE:
        return RecordedTopic(topic, count, list(_LOCATION_FIELDS), "", schema=name)
    if name.startswith("dev.sensetool."):
        # Timing points, markers, device status: not samples of a signal.
        return RecordedTopic(topic, count, [], "", plottable=False, schema=name)
    names, unit = _schema_fields(_parsed(schema))
    return RecordedTopic(topic, count, names, unit, schema=name)


def topics_in(data: bytes | str | Path) -> list[RecordedTopic]:
    """What a recording holds; from its summary when it has one, without
    decoding its messages."""
    summary = None
    try:
        with _reader(data) as reader:
            summary = reader.get_summary()
    except Exception:                                          # noqa: BLE001
        summary = None
    if summary is not None and summary.channels and summary.statistics:
        counts = summary.statistics.channel_message_counts
        out = [_describe(channel.topic, summary.schemas.get(channel.schema_id),
                         int(counts.get(channel.id, 0)), channel.message_encoding)
               for channel in summary.channels.values()]
    else:
        schemas, channels, counts = {}, {}, {}
        for record in _scan(data):
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Message):
                counts[record.channel_id] = counts.get(record.channel_id, 0) + 1
        out = [_describe(channel.topic, schemas.get(channel.schema_id),
                         counts.get(channel.id, 0), channel.message_encoding)
               for channel in channels.values()]
    return sorted(out, key=lambda t: t.topic)


def plottable_topics(data: bytes | str | Path,
                     catalog: list[RecordedTopic] | None = None) -> list[str]:
    """The topics worth opening: numbers to plot, and some of them."""
    return [t.topic for t in (catalog if catalog is not None else topics_in(data))
            if _plottable(t) and t.message_count]


def _plottable(topic: RecordedTopic) -> bool:
    # A schema that names no numeric field has nothing to plot either.
    return topic.plottable and bool(topic.components or not topic.schema)


def _collect(data, topics) -> dict[str, list[tuple[int, bytes]]]:
    """(publish_time, data) of each message on `topics`, in one pass, in
    file order."""
    wanted = set(topics)
    out: dict[str, list[tuple[int, bytes]]] = {t: [] for t in wanted}
    try:
        with _reader(data) as reader:
            summary = reader.get_summary()
            if summary is not None and summary.chunk_indexes:
                read = 0
                try:
                    for _schema, channel, message in reader.iter_messages(
                            topics=list(wanted), log_time_order=False):
                        out[channel.topic].append((message.publish_time, message.data))
                        read += 1
                except Exception:                              # noqa: BLE001
                    if read:
                        # What was read stands; scanning again would repeat it.
                        return out
                    raise
                return out
    except Exception:                                          # noqa: BLE001
        out = {t: [] for t in wanted}
    ids: dict[int, str] = {}
    for record in _scan(data):
        if isinstance(record, Channel) and record.topic in wanted:
            ids[record.id] = record.topic
        elif isinstance(record, Message) and record.channel_id in ids:
            out[ids[record.channel_id]].append((record.publish_time, record.data))
    return out


def _scan(data):
    """Every record up to the end, or up to where a file was cut short or
    stops making sense."""
    with _opened(data) as stream:
        try:
            yield from StreamReader(stream, emit_chunks=False).records
        except Exception:                                      # noqa: BLE001
            return


def read_mcap(
    data: bytes | str | Path,
    topic: str | None = None,
    *,
    epoch_ns: int | None = None,
    name: str | None = None,
    catalog: list[RecordedTopic] | None = None,
) -> TimeSeries:
    """One topic of a recording as a TimeSeries.

    `publish_time` is the producer's acquisition time. Time is returned in
    seconds from `epoch_ns`, defaulting to the first message, because
    nanoseconds since 1970 do not survive a float64 intact.

    Without `topic`, the only plottable topic in the file is used, and a
    file with several is an error rather than a guess. `catalog` is
    `topics_in(data)`, when the caller has it already.
    """
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    available = catalog if catalog is not None else topics_in(data)
    described = _choose(available, topic)
    return _series(described, _collect(data, [described.topic])[described.topic],
                   epoch_ns, name)


def read_mcap_topics(data: bytes | str | Path, topics,
                     catalog: list[RecordedTopic] | None = None) -> list[TimeSeries]:
    """Several topics of a recording, their times from one epoch -- the
    first message of any of them -- so they line up as they were sent."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    available = catalog if catalog is not None else topics_in(data)
    described = [_choose(available, topic) for topic in topics]
    collected = _collect(data, [d.topic for d in described])
    series = [_series(d, collected[d.topic], None, None) for d in described]
    epochs = [s.metadata["epoch_ns"] for s in series if "epoch_ns" in s.metadata]
    if not epochs:
        return series
    base = min(epochs)
    for one in series:
        epoch = one.metadata.get("epoch_ns")
        if epoch is not None and epoch != base:
            one.time = one.time + (epoch - base) / 1e9
            one.metadata["epoch_ns"] = base
    return series


def _choose(available: list[RecordedTopic], topic: str | None) -> RecordedTopic:
    if not available:
        raise ValueError("this recording holds no telemetry")
    if topic is None:
        candidates = plottable_topics(b"", available)
        if not candidates:
            raise ValueError(_nothing_to_plot(available))
        if len(candidates) > 1:
            raise ValueError(
                "this recording holds " + ", ".join(candidates) + "; name the one to read"
            )
        topic = candidates[0]
    known = {t.topic: t for t in available}
    if topic not in known:
        raise KeyError(f"no topic {topic!r}; have {', '.join(known)}")
    described = known[topic]
    if not _plottable(described):
        raise ValueError(f"{topic} holds no numbers to plot")
    return described


def _nothing_to_plot(available: list[RecordedTopic]) -> str:
    return ("no topic of this recording holds numbers to plot (JSON messages with "
            "numeric fields); it holds " + ", ".join(t.topic for t in available))


def _series(described: RecordedTopic, messages, epoch_ns, name) -> TimeSeries:
    topic = described.topic
    times: list[int] = []
    rows: list[list[float]] = []
    columns: list[str] = list(described.components)
    for publish_time, payload in messages:
        try:
            sample = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(sample, dict):
            continue
        if described.schema == _SENSOR_SAMPLE:
            row = _triple(sample.get("values"))
        else:
            if not columns:
                # No schema to order them: the writer's own order stands.
                columns = [k for k, v in sample.items() if _is_number(v)]
            row = [_number(sample.get(c)) for c in columns]
        times.append(publish_time)
        rows.append(row)

    if not times:
        return TimeSeries(
            name=name or topic,
            time=np.zeros(0),
            values=np.zeros((0, max(1, len(columns)))),
            channels=columns or ["c0"],
            units=described.unit,
            metadata={"topic": topic, "source": "mcap"},
        )

    stamps = np.asarray(times, dtype=np.int64)
    order = np.argsort(stamps, kind="stable")
    stamps = stamps[order]
    values = np.asarray(rows, dtype=float)[order]
    base = int(stamps[0]) if epoch_ns is None else int(epoch_ns)
    return TimeSeries(
        name=name or topic,
        time=(stamps - base) / 1e9,
        values=values,
        channels=columns,
        units=described.unit,
        metadata={
            "topic": topic,
            "source": "mcap",
            "epoch_ns": base,
            "message_count": len(stamps),
        },
    )


def _triple(values) -> list[float]:
    if isinstance(values, list) and len(values) == 3:
        return [_number(v) for v in values]
    return [float("nan")] * 3


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value) -> float:
    """A JSON value as a float; anything else is a gap, not a crash."""
    if isinstance(value, bool) or value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class _opened:
    """A binary stream over bytes or a path, closing what it opened."""

    def __init__(self, data: bytes | str | Path):
        self._opened = None
        if isinstance(data, (bytes, bytearray, memoryview)):
            self._stream = BytesIO(bytes(data))
        else:
            self._opened = open(data, "rb")
            self._stream = self._opened

    def __enter__(self):
        self._stream.seek(0)
        return self._stream

    def __exit__(self, *_exc) -> None:
        if self._opened is not None:
            self._opened.close()


class _reader(_opened):
    """make_reader over bytes or a path."""

    def __enter__(self):
        return make_reader(super().__enter__())
