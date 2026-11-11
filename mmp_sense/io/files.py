"""Opening a data file, whatever it holds.

    read_file("walk.mcap")                 every plottable topic, a series each
    read_file("clip.wav")                  sound, as the file holds it
    read_file("accel.wav", scale=0.0024)   a sensor's WAV, in m/s^2
    read_file("table.tsv")                 a column per channel
    read_file("capture.npy", rate=400)     an array, with its rate
    read_file("export/<session>")          a session srv-senselab exported with --raw

The format is told by the file's first bytes where it has a signature
(MCAP, WAV, NumPy), else by its suffix, else -- text -- as delimited
columns.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.data import TimeSeries
from .arrays import NUMPY_SUFFIXES, read_numpy
from .audio import AUDIO_SUFFIXES, is_sensor_wav, read_audio, read_sensor_wav
from .csv import TEXT_SUFFIXES, read_timeseries
from .mcap import MAGIC, plottable_topics, read_mcap, read_mcap_topics, topics_in
from .session_export import is_session_folder, read_session_folder

__all__ = ["read_file", "kind_of", "FileFormatError", "FILE_KINDS"]

#: What `kind_of` answers, and what each is read as.
FILE_KINDS = {
    "mcap": "an MCAP recording of JSON messages",
    "wav": "a WAV: sound, or a sensor's counts below 8 kHz",
    "audio": "compressed audio, decoded by ffmpeg",
    "array": "a NumPy .npy or .npz",
    "text": "delimited text: CSV, TSV, whitespace",
    "session": "a session folder of `sample export --raw`: its streams, mic names and clocks",
}

#: srv-senselab's clip export names its files <session>_<stream>_<span>.
_STREAM_UNITS = {
    "accelerometer": "m/s^2",
    "gyroscope": "rad/s",
    "magnetometer": "uT",
}


class FileFormatError(ValueError):
    """A file that is none of the formats read here."""


def kind_of(path: str | Path) -> str:
    """Which reader a file goes to; one of FILE_KINDS."""
    path = Path(path)
    if is_session_folder(path):
        return "session"
    suffix = path.suffix.lower()
    try:
        with open(path, "rb") as file:
            head = file.read(4096)
    except OSError as exc:
        raise FileFormatError(f"{path}: {exc.strerror or exc}") from None
    if not head:
        raise FileFormatError(f"{path}: the file is empty")
    if head.startswith(MAGIC[:5]):
        return "mcap"
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"\x93NUMPY") or (head.startswith(b"PK") and suffix == ".npz"):
        return "array"
    if suffix == ".mcap":
        return "mcap"
    if suffix in NUMPY_SUFFIXES:
        return "array"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in TEXT_SUFFIXES or (head and b"\0" not in head and _is_text(head)):
        return "text"
    raise FileFormatError(
        f"{path}: not a format mmp-sense reads; it reads "
        + "; ".join(FILE_KINDS.values())
    )


def read_file(
    path: str | Path,
    *,
    topics=None,
    rate: float | None = None,
    scale: float | None = None,
    units: str | None = None,
) -> list[TimeSeries]:
    """The series a file holds: one, or one per topic of an MCAP.

    `topics` picks an MCAP's topics, all plottable ones by default.
    `rate` is the sample rate of a file that carries no time (an array, a
    table without a time column). `scale` and `units` make a sensor's WAV
    counts into a physical quantity.
    """
    path = Path(path)
    kind = kind_of(path)
    if kind == "session":
        return read_session_folder(path)
    if topics and kind != "mcap":
        raise ValueError(f"{path.name} holds one stream; topics are an MCAP's")
    if kind == "mcap":
        return _read_mcap(path, list(topics or []))
    if kind == "wav" and (scale is not None or is_sensor_wav(path)):
        return [read_sensor_wav(path, scale=scale, units=units)]
    if kind in ("wav", "audio"):
        return [read_audio(path, decoder="wav" if kind == "wav" else "ffmpeg")]
    if kind == "array":
        series = read_numpy(path, rate)
    else:
        series = read_timeseries(path, rate=rate)
        if not series.units:
            series.units = _stream_units(path)
    if series.n_channels == 0:
        raise ValueError(f"{path}: no numeric columns besides time")
    if units is not None:
        series.units = units
    return [series]


def _read_mcap(path: Path, topics: list[str]) -> list[TimeSeries]:
    data = path.read_bytes()
    catalog = topics_in(data)
    try:
        if not topics:
            topics = plottable_topics(data, catalog)
        if len(topics) > 1:
            return read_mcap_topics(data, topics, catalog)
        return [read_mcap(data, topics[0] if topics else None, catalog=catalog)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path}: {exc.args[0] if exc.args else exc}") from None


def _stream_units(path: Path) -> str:
    for part in re.split(r"[_\W]+", path.stem.lower()):
        if part in _STREAM_UNITS:
            return _STREAM_UNITS[part]
    return ""


def _is_text(head: bytes) -> bool:
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut by the end of the sample is still text.
        return exc.start >= len(head) - 3
    return True
