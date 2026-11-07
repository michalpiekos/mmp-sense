"""Audio files as a time series: a voice clip from a phone, say.

    series = read_audio("clip.m4a")          # its own rate and channels

WAV is read here; what phone recorders make -- m4a, aac, 3gp, amr, mp3,
ogg, opus, flac, webm -- is decoded by the ffmpeg program. Either way the
result keeps the file's rate and channels, in full-scale units (-1..1),
starting at 0 s: one channel is called `audio`, several `ch0`, `ch1`, ...
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..core.data import TimeSeries

__all__ = ["read_audio", "read_sensor_wav", "is_sensor_wav", "AUDIO_SUFFIXES",
           "AudioError", "SENSOR_RATE_BELOW", "wav_rate"]

#: Suffixes opened as audio; all but .wav need ffmpeg.
AUDIO_SUFFIXES = frozenset({
    ".wav", ".m4a", ".aac", ".3gp", ".amr", ".mp3", ".ogg", ".opus", ".flac", ".webm",
})

#: A WAV sampled slower than this holds a sensor's counts, not sound:
#: srv-senselab stores accelerometers and gyroscopes that way.
SENSOR_RATE_BELOW = 8000


class AudioError(ValueError):
    """An audio file that cannot be read, and why."""


def read_audio(path, *, decoder: str | None = None) -> TimeSeries:
    """A clip as a TimeSeries at its own rate, with every channel: one is
    called `audio`, several `ch0`, `ch1`, ... `decoder` -- "wav" or
    "ffmpeg" -- overrides the one the suffix picks."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"{path}: no such file")
    decoder = decoder or ("wav" if path.suffix.lower() == ".wav" else "ffmpeg")
    values, rate = _read_wav(path) if decoder == "wav" else _decode(path)
    if len(values) == 0:
        raise AudioError(f"{path}: no audio samples in it")
    values = values.reshape(len(values), -1)
    count = values.shape[1]
    return TimeSeries(
        name=path.stem,
        time=np.arange(len(values)) / float(rate),
        values=values.astype(float),
        channels=["audio"] if count == 1 else [f"ch{i}" for i in range(count)],
        units="",
        sample_rate=float(rate),
        metadata={"source": str(path), "source_rate": rate,
                  "source_channels": count, "decoder": decoder},
    )


def is_sensor_wav(path) -> bool:
    """Whether a WAV is a sensor's: sampled below SENSOR_RATE_BELOW."""
    rate = wav_rate(path)
    return rate is not None and rate < SENSOR_RATE_BELOW


def wav_rate(path) -> int | None:
    """A WAV's sample rate from its `fmt ` chunk; None if it is not one."""
    try:
        with open(path, "rb") as file:
            head = file.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
                return None
            while True:
                chunk = file.read(8)
                if len(chunk) < 8:
                    return None
                size = int.from_bytes(chunk[4:], "little")
                if chunk[:4] == b"fmt ":
                    fmt = file.read(8)
                    return int.from_bytes(fmt[4:8], "little") if len(fmt) == 8 else None
                file.seek(size + (size & 1), 1)
    except OSError:
        return None


def read_sensor_wav(path, *, scale: float | None = None,
                    units: str | None = None) -> TimeSeries:
    """A sensor's WAV at its own rate, every channel kept: `x`, `y`, `z`
    for three, `ch0`.. otherwise. Values are the file's counts, or times
    `scale` (units per count, srv-senselab's `encoding.scale`)."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"{path}: no such file")
    try:
        native, data = wavfile.read(path)
    except Exception as exc:                                  # noqa: BLE001
        raise AudioError(f"{path}: not a WAV file mmp-sense can read ({exc})") from None
    values = np.asarray(data, dtype=float).reshape(len(data), -1)
    if len(values) == 0:
        raise AudioError(f"{path}: no samples in it")
    if scale is not None:
        values = values * float(scale)
    count = values.shape[1]
    names = ["x", "y", "z"] if count == 3 else [f"ch{i}" for i in range(count)]
    return TimeSeries(
        name=path.stem,
        time=np.arange(len(values)) / float(native),
        values=values,
        channels=names,
        units=units if units is not None else ("" if scale is not None else "counts"),
        sample_rate=float(native),
        metadata={"source": str(path), "source_rate": int(native),
                  "source_channels": count, "decoder": "wav", "scale": scale},
    )


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        native, data = wavfile.read(path)
    except Exception as exc:                                  # noqa: BLE001
        raise AudioError(f"{path}: not a WAV file mmp-sense can read ({exc})") from None
    data = np.asarray(data)
    if data.dtype.kind == "f":
        scaled = data.astype(float)
    elif data.dtype == np.uint8:                          # 8-bit WAV is unsigned
        scaled = (data.astype(float) - 128.0) / 128.0
    elif data.dtype.kind == "i":
        # scipy gives 24-bit samples in the top of an int32, so one scale fits.
        scaled = data.astype(float) / float(2 ** (8 * data.dtype.itemsize - 1))
    else:
        raise AudioError(f"{path}: WAV samples of type {data.dtype} are not supported")
    return scaled, int(native)


def _ffmpeg(tool: str, path: Path) -> str:
    found = shutil.which(tool)
    if found is None:
        raise AudioError(
            f"{path}: decoding {path.suffix} needs {tool}, which is not on PATH; "
            "install ffmpeg (e.g. apt install ffmpeg) or convert the clip to .wav"
        )
    return found


def _decode(path: Path) -> tuple[np.ndarray, int]:
    probe = subprocess.run(
        [_ffmpeg("ffprobe", path), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        native, channels = int(stream["sample_rate"]), int(stream["channels"])
    except (ValueError, KeyError, IndexError):
        reason = probe.stderr.strip().splitlines()[-1:] or ["no audio stream"]
        raise AudioError(f"{path}: not audio ffmpeg can read ({reason[0]})") from None
    command = [_ffmpeg("ffmpeg", path), "-v", "error", "-nostdin", "-i", str(path),
               "-vn", "-ac", str(channels), "-f", "f32le", "-"]
    decoded = subprocess.run(command, capture_output=True)
    if decoded.returncode != 0:
        reason = decoded.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["?"]
        raise AudioError(f"{path}: ffmpeg could not decode it ({reason[0]})")
    values = np.frombuffer(decoded.stdout, dtype="<f4")
    return values.reshape(-1, channels), native
