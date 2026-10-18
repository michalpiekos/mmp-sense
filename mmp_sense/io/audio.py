"""Audio files as a time series: a voice clip from a phone, say.

    series = read_audio("clip.m4a")          # 16 kHz, one channel "audio"

WAV is read here; what phone recorders make -- m4a, aac, 3gp, amr, mp3,
ogg, opus, flac, webm -- is decoded by the ffmpeg program, which does the
mixing and resampling as it goes. Either way the result is one channel
called `audio`, in full-scale units (-1..1), starting at 0 s.

16 kHz by default: speech lives below 8 kHz, and a minute at 48 kHz is
three million samples to draw and transform on every change.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal as _signal
from scipy.io import wavfile

from ..core.data import TimeSeries

__all__ = ["read_audio", "read_sensor_wav", "is_sensor_wav", "AUDIO_SUFFIXES",
           "AudioError", "SPEECH_RATE", "SENSOR_RATE_BELOW", "wav_rate"]

#: Suffixes opened as audio; all but .wav need ffmpeg.
AUDIO_SUFFIXES = frozenset({
    ".wav", ".m4a", ".aac", ".3gp", ".amr", ".mp3", ".ogg", ".opus", ".flac", ".webm",
})

#: The default rate: enough for speech, cheap enough to explore.
SPEECH_RATE = 16000


#: A WAV sampled slower than this holds a sensor's counts, not sound:
#: srv-senselab stores accelerometers and gyroscopes that way.
SENSOR_RATE_BELOW = 8000


class AudioError(ValueError):
    """An audio file that cannot be read, and why."""


def read_audio(path, rate: int | None = SPEECH_RATE, *,
               keep_channels: bool = False, decoder: str | None = None) -> TimeSeries:
    """A clip as a mono `audio` TimeSeries at `rate` Hz (None: its own rate);
    with `keep_channels`, one channel per channel of the file, `ch0`..
    `decoder` -- "wav" or "ffmpeg" -- overrides the one the suffix picks."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"{path}: no such file")
    if rate is not None and rate <= 0:
        raise AudioError(f"rate must be positive, got {rate}")
    if (decoder or ("wav" if path.suffix.lower() == ".wav" else "ffmpeg")) == "wav":
        values, native, channels = _read_wav(path, keep_channels)
        decoder = "wav"
        if rate is not None and rate != native:
            values = _resample(values, native, rate)
    else:
        values, native, channels = _decode(path, rate, keep_channels)
        decoder = "ffmpeg"
    values = values.reshape(len(values), -1)
    names = [f"ch{i}" for i in range(values.shape[1])] if keep_channels else ["audio"]
    out_rate = float(rate or native)
    if len(values) == 0:
        raise AudioError(f"{path}: no audio samples in it")
    return TimeSeries(
        name=path.stem,
        time=np.arange(len(values)) / out_rate,
        values=values.astype(float),
        channels=names,
        units="",
        sample_rate=out_rate,
        metadata={"source": str(path), "source_rate": native,
                  "source_channels": channels, "decoder": decoder},
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


def _read_wav(path: Path, keep_channels: bool = False) -> tuple[np.ndarray, int, int]:
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
    channels = 1 if scaled.ndim == 1 else scaled.shape[1]
    if keep_channels or scaled.ndim == 1:
        return scaled, int(native), channels
    return scaled.mean(axis=1), int(native), channels


def _resample(values: np.ndarray, native: int, rate: int) -> np.ndarray:
    common = math.gcd(int(native), int(rate))
    return _signal.resample_poly(values, int(rate) // common, int(native) // common, axis=0)


def _ffmpeg(tool: str, path: Path) -> str:
    found = shutil.which(tool)
    if found is None:
        raise AudioError(
            f"{path}: decoding {path.suffix} needs {tool}, which is not on PATH; "
            "install ffmpeg (e.g. apt install ffmpeg) or convert the clip to .wav"
        )
    return found


def _decode(path: Path, rate: int | None,
            keep_channels: bool = False) -> tuple[np.ndarray, int, int]:
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
               "-vn", "-ac", str(channels if keep_channels else 1), "-f", "f32le"]
    if rate is not None:
        command += ["-ar", str(int(rate))]
    decoded = subprocess.run(command + ["-"], capture_output=True)
    if decoded.returncode != 0:
        reason = decoded.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["?"]
        raise AudioError(f"{path}: ffmpeg could not decode it ({reason[0]})")
    values = np.frombuffer(decoded.stdout, dtype="<f4")
    return values.reshape(-1, channels) if keep_channels else values, native, channels
