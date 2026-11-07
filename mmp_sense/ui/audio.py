"""Sound out of the speakers: what the window's Play button sounds through.

A `Player` takes float samples within -1..1 and plays them once, at
the output device's own rate and channel count -- 16-bit where the
device takes it, else floats -- the same sound in every channel. A
machine without a sound device, or a Qt without its multimedia module,
gets a player that says it cannot play, and Play runs silently.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, Signal

__all__ = ["Player", "make_player"]

#: The rate asked for when the device does not say what it prefers.
DEFAULT_RATE = 48000


class Player(QObject):
    """Plays one sound at a time; `finished` when it ends or is stopped."""

    finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sink = None
        self._buffer = None
        self.device = None
        #: Why nothing can be played, or "" when something can.
        self.problem = ""
        try:
            from PySide6.QtMultimedia import QMediaDevices
        except ImportError:
            self.problem = "Qt's multimedia module is not installed"
            return
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.problem = "no sound output device"
            return
        self.device = device

    @property
    def available(self) -> bool:
        return not self.problem

    @property
    def rate(self) -> int:
        if self.device is None:
            return DEFAULT_RATE
        return int(self.device.preferredFormat().sampleRate() or DEFAULT_RATE)

    @property
    def playing(self) -> bool:
        return self._sink is not None

    def _format(self):
        """The device's preferred format, as 16-bit samples if it takes them."""
        from PySide6.QtMultimedia import QAudioFormat

        audio_format = self.device.preferredFormat()
        audio_format.setSampleRate(self.rate)
        wanted = QAudioFormat()
        wanted.setSampleRate(self.rate)
        wanted.setChannelCount(audio_format.channelCount())
        wanted.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if self.device.isFormatSupported(wanted):
            return wanted
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        return audio_format

    def play(self, samples: np.ndarray) -> None:
        """Play `samples` (at `rate`), stopping whatever was playing."""
        from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink

        self.stop(quietly=True)
        if not self.available:
            raise RuntimeError(f"cannot play sound: {self.problem}")
        audio_format = self._format()
        sound = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        if audio_format.sampleFormat() == QAudioFormat.SampleFormat.Int16:
            sound = np.round(sound * 32767).astype("<i2")
        frames = np.repeat(sound[:, None], max(1, audio_format.channelCount()), axis=1)
        self._buffer = QBuffer(self)
        self._buffer.setData(QByteArray(frames.tobytes()))
        self._buffer.open(QIODevice.ReadOnly)
        self._sink = QAudioSink(self.device, audio_format, self)
        self._sink.stateChanged.connect(
            lambda state: self.stop() if state == QAudio.State.IdleState else None)
        self._sink.start(self._buffer)

    def stop(self, quietly: bool = False) -> None:
        """Stop playing; `finished` is emitted unless `quietly`."""
        if self._sink is None:
            return
        sink, buffer = self._sink, self._buffer
        self._sink = self._buffer = None
        sink.stop()
        sink.deleteLater()
        buffer.close()
        buffer.deleteLater()
        if not quietly:
            self.finished.emit()


def make_player(parent=None) -> Player:
    return Player(parent)
