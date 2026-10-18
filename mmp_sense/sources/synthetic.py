"""A simulated accelerometer, so there is always something to look at.

Three axes, gravity on z, noise everywhere, an optional slow sway, and
transients -- the knocks. Knocks arrive on their own at random, and
`knock()` puts one in on demand, which is what makes the thing usable as
a demonstration: hit the key, see the transient, move the cursor onto it.

Samples are produced against a clock, so the stream is real-time by
default and exactly reproducible in a test that passes a clock of its
own.
"""

from __future__ import annotations

import time as _time

import numpy as np

from .base import EMPTY, Source

__all__ = ["SyntheticAccelerometer"]


class SyntheticAccelerometer(Source):
    """Accelerometer-shaped data at `rate` hertz."""

    kind = "synthetic"

    def __init__(
        self,
        *,
        name: str = "synthetic",
        rate: float = 1000.0,
        noise: float = 0.02,
        gravity: float = 9.81,
        sway_frequency: float = 0.4,
        sway_amplitude: float = 0.1,
        knock_interval: float = 4.0,
        knock_amplitude: float = 4.0,
        knock_frequency: float = 90.0,
        knock_duration: float = 0.08,
        seed: int = 0,
        clock=_time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.name = name
        self.rate = float(rate)
        self.channels = ["accel_x", "accel_y", "accel_z"]
        self.units = "m/s^2"
        self.noise = float(noise)
        self.gravity = float(gravity)
        self.sway_frequency = float(sway_frequency)
        self.sway_amplitude = float(sway_amplitude)
        self.knock_interval = float(knock_interval)
        self.knock_amplitude = float(knock_amplitude)
        self.knock_frequency = float(knock_frequency)
        self.knock_duration = float(knock_duration)
        self.clock = clock
        self._rng = np.random.default_rng(seed)
        self._started: float | None = None
        self._produced = 0
        #: (centre, amplitude, frequency, duration) of every knock, kept as
        #: ground truth -- what a detector is supposed to find.
        self.knocks: list[tuple[float, float, float, float]] = []
        self._next_knock = 0.0

    # -- lifecycle -----------------------------------------------------

    def start(self) -> "SyntheticAccelerometer":
        if self._started is None:
            self._started = self.clock()
            self._next_knock = self._schedule()
        return self

    def stop(self) -> None:
        self._started = None

    def restart(self) -> "SyntheticAccelerometer":
        # The sample count and the knocks are what start() cannot put
        # back on its own: left alone the count outruns the restarted
        # clock and read() returns nothing for as long as the last run
        # lasted, and the old knocks land again on the new samples.
        self.stop()
        self._produced = 0
        self.knocks.clear()
        return self.start()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else self.clock() - self._started

    def _schedule(self, after: float | None = None) -> float:
        """When the next knock falls, counted from the last one.

        From the last knock rather than from now, so a chunk covering
        several intervals gets several knocks rather than one.
        """
        if self.knock_interval <= 0:
            return float("inf")
        start = self.elapsed if after is None else after
        return start + float(
            self._rng.uniform(0.5 * self.knock_interval, 1.5 * self.knock_interval)
        )

    # -- producing -----------------------------------------------------

    def knock(self, amplitude: float | None = None, *, at: float | None = None,
              frequency: float | None = None, duration: float | None = None) -> float:
        """Put a transient in, now or at a given time. Returns its centre."""
        centre = self.elapsed if at is None else float(at)
        self.knocks.append((
            centre,
            float(self.knock_amplitude if amplitude is None else amplitude),
            float(self.knock_frequency if frequency is None else frequency),
            float(self.knock_duration if duration is None else duration),
        ))
        return centre

    def event_spans(self) -> list[tuple[float, float]]:
        """(start, end) of every knock so far -- the ground truth."""
        return [(c - d, c + d) for c, _a, _f, d in self.knocks]

    def read(self):
        if self._started is None:
            return EMPTY
        wanted = int(self.elapsed * self.rate)
        n = wanted - self._produced
        if n <= 0:
            return EMPTY
        # Bursts overlap chunk boundaries, so a knock is evaluated against
        # absolute time rather than against this chunk's samples.
        start = self._produced
        self._produced = wanted
        time = (start + np.arange(n)) / self.rate

        values = self._rng.normal(0.0, self.noise, size=(n, 3))
        values[:, 2] += self.gravity
        if self.sway_frequency > 0:
            sway = self.sway_amplitude * np.sin(2 * np.pi * self.sway_frequency * time)
            values[:, 0] += sway
            values[:, 1] += 0.5 * sway

        while self.elapsed >= self._next_knock:
            self.knock(at=self._next_knock)
            self._next_knock = self._schedule(self._next_knock)

        for centre, amplitude, frequency, duration in self.knocks:
            if time[-1] < centre - duration or time[0] > centre + duration:
                continue
            envelope = np.exp(-0.5 * ((time - centre) / (duration / 3.0)) ** 2)
            burst = amplitude * envelope * np.sin(
                2 * np.pi * frequency * (time - centre)
            )
            values[:, 0] += burst
            values[:, 1] += 0.7 * burst
            values[:, 2] += 0.4 * burst
        return time, values
