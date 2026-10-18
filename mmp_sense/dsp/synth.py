"""Synthetic signals, so mmp-sense has something to show with no file to hand."""

from __future__ import annotations

import numpy as np

__all__ = ["synthetic_accelerometer"]


def synthetic_accelerometer(
    *,
    duration: float = 20.0,
    rate: float = 200.0,
    noise: float = 0.02,
    n_events: int = 4,
    event_frequency: float = 25.0,
    event_amplitude: float = 1.0,
    event_duration: float = 0.25,
    periodic_frequency: float = 0.0,
    periodic_amplitude: float = 0.1,
    gravity: float = 9.81,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Three axes of plausible accelerometer data.

    Returns (time, values, events): values is (n, 3) for x, y, z, and
    events lists the (start, end) of each transient injected -- ground
    truth, which is what makes this useful for judging a detector rather
    than only for filling a chart.

    The signal is deliberately simple and separable: gravity on z, white
    noise everywhere, an optional slow sway, and Gaussian-enveloped
    bursts at `event_frequency`. Every one of those shows up somewhere
    different in the demo pipeline, which is how you tell the pipeline is
    wired up correctly.
    """
    if duration <= 0 or rate <= 0:
        raise ValueError("duration and rate must be positive")
    rng = np.random.default_rng(seed)
    n = int(round(duration * rate))
    time = np.arange(n) / rate

    values = rng.normal(0.0, noise, size=(n, 3))
    values[:, 2] += gravity

    if periodic_frequency > 0:
        sway = periodic_amplitude * np.sin(2 * np.pi * periodic_frequency * time)
        values[:, 0] += sway
        values[:, 1] += 0.5 * sway

    # Events are spread evenly with a little jitter, so they neither line
    # up with the window grid of WindowEnergy nor look machine-placed.
    events: list[tuple[float, float]] = []
    if n_events > 0:
        margin = max(event_duration * 3, duration * 0.05)
        centres = np.linspace(margin, duration - margin, n_events)
        centres = centres + rng.uniform(-0.2, 0.2, size=n_events)
        for centre in centres:
            envelope = np.exp(-0.5 * ((time - centre) / (event_duration / 3)) ** 2)
            burst = (
                event_amplitude
                * envelope
                * np.sin(2 * np.pi * event_frequency * (time - centre))
            )
            values[:, 0] += burst
            values[:, 1] += 0.7 * burst
            values[:, 2] += 0.4 * burst
            events.append(
                (float(centre - event_duration / 2), float(centre + event_duration / 2))
            )
    return time, values, events
