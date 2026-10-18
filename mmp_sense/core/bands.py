"""Frequency bands: stretches of the spectrum picked out to be heard alone.

A session holds a list of them. The ticked ones are what the window's
Listen plays through; none ticked, it plays everything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Band", "check_band", "ranges"]


@dataclass(frozen=True)
class Band:
    """`low_hz` to `high_hz`, called `name`, heard when `on`."""

    name: str
    low_hz: float
    high_hz: float
    on: bool = True

    @property
    def range(self) -> tuple[float, float]:
        return (self.low_hz, self.high_hz)


def check_band(low_hz, high_hz) -> tuple[float, float]:
    """The pair as floats, or a ValueError saying what is wrong with it."""
    low, high = float(low_hz), float(high_hz)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("a band's edges must be finite numbers")
    if low < 0:
        raise ValueError("a band cannot start below 0 Hz")
    if low >= high:
        raise ValueError(f"a band must end above where it starts ({low:g} >= {high:g} Hz)")
    return low, high


def ranges(bands) -> list[tuple[float, float]]:
    """(low, high) pairs from bands or pairs, each checked."""
    out = []
    for band in bands:
        low, high = band.range if isinstance(band, Band) else band
        out.append(check_band(low, high))
    return out
