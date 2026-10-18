"""Value over time and frequency."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .base import Dataset

__all__ = ["Spectrogram"]


@dataclass
class Spectrogram(Dataset):
    """A magnitude per (frequency, time) cell.

    `values` is (n_frequencies, n_times) -- frequency first, because that
    is the orientation every STFT returns it in and the one pyqtgraph and
    matplotlib draw.
    """

    time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frequency: np.ndarray = field(default_factory=lambda: np.zeros(0))
    values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    units: str = ""
    frequency_units: str = "Hz"

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.frequency = np.asarray(self.frequency, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2:
            raise ValueError("values must be 2-D (frequencies, times)")
        if self.values.shape != (len(self.frequency), len(self.time)):
            raise ValueError(
                f"values {self.values.shape} do not match "
                f"{len(self.frequency)} frequencies by {len(self.time)} times"
            )
        self.preferred_view = "spectrogram"

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.values.shape)

    def time_slice(self, t0: float, t1: float) -> "Spectrogram":
        if t1 < t0:
            t0, t1 = t1, t0
        keep = (self.time >= t0) & (self.time <= t1)
        return replace(self, time=self.time[keep], values=self.values[:, keep])

    def decibels(self, floor_ratio: float = 1e-6) -> np.ndarray:
        """20*log10 of the values, floored relative to the peak.

        A linear spectrogram of anything with a wide dynamic range draws
        as one bright line and nothing else, so every consumer wants this
        and it is better computed once here.
        """
        if not self.values.size:
            return self.values
        floor = max(float(self.values.max()) * floor_ratio, float(np.finfo(float).tiny))
        return 20.0 * np.log10(np.maximum(self.values, floor))

    def summary(self) -> str:
        rows, cols = self.shape
        if not rows or not cols:
            return f"{self.name}: empty"
        return (
            f"{self.name}: {rows} frequencies x {cols} times, "
            f"{self.frequency[0]:.4g}-{self.frequency[-1]:.4g} "
            f"{self.frequency_units}, {self.time[0]:.4g}-{self.time[-1]:.4g} s"
        )
