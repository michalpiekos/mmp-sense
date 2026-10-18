"""One-dimensional frequency-domain results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Dataset

__all__ = ["Spectrum"]


@dataclass
class Spectrum(Dataset):
    """Amplitude, power or any other value against frequency.

    Shaped like TimeSeries -- (n_frequencies, n_channels) -- so the two
    read the same way. What the values *mean* lives in `units` and in the
    metadata the producing module records: window, FFT size, and the time
    range the transform was taken over, without which a spectrum from a
    selection cannot be interpreted later.
    """

    frequency: np.ndarray = field(default_factory=lambda: np.zeros(0))
    values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    channels: list[str] = field(default_factory=list)
    units: str = ""
    frequency_units: str = "Hz"
    #: The (start, end) seconds the transform covered, when it came from a
    #: time series. None when the spectrum was built some other way.
    source_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("values must be 1-D or 2-D (bins, channels)")
        self.values = values

        if len(self.frequency) != len(self.values):
            raise ValueError(
                f"{len(self.frequency)} frequencies for {len(self.values)} bins"
            )
        if not self.channels:
            self.channels = [f"c{i}" for i in range(self.values.shape[1])]
        if len(self.channels) != self.values.shape[1]:
            raise ValueError("channel names do not match the values")
        self.preferred_view = "spectrum"

    @property
    def n_channels(self) -> int:
        return self.values.shape[1]

    def channel(self, which: str | int) -> np.ndarray:
        if isinstance(which, (int, np.integer)):
            return self.values[:, int(which)]
        return self.values[:, self.channels.index(which)]

    def peak_frequency(self, channel: str | int = 0) -> float:
        """The frequency of the largest bin.

        Bin 0 is included: a spectrum that was not detrended has its mean
        there, and hiding that would make a mistake harder to see, not
        less likely.
        """
        values = self.channel(channel)
        if not len(values):
            return float("nan")
        return float(self.frequency[int(np.argmax(values))])

    @property
    def resolution(self) -> float:
        """Hertz per bin."""
        if len(self.frequency) < 2:
            return float("nan")
        return float(self.frequency[1] - self.frequency[0])

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.values,
            columns=self.channels,
            index=pd.Index(self.frequency, name="frequency"),
        )

    def summary(self) -> str:
        span = ""
        if self.source_range:
            span = f", from {self.source_range[0]:.6g}-{self.source_range[1]:.6g} s"
        return (
            f"{self.name}: {len(self.frequency)} bins x {self.n_channels} "
            f"channels, {self.resolution:.4g} {self.frequency_units}/bin{span}"
        )
