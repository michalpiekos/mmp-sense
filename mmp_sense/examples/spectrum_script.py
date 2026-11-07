"""A script for the window: the spectrum of the selection, as a chart.

    mmp-sense-send --script mmp_sense/examples/spectrum_script.py

Select a stretch, send this, select another, send it again: the same
chart updates. Copy the file anywhere; it needs only the installed package.
"""

from __future__ import annotations

import numpy as np

from mmp_sense import Session


def main(session: Session) -> None:
    samples = session.samples(session.channels[0])
    magnitude = np.abs(np.fft.rfft(samples.values - samples.values.mean()))
    frequency = np.fft.rfftfreq(len(samples.values), 1.0 / samples.rate)
    session.plot(f"{samples.name} spectrum", frequency, magnitude,
                 x_label="Frequency", x_units="Hz",
                 y_label="Magnitude", y_units=samples.units)
