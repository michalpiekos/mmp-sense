"""Where samples come from: simulated, or read from a file.

    SyntheticAccelerometer   three axes and knocks, against a clock
    FileSources              whatever a data file holds, a source per series
    RecordingSource          a topic of an MCAP recording
    CsvSource                delimited text of time and channels
    SeriesSource             any TimeSeries, replayed or at once

All satisfy the same small contract in `base.py`, so the session -- and
everything built on it -- cannot tell which it has. A live source is
one more implementation of it.
"""

from .base import Source
from .recording import CsvSource, FileSources, RecordingSource, SeriesSource
from .synthetic import SyntheticAccelerometer

__all__ = [
    "CsvSource",
    "FileSources",
    "RecordingSource",
    "SeriesSource",
    "Source",
    "SyntheticAccelerometer",
]
