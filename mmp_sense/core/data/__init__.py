"""mmp-sense's typed data model.

Four types, because four kinds of thing come out of signal processing and
squeezing them into one table loses what makes each useful:

    TimeSeries      values against time, one or more aligned channels
    Spectrum        values against frequency
    Spectrogram     values against time and frequency
    EventSeries     events and labelled regions

They share `Dataset`: a name, metadata, provenance and a view hint. All
are NumPy or pandas values -- nothing here is the live storage engine,
which is `core.buffer`.
"""

from .base import Dataset, Provenance
from .events import COLUMNS as EVENT_COLUMNS, EventSeries
from .spectrogram import Spectrogram
from .spectrum import Spectrum
from .timeseries import TimeSeries

__all__ = [
    "Dataset",
    "EVENT_COLUMNS",
    "EventSeries",
    "Provenance",
    "Spectrogram",
    "Spectrum",
    "TimeSeries",
]
