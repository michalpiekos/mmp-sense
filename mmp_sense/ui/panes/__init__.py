"""The panes, and the names the GUI offers them under.

A pane draws; it never computes. Adding one is subclassing `Pane`,
implementing `refresh`, and registering it here so it appears in the
menu -- the registry is the core's, so a module outside mmp-sense can add a
pane without editing this file.
"""

from ..views import register_view
from .base import Pane
from .correlation_pane import CorrelationPane
from .direction_pane import DirectionPane
from .psd_pane import PsdPane
from .signal_pane import SignalPane
from .spectrogram_pane import SpectrogramPane
from .spectrum_pane import SpectrumPane

register_view("signal", SignalPane, accepts=("timeseries",),
              description="Signals against time")
register_view("spectrum", SpectrumPane, accepts=("spectrum",),
              description="The spectrum of the cursor window")
register_view("spectrogram", SpectrogramPane, accepts=("spectrogram",),
              description="Frequency over time")
register_view("psd", PsdPane, accepts=(),
              description="The averaged spectrum: power per Hz, in dB")
register_view("direction", DirectionPane, accepts=(),
              description="Where a sound comes from: an array's power by azimuth")
register_view("correlation", CorrelationPane, accepts=(),
              description="Two signals' correlation against lag: their delay")

__all__ = ["CorrelationPane", "DirectionPane", "Pane", "PsdPane", "SignalPane", "SpectrogramPane", "SpectrumPane"]
