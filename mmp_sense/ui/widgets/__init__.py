"""Docks: the signals of a session, its detectors, its frequency bands and the workspace."""

from .band_dock import BandDock
from .detector_dock import DetectorDock
from .signal_dock import SignalDock
from .workspace_dock import WorkspaceDock

__all__ = ["BandDock", "DetectorDock", "SignalDock", "WorkspaceDock"]
