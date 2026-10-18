"""mmp-sense -- look at signals, turn what you see into a detector.

    import mmp_sense

    session = mmp_sense.load("knocks.mcap")               # or the viewer's own session
    strength = mmp_sense.signal("ax", "ay", "az").highpass(2.0).magnitude().peak("60ms")
    session.values(strength, 4.0, 6.0)              # pandas, indexed by time

    knock = (strength > 0.7).events(refractory="145ms", name="knock")
    triple = knock.repeated(3, gap=("140ms", "450ms"), name="triple")
    session.events(triple)                          # a DataFrame of events
    session.show(triple)                            # drawn on the viewer's charts

The viewer (`python -m mmp_sense`) and the MCP server (`mmp-sense-mcp`) are adapters
over a session; see API.md.
"""

from . import io
from .analysis import from_frame, load
from .core.array import Array
from .core.bands import Band
from .core.graph import GraphError, Lookback, Node, beam, pointwise, register_op, signal
from .core.session import Session
from .core.signals import SignalError
from .sources import (
    CsvSource,
    FileSources,
    RecordingSource,
    SeriesSource,
    Source,
    SyntheticAccelerometer,
)

__version__ = "0.1.0"

__all__ = [
    "Array",
    "Band",
    "beam",
    "CsvSource",
    "FileSources",
    "GraphError",
    "Lookback",
    "Node",
    "RecordingSource",
    "SeriesSource",
    "Session",
    "SignalError",
    "Source",
    "SyntheticAccelerometer",
    "__version__",
    "from_frame",
    "load",
    "pointwise",
    "register_op",
    "signal",
]

