"""What a source is.

A source produces samples and says what they are. It does not buffer,
process or draw: the session owns the buffer, features do the processing
and panes do the drawing.

    start()   idempotent; after it, channels, rate and units are final
    read()    (time, values) for whatever is ready -- never blocking
    stop()    release whatever start took
    restart()  begin again, producing nothing made before the call

`read` returns seconds and an (n, len(channels)) array, with time
non-decreasing and in the session's own time base: seconds since the
source started, or since its own epoch for a live stream.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Source", "EMPTY"]

EMPTY = (np.zeros(0), np.zeros((0, 0)))


class Source:
    """The contract, and sensible do-nothing behaviour for the rest."""

    #: What the session calls this dataset.
    name: str = "source"
    #: "synthetic", "live" or "recording" -- for a tool asking what it has.
    kind: str = "synthetic"
    channels: list[str] = []
    units: str = ""
    rate: float = 0.0
    #: What a sample is: the buffer holding them keeps this type.
    dtype = np.dtype(np.float64)

    def start(self) -> "Source":
        return self

    def read(self):
        return EMPTY

    def stop(self) -> None:
        pass

    def restart(self) -> "Source":
        """Begin again from nothing.

        The session empties its buffer when it starts a source again, so
        a source must hand over nothing it made before this call -- and
        must not refuse to produce because of what it made before
        either. Stopping and starting is enough for a source that keeps
        no count of its own.
        """
        self.stop()
        return self.start()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "channels": list(self.channels),
            "units": self.units,
            "rate": float(self.rate),
            "dtype": str(np.dtype(self.dtype)),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}: {', '.join(self.channels)}>"
