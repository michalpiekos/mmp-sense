"""Results a script wants on screen: named (x, y) curves.

A plot is not a signal: it has no stream, no time axis and no recipe, so
it stays out of the signal table, and showing a new result under a name
replaces the old one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Plot", "make_plot"]


@dataclass(frozen=True, eq=False)
class Plot:
    """One named curve. The arrays are read-only copies."""

    name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x_label: str = ""
    x_units: str = ""
    y_label: str = ""
    y_units: str = ""


def _column(name: str, axis: str, data: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(data):
        raise ValueError(f"plot {name!r}: {axis} is complex; plot its magnitude or phase")
    values = np.array(data, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"plot {name!r}: {axis} must be 1-D, not {values.ndim}-D")
    values.setflags(write=False)
    return values


def make_plot(name: str, x: ArrayLike, y: ArrayLike, *, x_label: str = "", x_units: str = "",
              y_label: str = "", y_units: str = "") -> Plot:
    """A checked `Plot`; a ValueError says what is wrong with the data."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"a plot needs a name, not {name!r}")
    xs, ys = _column(name, "x", x), _column(name, "y", y)
    if len(xs) != len(ys):
        raise ValueError(f"plot {name!r}: x has {len(xs)} values, y has {len(ys)}")
    return Plot(name, xs, ys, x_label, x_units, y_label, y_units)
