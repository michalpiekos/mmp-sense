"""What every dataset carries, whoever produced it.

A dataset is a value: a module takes datasets and returns new ones rather
than mutating what it was given. The arrays inside are ordinary NumPy
arrays and nothing stops you writing to them, but no part of mmp-sense does.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["Provenance", "Dataset"]


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from.

    Enough to say what produced a result and with which settings, so a
    chart, a log line or a saved recipe can be traced back to it. It
    deliberately does not hold the input *data*, only names: provenance
    travels with results that may outlive the datasets they came from.
    """

    module: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()

    def describe(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        source = ", ".join(self.inputs) if self.inputs else "-"
        return f"{self.module}({args}) <- {source}"


@dataclass
class Dataset:
    """Base of the typed data model.

    Subclasses add the coordinates and values they need. What is common is
    a name, free-form metadata, provenance, and a hint about how it would
    like to be drawn -- a hint only; panes may ignore it.
    """

    name: str = "dataset"
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None

    #: What kind of pane suits this dataset. A hint for the GUI, not a
    #: promise, and never rendering code.
    preferred_view: str = field(default="text", init=False, repr=False)

    def with_provenance(self, provenance: Provenance) -> "Dataset":
        """A copy carrying `provenance`. Datasets are values; this is how
        the module base class stamps a result without touching the one a
        module returned."""
        return replace(self, provenance=provenance)

    def summary(self) -> str:
        """One line for a status bar or a log."""
        return f"{self.name}: {type(self).__name__}"
