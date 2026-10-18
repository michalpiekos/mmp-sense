"""Events and labelled regions on the time axis."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from .base import Dataset

__all__ = ["EventSeries", "COLUMNS"]

#: Every event table has these, whoever made it. An instant is a region
#: whose ends are equal, so consumers need only one shape.
COLUMNS = ["start_time", "end_time", "label", "confidence"]


@dataclass
class EventSeries(Dataset):
    """A table of (start, end, label, confidence) rows, plus free columns.

    What a detector emits and what a human labelling a recording
    produces, deliberately the same type: comparing the two is the point
    of the exercise, and it is much easier when they are not two types.
    """

    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNS))
    units: str = "s"

    def __post_init__(self) -> None:
        table = self.table if self.table is not None else pd.DataFrame()
        for column in [c for c in COLUMNS if c not in table.columns]:
            table[column] = 1.0 if column == "confidence" else (
                "" if column == "label" else float("nan")
            )
        rest = [c for c in table.columns if c not in COLUMNS]
        self.table = table[COLUMNS + rest].reset_index(drop=True)
        self.preferred_view = "events"

    @classmethod
    def from_records(cls, records, **kwargs) -> "EventSeries":
        return cls(table=pd.DataFrame(list(records)), **kwargs)

    @classmethod
    def from_spans(cls, spans, label: str = "", **kwargs) -> "EventSeries":
        return cls.from_records(
            [
                {"start_time": float(a), "end_time": float(b),
                 "label": label, "confidence": 1.0}
                for a, b in spans
            ],
            **kwargs,
        )

    def add(
        self,
        start_time: float,
        end_time: float | None = None,
        label: str = "",
        confidence: float = 1.0,
        **extra: Any,
    ) -> "EventSeries":
        """A copy with one more row. Datasets are values, so labelling
        never mutates what a pane is drawing."""
        row = {
            "start_time": float(start_time),
            "end_time": float(start_time if end_time is None else end_time),
            "label": label,
            "confidence": float(confidence),
            **extra,
        }
        return replace(self, table=pd.concat(
            [self.table, pd.DataFrame([row])], ignore_index=True))

    def in_range(self, t0: float, t1: float) -> "EventSeries":
        """Rows overlapping [t0, t1] at all, not only those contained."""
        if t1 < t0:
            t0, t1 = t1, t0
        table = self.table
        overlap = (table["end_time"] >= t0) & (table["start_time"] <= t1)
        return replace(self, table=table[overlap].reset_index(drop=True))

    def to_records(self) -> list[dict]:
        return [
            {k: (float(v) if k in ("start_time", "end_time", "confidence") else v)
             for k, v in row.items()}
            for row in self.table.to_dict("records")
        ]

    def __len__(self) -> int:
        return len(self.table)

    def summary(self) -> str:
        if not len(self.table):
            return f"{self.name}: no events"
        labels = ", ".join(sorted(set(self.table["label"].astype(str))))
        return f"{self.name}: {len(self.table)} events [{labels}]"
