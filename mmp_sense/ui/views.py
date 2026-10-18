"""The kinds of pane the window can offer, by name.

A pane type registers itself here, so a module outside mmp-sense can add one
without editing mmp-sense:

    register_view("histogram", HistogramPane, description="Distribution")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["View", "register_view", "view", "views"]


@dataclass
class View:
    """A pane type, by name."""

    name: str
    factory: Callable[..., Any]
    accepts: tuple[str, ...] = ()
    description: str = ""


_VIEWS: dict[str, View] = {}


def register_view(name: str, factory, *, accepts=(), description="") -> View:
    _VIEWS[name] = View(name, factory, tuple(accepts), description)
    return _VIEWS[name]


def views() -> dict[str, View]:
    return dict(_VIEWS)


def view(name: str) -> View:
    try:
        return _VIEWS[name]
    except KeyError:
        raise KeyError(f"no view {name!r}; have {', '.join(sorted(_VIEWS)) or 'none'}") from None
