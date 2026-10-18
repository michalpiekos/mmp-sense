"""Analyses as graphs of inspectable, time-aware nodes.

    node.py      Node: an op, its parameters and inputs; `signal()` makes a leaf
    ops.py       the op registry, and the ops mmp-sense comes with
    series.py    Series, what a numeric node computes, and its Timing
    evaluate.py  the Evaluator, and `run` for a whole graph over some data
    incremental.py  how a live result is carried on to new samples

A graph describes a computation and holds no data, so the same one runs
over a live buffer, over a recording and in a test. Nothing here imports
Qt.
"""

from .evaluate import Evaluator, GraphError, RunResult, run
from .incremental import Lookback, Rule, pointwise
from .node import Node, beam, signal
from .ops import EVENT_COLUMNS, Op, op, ops, register_op
from .series import Series, Timing, hold, seconds

__all__ = [
    "EVENT_COLUMNS",
    "Evaluator",
    "GraphError",
    "Lookback",
    "Node",
    "Op",
    "Rule",
    "RunResult",
    "Series",
    "Timing",
    "hold",
    "op",
    "ops",
    "pointwise",
    "register_op",
    "run",
    "seconds",
    "signal",
    "beam",
]
