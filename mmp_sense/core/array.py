"""Where the microphones are: an array, by signal name.

    ring = mmp_sense.Array({"mic:m0": (0.0, 0.0), "mic:m1": (0.3, 0.0),
                      "mic:m2": (0.3, 0.3), "mic:m3": (0.0, 0.3)})

Positions are metres, all in two or all in three dimensions; `speed` is
the speed of sound, 343 m/s in air at 20 C.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Array", "SPEED_OF_SOUND"]

SPEED_OF_SOUND = 343.0


class Array:
    """Microphone positions, by the name of the signal each one gives."""

    def __init__(self, positions: dict, speed: float = SPEED_OF_SOUND) -> None:
        if len(positions) < 2:
            raise ValueError("an array needs at least two microphones")
        dims = {len(p) for p in positions.values()}
        if len(dims) != 1 or dims.pop() not in (2, 3):
            raise ValueError("give every position as (x, y) or every one as (x, y, z)")
        if speed <= 0:
            raise ValueError("the speed of sound must be positive")
        self.names = [str(n) for n in positions]
        self.points = np.array([[float(c) for c in p] for p in positions.values()])
        self.speed = float(speed)
        if not np.isfinite(self.points).all():
            raise ValueError("every position must be a number of metres")
        gaps = np.linalg.norm(self.points[:, None] - self.points[None], axis=2)
        np.fill_diagonal(gaps, np.inf)
        if gaps.min() == 0:
            raise ValueError("two microphones at one place tell nothing apart")

    @property
    def dims(self) -> int:
        return self.points.shape[1]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        n = len(self.names)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    @property
    def reach(self) -> float:
        """The longest a sound can take between two of them, in seconds."""
        gaps = self.points[:, None] - self.points[None]
        return float(np.linalg.norm(gaps, axis=2).max()) / self.speed

    def frozen(self) -> tuple:
        """The array as plain data, for a recipe's parameters."""
        return tuple(tuple(p) for p in self.points.tolist())

    def describe(self) -> dict:
        return {"positions": dict(zip(self.names, self.points.tolist())), "speed": self.speed}

    def __repr__(self) -> str:
        return f"<Array {len(self.names)} microphones, {self.dims}-D: {', '.join(self.names)}>"
