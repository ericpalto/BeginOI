from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np

from beginoi.core.types import Program, ProgramBatch, ControlAction


@dataclass
class GridPolicy:
    """Non-adaptive baseline that sweeps the benchmark grid in order."""

    batch_size: int = 1
    cursor: int = 0
    points: np.ndarray = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        del seed
        self.regime = regime
        grid = benchmark.grid
        xs = np.array([(x1, x2) for x1 in grid.x1 for x2 in grid.x2], dtype=float)
        self.points = xs
        self.cursor = 0

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        k = min(int(self.batch_size), int(self.regime.max_programs_per_unit))
        programs: list[Program] = []
        for _ in range(k):
            x = self.points[self.cursor % len(self.points)]
            self.cursor += 1
            programs.append(
                Program(kind=self.regime.program_kind, u=np.asarray(x, dtype=float))
            )
        return ControlAction(batch=ProgramBatch(programs=programs), intervention=None)

    def update(self, history, new_result) -> None:
        del history, new_result
