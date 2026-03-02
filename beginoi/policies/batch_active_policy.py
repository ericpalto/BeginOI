from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np

from beginoi.core.types import Program, ProgramBatch, ControlAction


@dataclass
class BatchActivePolicy:
    """Simple space-filling batch selection (farthest-point sampling)."""

    batch_size: int = 4
    candidate_pool: int = 256
    rng: np.random.Generator = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        del benchmark
        self.rng = np.random.default_rng(int(seed))
        self.regime = regime

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del budget_remaining
        k = min(int(self.batch_size), int(self.regime.max_programs_per_unit))
        candidates = self.rng.uniform(0.0, 1.0, size=(int(self.candidate_pool), 2))
        if not history.observations:
            picked = candidates[:k]
        else:
            obs_x = np.array(
                [o.inputs_summary["u"] for o in history.observations], dtype=float
            )
            picked = []
            remaining = candidates.copy()
            for _ in range(k):
                dists = np.min(
                    np.sum((remaining[:, None, :] - obs_x[None, :, :]) ** 2, axis=2),
                    axis=1,
                )
                idx = int(np.argmax(dists))
                picked.append(remaining[idx])
                obs_x = np.vstack([obs_x, remaining[idx][None, :]])
                remaining = np.delete(remaining, idx, axis=0)
            picked = np.array(picked, dtype=float)

        programs = [Program(kind=self.regime.program_kind, u=p) for p in picked]
        return ControlAction(batch=ProgramBatch(programs=programs), intervention=None)

    def update(self, history, new_result) -> None:
        del history, new_result
