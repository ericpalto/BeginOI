from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np

from beginoi.core.types import Program, Intervention, ProgramBatch, ControlAction


@dataclass
class DriftAwarePolicy:
    """Baseline that keeps exposure low and samples space-filling inputs."""

    batch_size: int = 1
    rng: np.random.Generator = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        del benchmark
        self.rng = np.random.default_rng(int(seed))
        self.regime = regime

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        k = min(int(self.batch_size), int(self.regime.max_programs_per_unit))
        programs = [
            Program(
                kind=self.regime.program_kind, u=self.rng.uniform(0.0, 1.0, size=(2,))
            )
            for _ in range(k)
        ]
        intervention = None
        if "exposure_schedule" in self.regime.allowed_interventions():
            intervention = Intervention(
                kind="exposure_schedule", payload={"level": 0.0, "duration": 1.0}
            )
        return ControlAction(
            batch=ProgramBatch(programs=programs), intervention=intervention
        )

    def update(self, history, new_result) -> None:
        del history, new_result
