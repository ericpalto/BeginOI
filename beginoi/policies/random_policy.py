from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np

from beginoi.core.types import Program, Intervention, ProgramBatch, ControlAction


@dataclass
class RandomPolicy:
    """Baseline that samples programs uniformly at random.

    If exposure is enabled by the regime, also applies random exposures.
    """

    batch_size: int = 1
    low: float = 0.0
    high: float = 1.0
    timeseries_length: int = 8
    rng: np.random.Generator = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        del benchmark
        self.rng = np.random.default_rng(int(seed))
        self.regime = regime

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        k = min(int(self.batch_size), int(self.regime.max_programs_per_unit))
        programs: list[Program] = []
        for _ in range(k):
            if self.regime.program_kind == "constant":
                u = self.rng.uniform(self.low, self.high, size=(2,))
                programs.append(Program(kind="constant", u=u))
            else:
                u = self.rng.uniform(
                    self.low, self.high, size=(self.timeseries_length, 2)
                )
                programs.append(Program(kind="timeseries", u=u))

        intervention = None
        if "exposure_schedule" in self.regime.allowed_interventions():
            intervention = Intervention(
                kind="exposure_schedule",
                payload={"level": float(self.rng.uniform(0.0, 1.0))},
            )
        return ControlAction(
            batch=ProgramBatch(programs=programs), intervention=intervention
        )

    def update(self, history, new_result) -> None:
        del history, new_result
