from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import (
    Observation,
    ProgramBatch,
    ControlAction,
    ExperimentResult,
)
from beginoi.core.interfaces import Plant, FeedbackHandler


@dataclass(frozen=True)
class BasicBatchExperiment:
    """One budget unit runs a batch of programs under the current plant state."""

    max_programs: int
    cost_per_unit: float = 1.0
    timeseries_extra_cost_per_step: float = 0.0
    streaming: bool = False

    def max_programs_per_unit(self) -> int:
        return int(self.max_programs)

    def cost_of(self, batch: ProgramBatch) -> float:
        cost = float(self.cost_per_unit)
        if self.timeseries_extra_cost_per_step <= 0:
            return cost
        extra = 0.0
        for program in batch.programs:
            if program.kind != "timeseries":
                continue
            u = np.asarray(program.u)
            if u.ndim == 2:
                extra += max(0, u.shape[0] - 1) * float(
                    self.timeseries_extra_cost_per_step
                )
        return cost + extra

    def run_budget_unit(
        self,
        plant: Plant,
        theta: Any,
        action: ControlAction,
        *,
        unit_id: int,
        feedback_handler: FeedbackHandler | None = None,
        rng: Any = None,
    ) -> ExperimentResult:
        if rng is None:
            rng = np.random.default_rng(0)
        if len(action.batch.programs) > self.max_programs_per_unit():
            raise ValueError("Batch exceeds experiment capacity.")

        if action.intervention is not None:
            theta = plant.apply_intervention(theta, action.intervention, dt=None)

        observations: list[Observation] = []
        for idx, program in enumerate(action.batch.programs):
            y = float(plant.observe(program, theta, rng=rng))
            noise_meta = dict(program.meta.get("noise_meta", {}))
            obs = Observation(
                program_id=f"u{unit_id}_p{idx}",
                inputs_summary={
                    "kind": program.kind,
                    "u": np.asarray(program.as_constant_inputs()).tolist(),
                },
                y=y,
                t_obs=float(getattr(theta, "time", 0.0)),
                noise_meta=noise_meta,
                unit_id=int(unit_id),
                replicate_id=int(idx),
                extra={},
            )
            observations.append(obs)

            if self.streaming and feedback_handler is not None:
                maybe = feedback_handler(obs)
                if maybe is not None:
                    theta = plant.apply_intervention(theta, maybe, dt=None)

        consumed_cost = float(self.cost_of(action.batch))
        return ExperimentResult(
            consumed_cost=consumed_cost, observations=observations, final_theta=theta
        )
