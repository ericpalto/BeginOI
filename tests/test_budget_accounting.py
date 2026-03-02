from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Budget, Program, ProgramBatch, ControlAction
from beginoi.core.runner import run_loop
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime
from beginoi.core.experiments.basic import BasicBatchExperiment


@dataclass
class DummyTheta:
    """Minimal theta container for dummy plant."""

    time: float = 0.0


class DummyPlant:
    """Deterministic plant with y = x1 + x2."""

    def reset(self, seed: int) -> DummyTheta:
        del seed
        return DummyTheta()

    def simulate(self, program: Program, theta: DummyTheta) -> float:
        del theta
        x = program.as_constant_inputs()
        return float(x[0] + x[1])

    def observe(self, program: Program, theta: DummyTheta, rng) -> float:
        del theta, rng
        x = program.as_constant_inputs()
        return float(x[0] + x[1])

    def apply_intervention(self, theta: DummyTheta, intervention, dt):
        del intervention, dt
        return theta

    def evaluate_heatmap(self, theta: DummyTheta, grid_spec):
        del theta, grid_spec
        raise NotImplementedError


class DummyBenchmark:
    """Placeholder benchmark object for runner signature."""

    name = "dummy"
    grid = None


class DummyPolicy:
    """Fixed policy that always selects the same constant input."""

    def init(self, seed: int, *, benchmark, regime) -> None:
        del seed, benchmark, regime

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        program = Program(kind="constant", u=np.array([0.25, 0.75], dtype=float))
        return ControlAction(batch=ProgramBatch(programs=[program]), intervention=None)

    def update(self, history, new_result) -> None:
        del history, new_result


def test_budget_accounting_stops_on_budget() -> None:
    regime = Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="standard_lab",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=1,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )
    experiment = BasicBatchExperiment(max_programs=1, cost_per_unit=1.0)
    out = run_loop(
        plant=DummyPlant(),
        experiment=experiment,
        policy=DummyPolicy(),
        regime=regime,
        benchmark=DummyBenchmark(),
        budget=Budget(total=2.0),
        metrics=[],
        rng=np.random.default_rng(0),
        loggers=[],
    )
    assert out.history.budget_spent == 2.0
    assert len(out.history.actions) == 2
    assert len(out.history.observations) == 2
