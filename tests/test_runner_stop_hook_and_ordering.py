from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beginoi.core.types import (
    Budget,
    Program,
    Intervention,
    ProgramBatch,
    ControlAction,
)
from beginoi.core.runner import run_loop
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime
from beginoi.core.experiments.basic import BasicBatchExperiment


@dataclass
class DummyTheta:
    theta: np.ndarray
    time: float = 0.0


class DummyPlant:
    def __init__(self) -> None:
        self.observed_thetas: list[np.ndarray] = []

    def reset(self, seed: int) -> DummyTheta:
        del seed
        return DummyTheta(theta=np.array([0.0, 0.0], dtype=float))

    def simulate(self, program: Program, theta: DummyTheta) -> float:
        del theta
        u = program.as_constant_inputs()
        return float(np.sum(u))

    def observe(self, program: Program, theta: DummyTheta, rng) -> float:
        del rng
        self.observed_thetas.append(np.asarray(theta.theta, dtype=float).copy())
        u = program.as_constant_inputs()
        return float(np.sum(u) + np.sum(theta.theta))

    def apply_intervention(self, theta: DummyTheta, intervention, dt):
        del dt
        if intervention is None:
            return theta
        if intervention.kind == "theta_edit":
            vec = np.asarray(intervention.payload["set"]["theta"], dtype=float)
            return DummyTheta(theta=vec, time=float(theta.time))
        return theta

    def evaluate_heatmap(self, theta: DummyTheta, grid_spec):
        del theta, grid_spec
        raise NotImplementedError


class DummyBenchmark:
    name = "dummy"
    grid = None


class StopHookPolicy:
    def __init__(self) -> None:
        self.updates = 0

    def init(self, seed: int, *, benchmark, regime) -> None:
        del seed, benchmark, regime
        self.updates = 0

    def should_stop(self, history, *, budget_remaining: float) -> bool:
        del history, budget_remaining
        return bool(self.updates >= 1)

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        p = Program(kind="constant", u=np.array([0.2, 0.3], dtype=float))
        return ControlAction(batch=ProgramBatch(programs=[p]), intervention=None)

    def update(self, history, new_result) -> None:
        del history, new_result
        self.updates += 1


class EditOrderingPolicy:
    def init(self, seed: int, *, benchmark, regime) -> None:
        del seed, benchmark, regime

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        p = Program(kind="constant", u=np.array([0.2, 0.3], dtype=float))
        return ControlAction(
            batch=ProgramBatch(programs=[p]),
            intervention=Intervention(
                kind="theta_edit",
                payload={"set": {"theta": [1.0, 2.0]}},
            ),
        )

    def update(self, history, new_result) -> None:
        del history, new_result


def _regime() -> Regime:
    return Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="standard_lab",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=4,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )


def test_runner_optional_should_stop_hook_exits_early() -> None:
    out = run_loop(
        plant=DummyPlant(),
        experiment=BasicBatchExperiment(max_programs=4, cost_per_unit=1.0),
        policy=StopHookPolicy(),
        regime=_regime(),
        benchmark=DummyBenchmark(),
        budget=Budget(total=5.0),
        metrics=[],
        rng=np.random.default_rng(0),
        loggers=[],
    )
    assert len(out.history.actions) == 1
    assert out.history.budget_spent == 1.0


def test_basic_experiment_applies_edit_before_observe() -> None:
    plant = DummyPlant()
    out = run_loop(
        plant=plant,
        experiment=BasicBatchExperiment(max_programs=4, cost_per_unit=1.0),
        policy=EditOrderingPolicy(),
        regime=_regime(),
        benchmark=DummyBenchmark(),
        budget=Budget(total=1.0),
        metrics=[],
        rng=np.random.default_rng(0),
        loggers=[],
    )
    assert len(out.history.observations) == 1
    assert len(plant.observed_thetas) == 1
    assert np.allclose(plant.observed_thetas[0], np.array([1.0, 2.0], dtype=float))
