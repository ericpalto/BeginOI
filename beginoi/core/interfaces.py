from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .types import (
    Heatmap,
    History,
    Program,
    GridSpec,
    Observation,
    ProgramBatch,
    ControlAction,
    ExperimentResult,
)

ThetaState = Any


@runtime_checkable
class Plant(Protocol):
    """A plant provides both a simulator and an oracle (real) interface."""

    def reset(self, seed: int) -> ThetaState:
        ...

    def simulate(self, program: Program, theta: ThetaState) -> float:
        ...

    def observe(self, program: Program, theta: ThetaState, rng: Any) -> float:
        ...

    def apply_intervention(
        self,
        theta: ThetaState,
        intervention: Any,
        dt: float | None,
    ) -> ThetaState:
        ...

    def evaluate_heatmap(self, theta: ThetaState, grid_spec: GridSpec) -> Heatmap:
        ...


FeedbackHandler = Callable[[Observation], Any | None]


@runtime_checkable
class Experiment(Protocol):
    """Defines one budget unit: cost model + execution producing observations."""

    def max_programs_per_unit(self) -> int:
        ...

    def cost_of(self, batch: ProgramBatch) -> float:
        ...

    def run_budget_unit(
        self,
        plant: Plant,
        theta: ThetaState,
        action: ControlAction,
        *,
        unit_id: int,
        feedback_handler: FeedbackHandler | None = None,
        rng: Any = None,
    ) -> ExperimentResult:
        ...


@runtime_checkable
class Policy(Protocol):
    """Selects the next action given the run history."""

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        ...

    def act(self, history: History, *, budget_remaining: float) -> ControlAction:
        ...

    def update(self, history: History, new_result: ExperimentResult) -> None:
        ...

    # Optional: policies may implement `on_observation(obs)` for feedback regimes.


MetricFn = Callable[[History, Any], dict[str, Any]]
