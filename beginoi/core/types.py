from __future__ import annotations

from typing import Any, Literal, Mapping
from dataclasses import field, dataclass

import numpy as np

ArrayLike = Any

ProgramKind = Literal["constant", "timeseries"]
InterventionKind = Literal["theta_edit", "exposure_schedule"]


@dataclass(frozen=True)
class Program:
    """An input program, either constant (2,) or time-series (T,2)."""

    kind: ProgramKind
    u: ArrayLike
    t: ArrayLike | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_constant_inputs(self) -> np.ndarray:
        """Return a (2,) array for plants that only support constant inputs."""
        if self.kind == "constant":
            arr = np.asarray(self.u, dtype=float)
            if arr.shape != (2,):
                raise ValueError(
                    f"Expected constant u with shape (2,), got {arr.shape}."
                )
            return arr

        u = np.asarray(self.u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError(f"Expected timeseries u with shape (T,2), got {u.shape}.")
        return np.mean(u, axis=0)


@dataclass(frozen=True)
class ProgramBatch:
    """A batch of programs executed within one budget unit."""

    programs: list[Program]


@dataclass(frozen=True)
class Intervention:
    """A control/edit applied between or during budget units."""

    kind: InterventionKind
    payload: Any


@dataclass(frozen=True)
class ControlAction:
    """What a policy selects for the next budget unit."""

    batch: ProgramBatch
    intervention: Intervention | None = None


@dataclass(frozen=True)
class Observation:
    """A single scalar output observation."""

    program_id: str
    inputs_summary: Mapping[str, Any]
    y: float
    t_obs: float
    theta: ArrayLike | None = None
    round_id: int = 0
    schema_version: int = 2
    noise_meta: Mapping[str, Any] = field(default_factory=dict)
    unit_id: int = 0
    replicate_id: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """Result from executing one budget unit."""

    consumed_cost: float
    observations: list[Observation]
    final_theta: Any


@dataclass
class History:
    """Cumulative run history."""

    observations: list[Observation] = field(default_factory=list)
    actions: list[ControlAction] = field(default_factory=list)
    theta_snapshots: list[Any] = field(default_factory=list)
    budget_spent: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Float-valued experiment budget."""

    total: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return float(self.total - self.spent)

    def can_afford(self, cost: float) -> bool:
        return self.remaining + 1e-12 >= float(cost)


@dataclass(frozen=True)
class GridSpec:
    """A 2D grid over (x1, x2) for heatmap evaluation."""

    x1: np.ndarray
    x2: np.ndarray

    def __post_init__(self) -> None:
        x1 = np.asarray(self.x1, dtype=float)
        x2 = np.asarray(self.x2, dtype=float)
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("GridSpec x1/x2 must be 1D arrays.")
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "x2", x2)


@dataclass(frozen=True)
class Heatmap:
    """Heatmap values evaluated on a GridSpec."""

    grid: GridSpec
    y: np.ndarray  # shape (len(x1), len(x2))

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float)
        expected = (len(self.grid.x1), len(self.grid.x2))
        if y.shape != expected:
            raise ValueError(f"Expected heatmap shape {expected}, got {y.shape}.")
        object.__setattr__(self, "y", y)
