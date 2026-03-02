from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Heatmap, Program, GridSpec
from beginoi.benchmarks.spec import make_grid
from beginoi.plants.ode_plant import ODESpec, GenericODESimToRealPlant
from beginoi.benchmarks.mismatch.noise import ObservationNoise
from beginoi.benchmarks.mismatch.param_drift import ParamDriftMismatch
from beginoi.benchmarks.mismatch.structured_residual import RandomStructuredResidual


def _brusselator_rhs(
    t: float, state: np.ndarray, x: np.ndarray, params: Any
) -> np.ndarray:
    del t
    a = float(params["a"])
    b = float(params["b"])
    u = float(x[0])
    v = float(x[1])
    # Two-state Brusselator-like system with inputs injecting into reaction rates.
    x1, x2 = float(state[0]), float(state[1])
    dx1 = a + u - (b + 1.0 + v) * x1 + x1 * x1 * x2
    dx2 = (b + v) * x1 - x1 * x1 * x2
    return np.array([dx1, dx2], dtype=float)


@dataclass(frozen=True)
class BrusselatorBenchmark:
    """Generic ODE benchmark: Brusselator-like steady-state output."""

    name: str = "ode_brusselator"
    grid_n1: int = 21
    grid_n2: int = 21
    low: float = 0.0
    high: float = 1.0
    t_final: float = 10.0
    dt: float = 0.02
    noise: ObservationNoise | None = None
    structured_residual: RandomStructuredResidual | None = None
    param_drift: ParamDriftMismatch | None = None

    @property
    def grid(self) -> GridSpec:
        return make_grid(n1=self.grid_n1, n2=self.grid_n2, low=self.low, high=self.high)

    def make_plant(self, *, regime: Any, seed: int) -> Any:
        del regime
        rng = np.random.default_rng(int(seed))
        sim_params = {"a": 1.0, "b": 2.0}
        real_params = {
            "a": float(sim_params["a"] * rng.lognormal(0.0, 0.05)),
            "b": float(sim_params["b"] * rng.lognormal(0.0, 0.05)),
        }
        ode = ODESpec(
            state_size=2, rhs=_brusselator_rhs, t_final=self.t_final, dt=self.dt
        )
        residual = (
            None
            if self.structured_residual is None
            else self.structured_residual.build()
        )
        return GenericODESimToRealPlant(
            ode=ode,
            sim_params=sim_params,
            real_params=real_params,
            output_index=0,
            y0=np.array([1.0, 1.0], dtype=float),
            noise=self.noise,
            structured_residual=residual,
            param_drift=self.param_drift,
        )

    def oracle_heatmap(self, theta: Any, *, seed: int) -> Heatmap:
        del seed
        plant = self.make_plant(regime=None, seed=0)
        # Override real params from theta (keeps deterministic oracle evaluation).
        plant_theta = theta
        y = np.zeros((len(self.grid.x1), len(self.grid.x2)), dtype=float)
        for i, x1 in enumerate(self.grid.x1):
            for j, x2 in enumerate(self.grid.x2):
                program = Program(kind="constant", u=np.array([x1, x2], dtype=float))
                y[i, j] = float(
                    plant.observe(program, plant_theta, rng=np.random.default_rng(0))
                )
        return Heatmap(grid=self.grid, y=y)

    def simulator_heatmap(self, plant: Any, theta: Any) -> Heatmap:
        return plant.evaluate_heatmap(theta, self.grid)
