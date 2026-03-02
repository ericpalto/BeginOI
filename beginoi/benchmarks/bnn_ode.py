from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Heatmap, Program, GridSpec
from beginoi.benchmarks.spec import make_grid
from beginoi.plants.pybnn_plant import PyBNNSimToRealPlant
from beginoi.benchmarks.mismatch.noise import ObservationNoise
from beginoi.benchmarks.mismatch.param_drift import ParamDriftMismatch
from beginoi.benchmarks.mismatch.structured_residual import RandomStructuredResidual


@dataclass(frozen=True)
class BNNPerceptronBenchmark:
    """PyBNN-backed benchmark using the Moorman perceptron circuit."""

    name: str = "bnn_perceptron"
    grid_n1: int = 21
    grid_n2: int = 21
    low: float = 0.0
    high: float = 1.0
    formulation: str = "moorman"
    model: str = "perceptron"
    backend: str = "numpy"
    t_final: float = 5.0
    dt: float = 0.02
    noise: ObservationNoise = ObservationNoise(0.0)
    structured_residual: RandomStructuredResidual | None = None
    param_drift: ParamDriftMismatch | None = None

    @property
    def grid(self) -> GridSpec:
        return make_grid(n1=self.grid_n1, n2=self.grid_n2, low=self.low, high=self.high)

    def make_plant(self, *, regime: Any, seed: int) -> PyBNNSimToRealPlant:
        del regime, seed
        residual = (
            None
            if self.structured_residual is None
            else self.structured_residual.build()
        )
        return PyBNNSimToRealPlant(
            formulation=self.formulation,
            model=self.model,
            backend=self.backend,
            t_final=self.t_final,
            dt=self.dt,
            noise=self.noise,
            structured_residual=residual,
            param_drift=self.param_drift,
        )

    def oracle_heatmap(self, theta: Any, *, seed: int) -> Heatmap:
        del seed
        residual = (
            None
            if self.structured_residual is None
            else self.structured_residual.build()
        )
        plant = PyBNNSimToRealPlant(
            formulation=self.formulation,
            model=self.model,
            backend=self.backend,
            t_final=self.t_final,
            dt=self.dt,
            noise=ObservationNoise(0.0),
            structured_residual=residual,
            param_drift=self.param_drift,
        )
        y = np.zeros((len(self.grid.x1), len(self.grid.x2)), dtype=float)
        for i, x1 in enumerate(self.grid.x1):
            for j, x2 in enumerate(self.grid.x2):
                program = Program(kind="constant", u=np.array([x1, x2], dtype=float))
                y[i, j] = float(
                    plant.observe(program, theta, rng=np.random.default_rng(0))
                )
        return Heatmap(grid=self.grid, y=y)

    def simulator_heatmap(self, plant: Any, theta: Any) -> Heatmap:
        return plant.evaluate_heatmap(theta, self.grid)
