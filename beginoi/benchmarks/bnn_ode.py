from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Heatmap, Program, GridSpec
from beginoi.benchmarks.spec import make_grid
from beginoi.plants.pybnn_plant import (
    MoormanXORSurface,
    PyBNNSimToRealPlant,
    MoormanXORFunctionPlant,
)
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


@dataclass(frozen=True)
class BNNXORFunctionBenchmark(BNNPerceptronBenchmark):
    """Moorman XOR benchmark with Paulsson-style editable 4D theta."""

    name: str = "bnn_xor_function"
    model: str = "two_node_xor"
    t_final: float = 5.0
    dt: float = 0.005
    surface_n: int = 41
    theta0: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.0)
    theta_low: tuple[float, float, float, float] = (0.7, -0.2, 0.7, -0.2)
    theta_high: tuple[float, float, float, float] = (1.3, 0.2, 1.3, 0.2)
    target_theta: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.0)
    a_true: float = 1.0
    b_true: float = 0.0
    s_true: tuple[float, float] = (1.0, 1.0)
    t_true: tuple[float, float] = (0.0, 0.0)

    @property
    def surface(self) -> MoormanXORSurface:
        return MoormanXORSurface.build(
            grid_n=int(self.surface_n),
            t_final=float(self.t_final),
            dt=float(self.dt),
        )

    def simulator_y_batch(self, U: np.ndarray, theta: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[None, :]
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError(f"Expected U shape (N,2), got {U.shape}.")
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if theta_vec.shape != (4,):
            raise ValueError(f"Expected theta shape (4,), got {theta_vec.shape}.")
        low = np.asarray(self.theta_low, dtype=float)
        high = np.asarray(self.theta_high, dtype=float)
        theta_vec = np.clip(theta_vec, low, high)

        Uw = np.empty_like(U, dtype=float)
        Uw[:, 0] = theta_vec[0] * U[:, 0] + theta_vec[1]
        Uw[:, 1] = theta_vec[2] * U[:, 1] + theta_vec[3]
        return self.surface.evaluate(np.clip(Uw, 0.0, 1.0))

    def target_g_batch(self, U: np.ndarray) -> np.ndarray:
        return self.simulator_y_batch(U, np.asarray(self.target_theta, dtype=float))

    def make_plant(self, *, regime: Any, seed: int) -> MoormanXORFunctionPlant:
        del regime, seed
        residual = (
            None
            if self.structured_residual is None
            else self.structured_residual.build()
        )
        return MoormanXORFunctionPlant(
            backend=self.backend,
            t_final=self.t_final,
            dt=self.dt,
            surface=self.surface,
            theta0=np.asarray(self.theta0, dtype=float),
            theta_low=np.asarray(self.theta_low, dtype=float),
            theta_high=np.asarray(self.theta_high, dtype=float),
            real_a=self.a_true,
            real_b=self.b_true,
            real_s=np.asarray(self.s_true, dtype=float),
            real_t=np.asarray(self.t_true, dtype=float),
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
        plant = MoormanXORFunctionPlant(
            backend=self.backend,
            t_final=self.t_final,
            dt=self.dt,
            surface=self.surface,
            theta0=np.asarray(self.theta0, dtype=float),
            theta_low=np.asarray(self.theta_low, dtype=float),
            theta_high=np.asarray(self.theta_high, dtype=float),
            real_a=self.a_true,
            real_b=self.b_true,
            real_s=np.asarray(self.s_true, dtype=float),
            real_t=np.asarray(self.t_true, dtype=float),
            noise=ObservationNoise(0.0),
            structured_residual=residual,
            param_drift=self.param_drift,
        )
        y = np.zeros((len(self.grid.x1), len(self.grid.x2)), dtype=float)
        theta_vec = np.asarray(getattr(theta, "theta", theta), dtype=float)
        for i, x1 in enumerate(self.grid.x1):
            for j, x2 in enumerate(self.grid.x2):
                y[i, j] = float(
                    plant.real_mean_batch(
                        np.array([[x1, x2]], dtype=float),
                        theta=theta_vec,
                    )[0]
                )
        return Heatmap(grid=self.grid, y=y)
