from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from beginoi.benchmarks.spec import Heatmap, GridSpec, make_grid
from beginoi.plants.paulsson_machine_plant import (
    PaulssonMachinePlant,
    PaulssonRealMismatch,
    PaulssonMachineSimulator,
    _ring_bandpass,
)
from beginoi.benchmarks.mismatch.structured_residual import StructuredResidualMismatch


@dataclass(frozen=True)
class TargetBandpass:
    """Analytic bandpass target g(u) in [0,1]."""

    center: tuple[float, float] = (0.5, 0.5)
    r_mid: float = 0.45
    thickness: float = 0.18
    width: float = 0.03

    def g_batch(self, U: np.ndarray) -> np.ndarray:
        return _ring_bandpass(
            U,
            center=np.array(self.center, dtype=float),
            r_mid=float(self.r_mid),
            thickness=float(self.thickness),
            width=float(self.width),
        )

    def g(self, u: np.ndarray) -> float:
        return float(self.g_batch(np.asarray(u, dtype=float))[0])


@dataclass(frozen=True)
class PaulssonMachineBenchmark:
    """Synthetic benchmark for `regime=single_paulsson` (steady-state I/O heatmap)."""

    name: str = "paulsson_machine"
    grid_n1: int = 41
    grid_n2: int = 41
    low: float = 0.0
    high: float = 1.0
    seed: int = 0
    target: TargetBandpass = TargetBandpass()
    simulator_width: float = 0.03
    theta0: tuple[float, float, float, float] = (0.35, 0.35, 0.35, 0.25)
    theta_low: tuple[float, float, float, float] = (0.1, 0.1, 0.05, 0.05)
    theta_high: tuple[float, float, float, float] = (0.9, 0.9, 0.85, 0.6)
    # True mismatch params (defaults emphasize a learnable gap via input warp).
    a_true: float = 1.0
    b_true: float = 0.0
    s_true: tuple[float, float] = (1.25, 0.75)
    t_true: tuple[float, float] = (0.12, -0.08)
    residual_seed: int = 0
    residual_num_features: int = 0
    residual_lengthscale: float = 0.25
    residual_weight_scale: float = 0.02
    noise_sigma: float = 0.02
    heteroscedastic: bool = False
    hetero_scale: float = 0.0

    @property
    def grid(self) -> GridSpec:
        return make_grid(n1=self.grid_n1, n2=self.grid_n2, low=self.low, high=self.high)

    def _make_simulator(self) -> PaulssonMachineSimulator:
        return PaulssonMachineSimulator(
            theta_low=np.array(self.theta_low, dtype=float),
            theta_high=np.array(self.theta_high, dtype=float),
            width=float(self.simulator_width),
        )

    def target_g_batch(self, U: np.ndarray) -> np.ndarray:
        return self.target.g_batch(U)

    def simulator_y_batch(self, U: np.ndarray, theta: np.ndarray) -> np.ndarray:
        sim = self._make_simulator()
        return sim.y_batch(U, theta)

    def make_plant(self, *, regime: Any, seed: int) -> PaulssonMachinePlant:
        del regime
        rng = np.random.default_rng(int(seed if seed is not None else self.seed))
        residual = None
        if int(self.residual_num_features) > 0:
            residual = StructuredResidualMismatch.random(
                seed=int(self.residual_seed),
                num_features=int(self.residual_num_features),
                lengthscale=float(self.residual_lengthscale),
                weight_scale=float(self.residual_weight_scale),
            )
        mismatch = PaulssonRealMismatch(
            a_true=float(self.a_true),
            b_true=float(self.b_true),
            s_true=np.array(self.s_true, dtype=float),
            t_true=np.array(self.t_true, dtype=float),
            residual=residual,
            noise_sigma=float(self.noise_sigma),
            heteroscedastic=bool(self.heteroscedastic),
            hetero_scale=float(self.hetero_scale),
        )
        sim = self._make_simulator()
        theta0 = np.array(self.theta0, dtype=float)
        # Optional tiny random jitter to avoid perfectly symmetric starts.
        theta0 = theta0 + rng.normal(0.0, 0.01, size=(4,))
        return PaulssonMachinePlant(simulator=sim, mismatch=mismatch, theta0=theta0)

    def oracle_heatmap(self, theta: Any, *, seed: int) -> Heatmap:
        rng = np.random.default_rng(int(seed))
        sim = self._make_simulator()
        # Oracle heatmap is noiseless by default for evaluation/visualization.
        mismatch = PaulssonRealMismatch(
            a_true=float(self.a_true),
            b_true=float(self.b_true),
            s_true=np.array(self.s_true, dtype=float),
            t_true=np.array(self.t_true, dtype=float),
            residual=None,
            noise_sigma=0.0,
        )
        th_raw = getattr(theta, "theta", theta)
        th = np.asarray(th_raw, dtype=float)
        y = np.zeros((len(self.grid.x1), len(self.grid.x2)), dtype=float)
        for i, x1 in enumerate(self.grid.x1):
            for j, x2 in enumerate(self.grid.x2):
                u = np.array([x1, x2], dtype=float)
                val, _ = mismatch.apply(simulator=sim, u=u, theta=th, rng=rng)
                y[i, j] = float(val)
        return Heatmap(grid=self.grid, y=y)

    def simulator_heatmap(self, plant: Any, theta: Any) -> Heatmap:
        # Uses simulator-only heatmap via the plant.
        return plant.evaluate_heatmap(theta, self.grid)
