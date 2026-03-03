from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np

from beginoi.core.types import Heatmap, Program, GridSpec, Intervention
from beginoi.benchmarks.mismatch.structured_residual import StructuredResidualMismatch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _ring_bandpass(
    U: np.ndarray,
    *,
    center: np.ndarray,
    r_mid: float,
    thickness: float,
    width: float,
) -> np.ndarray:
    """Smooth ring-shaped bandpass in [0,1] over U in [0,1]^2.

    Uses difference-of-sigmoids in radius:
      sigmoid((r - r_in)/w) - sigmoid((r - r_out)/w)
    """
    U = np.asarray(U, dtype=float)
    center = np.asarray(center, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError(f"Expected U shape (N,2), got {U.shape}.")
    if center.shape != (2,):
        raise ValueError(f"Expected center shape (2,), got {center.shape}.")

    r = np.sqrt(np.sum((U - center[None, :]) ** 2, axis=1))
    thickness = float(max(thickness, 1e-6))
    r_in = max(0.0, float(r_mid) - 0.5 * thickness)
    r_out = max(r_in + 1e-6, float(r_mid) + 0.5 * thickness)
    w = float(max(width, 1e-6))
    band = _sigmoid((r - r_in) / w) - _sigmoid((r - r_out) / w)
    return np.clip(band, 0.0, 1.0)


def _warp(U: np.ndarray, *, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    if s.shape != (2,) or t.shape != (2,):
        raise ValueError("Expected s,t with shape (2,).")
    return np.clip(U * s[None, :] + t[None, :], 0.0, 1.0)


@dataclass(frozen=True)
class PaulssonMachineSimulator:
    """Analytic simulator family y_sim(u, theta) with theta in R^4.

    theta = [center_x, center_y, r_mid, thickness] with clipping to bounds.
    """

    theta_low: np.ndarray
    theta_high: np.ndarray
    width: float = 0.03

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (4,):
            raise ValueError(f"Expected theta shape (4,), got {theta.shape}.")
        return np.clip(theta, self.theta_low, self.theta_high)

    def y_batch(self, U: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta = self.project_theta(theta)
        center = theta[:2]
        r_mid = float(theta[2])
        thickness = float(theta[3])
        return _ring_bandpass(
            U, center=center, r_mid=r_mid, thickness=thickness, width=self.width
        )

    def y(self, u: np.ndarray, theta: np.ndarray) -> float:
        return float(self.y_batch(np.asarray(u, dtype=float), theta)[0])


@dataclass
class PaulssonThetaState:
    """Editable plant state holding the current simulator parameter vector theta."""

    theta: np.ndarray
    time: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaulssonRealMismatch:
    """Defines the synthetic sim-to-real mismatch for the PaulssonMachinePlant."""

    a_true: float = 1.0
    b_true: float = 0.0
    s_true: np.ndarray = field(default_factory=lambda: np.ones((2,), dtype=float))
    t_true: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=float))
    residual: StructuredResidualMismatch | None = None
    noise_sigma: float = 0.0
    heteroscedastic: bool = False
    hetero_scale: float = 0.0

    def mean_batch(
        self,
        *,
        simulator: PaulssonMachineSimulator,
        U: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[None, :]
        Uw = _warp(U, s=self.s_true, t=self.t_true)
        y_sim = np.asarray(simulator.y_batch(Uw, theta), dtype=float)
        y = self.a_true * y_sim + self.b_true
        if self.residual is not None:
            r = np.array([self.residual.residual(u) for u in U], dtype=float)
            y = y + r
        return np.asarray(y, dtype=float)

    def mean(
        self,
        *,
        simulator: PaulssonMachineSimulator,
        u: np.ndarray,
        theta: np.ndarray,
    ) -> float:
        return float(
            self.mean_batch(simulator=simulator, U=np.asarray(u), theta=theta)[0]
        )

    def apply(
        self,
        *,
        simulator: PaulssonMachineSimulator,
        u: np.ndarray,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[float, dict[str, Any]]:
        u = np.asarray(u, dtype=float)
        u_w = _warp(u, s=self.s_true, t=self.t_true)[0]
        y_sim = float(simulator.y(u_w, theta))
        y = float(self.a_true * y_sim + self.b_true)
        meta: dict[str, Any] = {"y_sim_warped": y_sim}
        if self.residual is not None:
            y, rmeta = self.residual.apply(y, x=u)
            meta.update(rmeta)
        sigma = float(self.noise_sigma)
        if self.heteroscedastic:
            sigma = float(self.noise_sigma * (1.0 + float(self.hetero_scale) * abs(y)))
        if sigma > 0:
            eps = float(rng.normal(0.0, sigma))
            y = float(y + eps)
            meta.update({"sigma": sigma, "eps": eps})
        else:
            meta.update({"sigma": sigma})
        return float(y), meta


class PaulssonMachinePlant:
    """Synthetic sim-to-real plant with an analytic simulator + mock real oracle."""

    def __init__(
        self,
        *,
        simulator: PaulssonMachineSimulator,
        mismatch: PaulssonRealMismatch,
        theta0: np.ndarray,
    ) -> None:
        self.simulator = simulator
        self.mismatch = mismatch
        self._theta0 = np.asarray(theta0, dtype=float)
        if self._theta0.shape != (4,):
            raise ValueError("PaulssonMachinePlant theta0 must have shape (4,).")

    def reset(self, seed: int) -> PaulssonThetaState:
        del seed
        return PaulssonThetaState(theta=self.simulator.project_theta(self._theta0))

    def simulate(self, program: Program, theta: PaulssonThetaState) -> float:
        u = program.as_constant_inputs()
        return float(self.simulator.y(u, theta.theta))

    def observe(self, program: Program, theta: PaulssonThetaState, rng: Any) -> float:
        if rng is None:
            rng = np.random.default_rng(0)
        u = program.as_constant_inputs()
        y, noise_meta = self.mismatch.apply(
            simulator=self.simulator, u=u, theta=theta.theta, rng=rng
        )
        program.meta.setdefault("noise_meta", dict(noise_meta))
        return float(y)

    def real_mean_batch(self, U: np.ndarray, *, theta: np.ndarray) -> np.ndarray:
        """Noise-free oracle mean over a batch of inputs (for plotting/debug)."""
        return self.mismatch.mean_batch(
            simulator=self.simulator,
            U=np.asarray(U, dtype=float),
            theta=np.asarray(theta, dtype=float),
        )

    def apply_intervention(
        self,
        theta: PaulssonThetaState,
        intervention: Intervention,
        dt: float | None,
    ) -> PaulssonThetaState:
        del dt
        if intervention.kind != "theta_edit":
            raise ValueError(f"Unknown intervention kind: {intervention.kind!r}")
        payload = intervention.payload or {}
        th = np.asarray(theta.theta, dtype=float)
        if "set" in payload and payload["set"] is not None:
            s = dict(payload["set"])
            if "theta" in s:
                th = np.asarray(s["theta"], dtype=float)
        if "delta" in payload and payload["delta"] is not None:
            d = dict(payload["delta"])
            if "theta" in d:
                th = th + np.asarray(d["theta"], dtype=float)
        th = self.simulator.project_theta(th)
        return PaulssonThetaState(
            theta=th, time=float(theta.time), meta=dict(theta.meta)
        )

    def evaluate_heatmap(
        self, theta: PaulssonThetaState, grid_spec: GridSpec
    ) -> Heatmap:
        y = np.zeros((len(grid_spec.x1), len(grid_spec.x2)), dtype=float)
        for i, x1 in enumerate(grid_spec.x1):
            for j, x2 in enumerate(grid_spec.x2):
                u = np.array([x1, x2], dtype=float)
                y[i, j] = self.simulator.y(u, theta.theta)
        return Heatmap(grid=grid_spec, y=y)
