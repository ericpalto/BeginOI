from __future__ import annotations

from typing import Callable
from dataclasses import field, dataclass

import numpy as np

from .mismatch import PhiHat, _warp, _rbf_features

SimFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # y_sim_batch(U, theta)->(N,)
TargetFn = Callable[[np.ndarray], np.ndarray]  # g_batch(U)->(N,)


@dataclass(frozen=True)
class EditConfig:
    """Configuration for SPARC trust-region edit proposals."""

    trust_radius: float = 0.05
    edit_penalty: float = 1e-2
    theta_low: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.05, 0.05], dtype=float)
    )
    theta_high: np.ndarray = field(
        default_factory=lambda: np.array([0.9, 0.9, 0.85, 0.6], dtype=float)
    )
    n_candidates: int = 256


def y_hat_batch(
    U: np.ndarray, *, theta: np.ndarray, phi: PhiHat, y_sim_batch: SimFn
) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    theta = np.asarray(theta, dtype=float)
    Uw = _warp(U, s=phi.s, t=phi.t)
    y = phi.a * np.asarray(y_sim_batch(Uw, theta), dtype=float) + float(phi.b)
    if phi.centers is not None and phi.c is not None and len(phi.c) > 0:
        y = y + (
            _rbf_features(U, centers=phi.centers, lengthscale=float(phi.lengthscale))
            @ phi.c
        )
    return np.asarray(y, dtype=float)


def propose_theta(
    *,
    theta_k: np.ndarray,
    phi: PhiHat,
    cfg: EditConfig,
    rng: np.random.Generator,
    U_mc: np.ndarray,
    y_sim_batch: SimFn,
    g_batch: TargetFn,
) -> tuple[np.ndarray, dict[str, float]]:
    theta_k = np.asarray(theta_k, dtype=float)
    U_mc = np.asarray(U_mc, dtype=float)
    low = np.asarray(cfg.theta_low, dtype=float)
    high = np.asarray(cfg.theta_high, dtype=float)

    def _obj(theta: np.ndarray) -> float:
        theta = np.clip(theta, low, high)
        pred = y_hat_batch(U_mc, theta=theta, phi=phi, y_sim_batch=y_sim_batch)
        g = np.asarray(g_batch(U_mc), dtype=float)
        mse = float(np.mean((pred - g) ** 2))
        pen = float(cfg.edit_penalty) * float(np.sum((theta - theta_k) ** 2))
        return float(mse + pen)

    best_theta = np.clip(theta_k, low, high)
    best_val = _obj(best_theta)

    d = len(theta_k)
    for _ in range(int(cfg.n_candidates)):
        z = rng.normal(0.0, 1.0, size=(d,))
        nz = float(np.linalg.norm(z) + 1e-12)
        # Uniform in L2 ball radius.
        rad = float(cfg.trust_radius) * float(rng.uniform(0.0, 1.0) ** (1.0 / d))
        cand = theta_k + (rad / nz) * z
        cand = np.clip(cand, low, high)
        val = _obj(cand)
        if val < best_val:
            best_val = val
            best_theta = cand

    return best_theta, {"pred_obj": float(best_val)}
