from __future__ import annotations

from typing import Callable
from dataclasses import field, dataclass

import numpy as np

from .types import ProbeObservation

SimFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # y_sim_batch(U, theta)->(N,)


def _warp(U: np.ndarray, *, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    return np.clip(U * s[None, :] + t[None, :], 0.0, 1.0)


def _rbf_features(
    U: np.ndarray, *, centers: np.ndarray, lengthscale: float
) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    centers = np.asarray(centers, dtype=float)
    diff = U[:, None, :] - centers[None, :, :]
    r2 = np.sum(diff**2, axis=2)
    ell2 = float(lengthscale) ** 2 + 1e-12
    return np.exp(-0.5 * r2 / ell2)


@dataclass(frozen=True)
class MismatchFitConfig:
    """Configuration for Phase-0 mismatch fitting."""

    use_affine_y: bool = True
    use_input_warp: bool = True
    use_rbf_residual: bool = False
    n_rbf: int = 16
    rbf_lengthscale: float = 0.25
    rbf_seed: int = 0
    ridge_ab: float = 1e-6
    ridge_c: float = 1e-2
    warp_penalty: float = 1e-2
    warp_lr: float = 0.1
    warp_fd_eps: float = 1e-3
    n_alt: int = 6
    heteroscedastic_fit: bool = False
    weight_eps: float = 1e-6
    warp_s_min: float = 0.7
    warp_s_max: float = 1.3
    warp_t_min: float = -0.2
    warp_t_max: float = 0.2


@dataclass(frozen=True)
class PhiHat:
    """Fitted mismatch parameters (phi) used to correct simulator predictions."""

    a: float = 1.0
    b: float = 0.0
    s: np.ndarray = field(default_factory=lambda: np.ones((2,), dtype=float))
    t: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=float))
    centers: np.ndarray | None = None
    lengthscale: float = 0.25
    c: np.ndarray | None = None

    def residual_batch(self, U: np.ndarray) -> np.ndarray:
        if self.centers is None or self.c is None or len(self.c) == 0:
            return np.zeros((len(U),), dtype=float)
        feats = _rbf_features(
            U, centers=self.centers, lengthscale=float(self.lengthscale)
        )
        return feats @ np.asarray(self.c, dtype=float)


def _solve_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    w: np.ndarray,
    ridge: np.ndarray,
) -> np.ndarray:
    # Solve (X^T W X + diag(ridge)) beta = X^T W y
    w = np.asarray(w, dtype=float)
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(ridge)
    b = X.T @ (y * w)
    return np.linalg.solve(A, b)


def fit_mismatch(
    *,
    theta_k: np.ndarray,
    obs_design: list[ProbeObservation],
    y_sim_batch: SimFn,
    cfg: MismatchFitConfig,
) -> tuple[PhiHat, dict[str, float]]:
    if not obs_design:
        return PhiHat(), {"train_rmse": float("nan")}

    U = np.array([o.u for o in obs_design], dtype=float)
    y = np.array([o.y_mean for o in obs_design], dtype=float)
    y_var = np.array([o.y_var for o in obs_design], dtype=float)
    if cfg.heteroscedastic_fit:
        w = 1.0 / np.maximum(y_var + float(cfg.weight_eps), float(cfg.weight_eps))
        w = w / np.mean(w)
    else:
        w = np.ones_like(y, dtype=float)

    if cfg.use_rbf_residual and int(cfg.n_rbf) > 0:
        rng = np.random.default_rng(int(cfg.rbf_seed))
        centers = rng.uniform(0.0, 1.0, size=(int(cfg.n_rbf), 2))
        lengthscale = float(cfg.rbf_lengthscale)
    else:
        centers = None
        lengthscale = float(cfg.rbf_lengthscale)

    use_aff = bool(cfg.use_affine_y)
    use_warp = bool(cfg.use_input_warp)

    s = np.ones((2,), dtype=float)
    t = np.zeros((2,), dtype=float)
    if not use_warp:
        s = np.ones((2,), dtype=float)
        t = np.zeros((2,), dtype=float)

    def _linear_fit(sv: np.ndarray, tv: np.ndarray) -> tuple[PhiHat, np.ndarray, float]:
        Uw = _warp(U, s=sv, t=tv) if use_warp else U
        x = np.asarray(y_sim_batch(Uw, np.asarray(theta_k, dtype=float)), dtype=float)
        cols: list[np.ndarray] = []
        ridge_terms: list[float] = []
        if use_aff:
            cols.append(x)
            ridge_terms.append(float(cfg.ridge_ab))
        cols.append(np.ones_like(x))
        ridge_terms.append(float(cfg.ridge_ab))
        if centers is not None:
            feats = _rbf_features(U, centers=centers, lengthscale=lengthscale)
            cols.append(feats)
            ridge_terms.extend([float(cfg.ridge_c)] * feats.shape[1])
        X = np.column_stack(cols)
        ridge = np.array(ridge_terms, dtype=float)
        beta = _solve_ridge(X, y, w=w, ridge=ridge)
        idx = 0
        if use_aff:
            a = float(beta[idx])
            idx += 1
        else:
            a = 1.0
        b = float(beta[idx])
        idx += 1
        c = None
        if centers is not None:
            c = np.asarray(beta[idx:], dtype=float)
        phi = PhiHat(
            a=a,
            b=b,
            s=np.asarray(sv, dtype=float),
            t=np.asarray(tv, dtype=float),
            centers=None if centers is None else np.asarray(centers, dtype=float),
            lengthscale=float(lengthscale),
            c=c,
        )
        yhat = phi.a * x + phi.b
        if centers is not None and c is not None:
            yhat = yhat + (
                _rbf_features(U, centers=centers, lengthscale=lengthscale) @ c
            )
        sse = float(np.mean((yhat - y) ** 2))
        return phi, yhat, sse

    def _objective(sv: np.ndarray, tv: np.ndarray) -> float:
        phi, _, mse = _linear_fit(sv, tv)
        pen = 0.0
        if use_warp:
            pen += float(cfg.warp_penalty) * float(
                np.sum((phi.s - 1.0) ** 2) + np.sum(phi.t**2)
            )
        return float(mse + pen)

    for _ in range(int(cfg.n_alt)):
        if not use_warp:
            break
        base = _objective(s, t)
        grad = np.zeros((4,), dtype=float)
        eps = float(cfg.warp_fd_eps)
        for j in range(4):
            ds = np.zeros((2,), dtype=float)
            dt = np.zeros((2,), dtype=float)
            if j == 0:
                ds[0] = eps
            elif j == 1:
                dt[0] = eps
            elif j == 2:
                ds[1] = eps
            else:
                dt[1] = eps
            up = _objective(s + ds, t + dt)
            dn = _objective(s - ds, t - dt)
            grad[j] = (up - dn) / (2.0 * eps)
        step = float(cfg.warp_lr)
        # Map grad back to (s,t)
        s = s - step * np.array([grad[0], grad[2]], dtype=float)
        t = t - step * np.array([grad[1], grad[3]], dtype=float)
        s = np.clip(s, float(cfg.warp_s_min), float(cfg.warp_s_max))
        t = np.clip(t, float(cfg.warp_t_min), float(cfg.warp_t_max))
        # If a step increases objective badly, damp it.
        if _objective(s, t) > base + 1e-9:
            s = 0.5 * (s + np.ones((2,), dtype=float))
            t = 0.5 * t

    phi_final, yhat, _ = _linear_fit(s, t)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    return phi_final, {"train_rmse": rmse}
