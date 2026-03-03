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
    return U * s[None, :] + t[None, :]


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
    smoothing: float = 0.0
    enforce_positive_a: bool = True
    a_min: float = 0.05
    a_max: float = 3.0
    min_design_points_for_warp: int = 40
    min_design_points_for_rbf: int = 120
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
    theta_k: np.ndarray | None = None,
    obs_design: list[ProbeObservation],
    y_sim_batch: SimFn,
    cfg: MismatchFitConfig,
    init_phi: PhiHat | None = None,
) -> tuple[PhiHat, dict[str, float | bool]]:
    if not obs_design:
        return PhiHat(), {"train_rmse": float("nan")}

    U = np.array([o.u for o in obs_design], dtype=float)
    y = np.array([o.y_mean for o in obs_design], dtype=float)
    y_var = np.array([o.y_var for o in obs_design], dtype=float)
    theta_ref = None
    if theta_k is not None:
        theta_ref = np.asarray(theta_k, dtype=float).reshape(-1)
    theta_rows: list[np.ndarray] = []
    missing_theta_count = 0
    for o in obs_design:
        if o.theta is None:
            if theta_ref is None:
                raise ValueError(
                    "ProbeObservation.theta is missing and theta_k fallback "
                    "was not provided."
                )
            theta_rows.append(np.asarray(theta_ref, dtype=float))
            missing_theta_count += 1
        else:
            theta_rows.append(np.asarray(o.theta, dtype=float).reshape(-1))
    Theta = np.vstack(theta_rows)

    def _sim_with_theta_obs(Uw: np.ndarray) -> np.ndarray:
        Uw = np.asarray(Uw, dtype=float)
        if Uw.ndim != 2 or Uw.shape[1] != 2:
            raise ValueError(f"Expected Uw shape (N,2), got {Uw.shape}.")
        if len(Uw) != len(Theta):
            raise ValueError(
                f"Mismatch between Uw rows ({len(Uw)}) and theta rows ({len(Theta)})."
            )
        if len(Uw) == 0:
            return np.zeros((0,), dtype=float)
        if float(np.max(np.abs(Theta - Theta[:1, :]))) < 1e-12:
            return np.asarray(y_sim_batch(Uw, Theta[0]), dtype=float)
        out = np.zeros((len(Uw),), dtype=float)
        for i, (u_row, th_row) in enumerate(zip(Uw, Theta)):
            out[i] = float(
                np.asarray(y_sim_batch(u_row[None, :], th_row), dtype=float)[0]
            )
        return out

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

    s_identity = np.ones((2,), dtype=float)
    t_identity = np.zeros((2,), dtype=float)
    if use_warp and init_phi is not None:
        s_init = np.asarray(init_phi.s, dtype=float).copy()
        t_init = np.asarray(init_phi.t, dtype=float).copy()
    else:
        s_init = s_identity.copy()
        t_init = t_identity.copy()
    s_init = np.clip(s_init, float(cfg.warp_s_min), float(cfg.warp_s_max))
    t_init = np.clip(t_init, float(cfg.warp_t_min), float(cfg.warp_t_max))

    def _linear_fit(
        sv: np.ndarray, tv: np.ndarray
    ) -> tuple[PhiHat, np.ndarray, float, bool]:
        Uw = _warp(U, s=sv, t=tv) if use_warp else U
        x = np.asarray(_sim_with_theta_obs(Uw), dtype=float)
        feats = None
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
        a_clamped = False
        if use_aff and bool(cfg.enforce_positive_a):
            a_lo = float(min(cfg.a_min, cfg.a_max))
            a_hi = float(max(cfg.a_min, cfg.a_max))
            a_new = float(np.clip(a, a_lo, a_hi))
            if abs(a_new - a) > 1e-12:
                a = a_new
                a_clamped = True
                cols_tail: list[np.ndarray] = [np.ones_like(x)]
                ridge_tail: list[float] = [float(cfg.ridge_ab)]
                if feats is not None:
                    cols_tail.append(feats)
                    ridge_tail.extend([float(cfg.ridge_c)] * feats.shape[1])
                X_tail = np.column_stack(cols_tail)
                y_adj = y - a * x
                beta_tail = _solve_ridge(
                    X_tail,
                    y_adj,
                    w=w,
                    ridge=np.array(ridge_tail, dtype=float),
                )
                b = float(beta_tail[0])
                if feats is not None:
                    c = np.asarray(beta_tail[1:], dtype=float)
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
        return phi, yhat, sse, a_clamped

    def _objective(sv: np.ndarray, tv: np.ndarray) -> float:
        phi, _, mse, _ = _linear_fit(sv, tv)
        pen = 0.0
        if use_warp:
            pen += float(cfg.warp_penalty) * float(
                np.sum((phi.s - 1.0) ** 2) + np.sum(phi.t**2)
            )
        return float(mse + pen)

    def _optimize_warp(s0: np.ndarray, t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s0, dtype=float).copy()
        t = np.asarray(t0, dtype=float).copy()
        s = np.clip(s, float(cfg.warp_s_min), float(cfg.warp_s_max))
        t = np.clip(t, float(cfg.warp_t_min), float(cfg.warp_t_max))
        for _ in range(int(cfg.n_alt)):
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
            s = s - step * np.array([grad[0], grad[2]], dtype=float)
            t = t - step * np.array([grad[1], grad[3]], dtype=float)
            s = np.clip(s, float(cfg.warp_s_min), float(cfg.warp_s_max))
            t = np.clip(t, float(cfg.warp_t_min), float(cfg.warp_t_max))
            if _objective(s, t) > base + 1e-9:
                s = 0.5 * (s + s_identity)
                t = 0.5 * t
        return s, t

    if use_warp:
        starts = [(s_identity, t_identity), (s_init, t_init)]
        best_s = s_identity
        best_t = t_identity
        best_obj = float("inf")
        for s0, t0 in starts:
            s_try, t_try = _optimize_warp(s0, t0)
            val = _objective(s_try, t_try)
            if val < best_obj:
                best_obj = val
                best_s = s_try
                best_t = t_try
        s, t = best_s, best_t
    else:
        s, t = s_identity, t_identity

    phi_final, yhat, _, a_clamped = _linear_fit(s, t)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    return phi_final, {
        "train_rmse": rmse,
        "a_clamped": bool(a_clamped),
        "theta_fallback_used": bool(missing_theta_count > 0),
    }
