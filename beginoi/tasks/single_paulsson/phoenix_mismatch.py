from __future__ import annotations

from typing import Literal, Callable
from dataclasses import field, dataclass

import numpy as np

from .types import ProbeObservation

SimFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # y_sim_batch(U, theta)->(N,)


@dataclass(frozen=True)
class PhoenixMismatchConfig:
    """Configuration for PHOENIX mismatch fitting."""

    var_epsilon: float = 1e-6
    loss: Literal["wls", "huber_wls"] = "wls"
    huber_delta: float = 1.5
    lambda_s: float = 1e-2
    lambda_t: float = 1e-2
    lambda_ab: float = 1e-2
    warp_s_min: float = 0.7
    warp_s_max: float = 1.3
    warp_t_min: float = -0.2
    warp_t_max: float = 0.2
    a_min: float = 1e-3
    n_alt: int = 8
    n_warp_steps: int = 2
    warp_lr: float = 0.08
    warp_fd_eps: float = 1e-3
    n_ab_steps: int = 25
    ab_lr: float = 0.05


@dataclass(frozen=True)
class PhoenixPhi:
    """Fitted PHOENIX mismatch parameters."""

    alpha: float = 0.0
    b: float = 0.0
    s: np.ndarray = field(default_factory=lambda: np.ones((2,), dtype=float))
    t: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=float))

    @property
    def a(self) -> float:
        return float(np.exp(float(self.alpha)))


def warp_clip(U: np.ndarray, *, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    Uw = U * np.asarray(s, dtype=float)[None, :] + np.asarray(t, dtype=float)[None, :]
    return np.clip(Uw, 0.0, 1.0)


def y_hat_batch(
    U: np.ndarray,
    *,
    theta: np.ndarray,
    phi: PhoenixPhi,
    y_sim_batch: SimFn,
) -> np.ndarray:
    Uw = warp_clip(U, s=phi.s, t=phi.t)
    y_sim = np.asarray(y_sim_batch(Uw, np.asarray(theta, dtype=float)), dtype=float)
    return float(phi.a) * y_sim + float(phi.b)


def _sim_with_obs_theta(
    Uw: np.ndarray,
    *,
    Theta: np.ndarray,
    y_sim_batch: SimFn,
) -> np.ndarray:
    Uw = np.asarray(Uw, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    if len(Uw) == 0:
        return np.zeros((0,), dtype=float)
    if len(Uw) != len(Theta):
        raise ValueError(
            f"Mismatch between Uw rows ({len(Uw)}) and theta rows ({len(Theta)})."
        )
    if float(np.max(np.abs(Theta - Theta[:1, :]))) < 1e-12:
        return np.asarray(y_sim_batch(Uw, Theta[0]), dtype=float)
    out = np.zeros((len(Uw),), dtype=float)
    for i, (u_row, theta_row) in enumerate(zip(Uw, Theta)):
        out[i] = float(
            np.asarray(y_sim_batch(u_row[None, :], theta_row), dtype=float)[0]
        )
    return out


def _huber_scaled_loss_and_grad(
    r_scaled: np.ndarray, *, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    abs_r = np.abs(r_scaled)
    quad = abs_r <= float(delta)
    loss = np.where(
        quad,
        0.5 * (r_scaled**2),
        float(delta) * (abs_r - 0.5 * float(delta)),
    )
    grad = np.where(quad, r_scaled, float(delta) * np.sign(r_scaled))
    return loss, grad


def _fit_ab_for_fixed_warp(
    *,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    cfg: PhoenixMismatchConfig,
    alpha_init: float,
    b_init: float,
) -> tuple[float, float, float]:
    # Weighted linear solve for initialization, then refine alpha/b with fixed steps.
    sw = np.sqrt(w)
    X = np.column_stack([x, np.ones_like(x)])
    XtWX = (X * sw[:, None]).T @ (X * sw[:, None])
    ridge = np.diag(
        [
            float(cfg.lambda_ab),
            float(cfg.lambda_ab),
        ]
    )
    XtWy = (X * sw[:, None]).T @ (y * sw)
    try:
        beta = np.linalg.solve(XtWX + ridge, XtWy)
    except np.linalg.LinAlgError:
        beta = np.array([1.0, 0.0], dtype=float)
    a_init = float(max(float(beta[0]), float(cfg.a_min)))
    alpha = float(np.log(a_init))
    b = float(beta[1])
    if np.isfinite(alpha_init):
        alpha = 0.5 * alpha + 0.5 * float(alpha_init)
    if np.isfinite(b_init):
        b = 0.5 * b + 0.5 * float(b_init)

    def _objective(alpha_v: float, b_v: float) -> float:
        a_v = float(np.exp(alpha_v))
        pred = a_v * x + float(b_v)
        resid = pred - y
        if str(cfg.loss) == "huber_wls":
            r_scaled = sw * resid
            loss, _ = _huber_scaled_loss_and_grad(
                r_scaled, delta=float(cfg.huber_delta)
            )
            data = float(np.sum(loss))
        else:
            data = 0.5 * float(np.sum(w * (resid**2)))
        reg = float(cfg.lambda_ab) * float(alpha_v**2 + float(b_v) ** 2)
        return float(data + reg)

    best_obj = _objective(alpha, b)
    for _ in range(int(cfg.n_ab_steps)):
        a = float(np.exp(alpha))
        pred = a * x + b
        resid = pred - y
        if str(cfg.loss) == "huber_wls":
            r_scaled = sw * resid
            _, grad_scaled = _huber_scaled_loss_and_grad(
                r_scaled, delta=float(cfg.huber_delta)
            )
            # d loss / d resid
            dloss_dresid = sw * grad_scaled
        else:
            dloss_dresid = w * resid
        grad_alpha = float(np.sum(dloss_dresid * (a * x)))
        grad_b = float(np.sum(dloss_dresid))
        grad_alpha += 2.0 * float(cfg.lambda_ab) * alpha
        grad_b += 2.0 * float(cfg.lambda_ab) * b

        step = float(cfg.ab_lr)
        improved = False
        for _ in range(10):
            alpha_try = alpha - step * grad_alpha
            min_alpha = float(np.log(max(float(cfg.a_min), 1e-12)))
            alpha_try = float(max(alpha_try, min_alpha))
            b_try = b - step * grad_b
            obj_try = _objective(alpha_try, b_try)
            if obj_try <= best_obj + 1e-12:
                alpha, b, best_obj = alpha_try, b_try, obj_try
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return float(alpha), float(b), float(best_obj)


def fit_mismatch(
    *,
    obs_fit: list[ProbeObservation],
    y_sim_batch: SimFn,
    cfg: PhoenixMismatchConfig,
    init_phi: PhoenixPhi | None = None,
    theta_fallback: np.ndarray | None = None,
) -> tuple[PhoenixPhi, dict[str, float | bool]]:
    if not obs_fit:
        return PhoenixPhi(), {"fit_n": 0.0, "train_rmse_weighted": float("nan")}

    U = np.array([o.u for o in obs_fit], dtype=float)
    y = np.array([o.y_mean for o in obs_fit], dtype=float)
    y_var = np.array([o.y_var for o in obs_fit], dtype=float)
    y_var_eps = y_var + float(cfg.var_epsilon)
    w = 1.0 / np.maximum(y_var_eps, float(cfg.var_epsilon))
    w = w / max(float(np.mean(w)), 1e-12)

    theta_ref = None
    if theta_fallback is not None:
        theta_ref = np.asarray(theta_fallback, dtype=float).reshape(-1)
    theta_rows: list[np.ndarray] = []
    theta_fallback_used = False
    for o in obs_fit:
        if o.theta is None:
            if theta_ref is None:
                raise ValueError(
                    "ProbeObservation.theta is missing and theta_fallback was not provided."
                )
            theta_rows.append(np.asarray(theta_ref, dtype=float))
            theta_fallback_used = True
        else:
            theta_rows.append(np.asarray(o.theta, dtype=float).reshape(-1))
    Theta = np.vstack(theta_rows)

    if init_phi is None:
        s = np.ones((2,), dtype=float)
        t = np.zeros((2,), dtype=float)
        alpha = 0.0
        b = 0.0
    else:
        s = np.asarray(init_phi.s, dtype=float).copy()
        t = np.asarray(init_phi.t, dtype=float).copy()
        alpha = float(init_phi.alpha)
        b = float(init_phi.b)

    s = np.clip(s, float(cfg.warp_s_min), float(cfg.warp_s_max))
    t = np.clip(t, float(cfg.warp_t_min), float(cfg.warp_t_max))

    def _objective(s_v: np.ndarray, t_v: np.ndarray) -> tuple[float, float, float]:
        Uw = warp_clip(U, s=s_v, t=t_v)
        x = _sim_with_obs_theta(Uw, Theta=Theta, y_sim_batch=y_sim_batch)
        alpha_v, b_v, fit_obj = _fit_ab_for_fixed_warp(
            x=x,
            y=y,
            w=w,
            cfg=cfg,
            alpha_init=alpha,
            b_init=b,
        )
        reg_st = float(cfg.lambda_s) * float(np.sum((s_v - 1.0) ** 2)) + float(
            cfg.lambda_t
        ) * float(np.sum(t_v**2))
        return float(fit_obj + reg_st), float(alpha_v), float(b_v)

    best_obj, alpha, b = _objective(s, t)
    for _ in range(int(cfg.n_alt)):
        for _ in range(int(cfg.n_warp_steps)):
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
                up, _, _ = _objective(s + ds, t + dt)
                dn, _, _ = _objective(s - ds, t - dt)
                grad[j] = (up - dn) / (2.0 * eps)
            step = float(cfg.warp_lr)
            improved = False
            for _ in range(8):
                s_try = np.clip(
                    s - step * np.asarray([grad[0], grad[2]], dtype=float),
                    float(cfg.warp_s_min),
                    float(cfg.warp_s_max),
                )
                t_try = np.clip(
                    t - step * np.asarray([grad[1], grad[3]], dtype=float),
                    float(cfg.warp_t_min),
                    float(cfg.warp_t_max),
                )
                obj_try, alpha_try, b_try = _objective(s_try, t_try)
                if obj_try <= best_obj + 1e-10:
                    s, t = s_try, t_try
                    alpha, b = alpha_try, b_try
                    best_obj = obj_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        best_obj, alpha, b = _objective(s, t)

    phi = PhoenixPhi(alpha=float(alpha), b=float(b), s=s, t=t)
    Uw = warp_clip(U, s=phi.s, t=phi.t)
    x = _sim_with_obs_theta(Uw, Theta=Theta, y_sim_batch=y_sim_batch)
    y_hat = float(phi.a) * x + float(phi.b)
    rmse_w = float(np.sqrt(np.sum(w * ((y_hat - y) ** 2)) / np.sum(w)))
    noise_floor = float(np.sqrt(np.mean(y_var_eps)))
    return phi, {
        "fit_n": float(len(obs_fit)),
        "train_rmse_weighted": rmse_w,
        "noise_floor_rmse": noise_floor,
        "theta_fallback_used": bool(theta_fallback_used),
        "a_positive": bool(phi.a > 0.0),
    }
