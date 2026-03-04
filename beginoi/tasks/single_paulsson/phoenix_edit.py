from __future__ import annotations

from dataclasses import field, dataclass

import numpy as np

from .phoenix_mismatch import PhoenixPhi, y_hat_batch


@dataclass(frozen=True)
class PhoenixEditConfig:
    """Configuration for PHOENIX one-shot edit and alpha selection."""

    eta: float = 1e-2
    theta_low: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.05, 0.05], dtype=float)
    )
    theta_high: np.ndarray = field(
        default_factory=lambda: np.array([0.9, 0.9, 0.85, 0.6], dtype=float)
    )
    n_starts: int = 32
    pattern_max_iters: int = 120
    init_step_frac: float = 0.2
    step_decay: float = 0.5
    step_tol: float = 1e-4
    alpha_candidates: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    min_pred_improvement: float = 1e-4


def _first_primes(n: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < int(n):
        is_prime = True
        for p in primes:
            if candidate % p == 0:
                is_prime = False
                break
            if p * p > candidate:
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(index: int, base: int) -> float:
    out = 0.0
    denom = 1.0
    i = int(index)
    b = int(base)
    while i > 0:
        i, rem = divmod(i, b)
        denom *= b
        out += rem / denom
    return float(out)


def _halton_points(n: int, d: int) -> np.ndarray:
    n = int(max(0, n))
    d = int(max(0, d))
    if n == 0 or d == 0:
        return np.zeros((0, d), dtype=float)
    primes = _first_primes(d)
    out = np.zeros((n, d), dtype=float)
    for i in range(n):
        for j in range(d):
            out[i, j] = _van_der_corput(i + 1, primes[j])
    return out


def _predicted_loss(
    U: np.ndarray,
    *,
    theta: np.ndarray,
    phi: PhoenixPhi,
    y_sim_batch,
    g_batch,
) -> float:
    pred = y_hat_batch(U, theta=theta, phi=phi, y_sim_batch=y_sim_batch)
    g = np.asarray(g_batch(U), dtype=float)
    return float(np.mean((pred - g) ** 2))


def _objective(
    theta: np.ndarray,
    *,
    theta_ref: np.ndarray,
    phi: PhoenixPhi,
    cfg: PhoenixEditConfig,
    U_mc: np.ndarray,
    y_sim_batch,
    g_batch,
) -> float:
    theta = np.asarray(theta, dtype=float)
    low = np.asarray(cfg.theta_low, dtype=float)
    high = np.asarray(cfg.theta_high, dtype=float)
    theta = np.clip(theta, low, high)
    mse = _predicted_loss(
        U_mc, theta=theta, phi=phi, y_sim_batch=y_sim_batch, g_batch=g_batch
    )
    reg = float(cfg.eta) * float(np.sum((theta - theta_ref) ** 2))
    return float(mse + reg)


def _pattern_search(
    start: np.ndarray,
    *,
    obj_fn,
    low: np.ndarray,
    high: np.ndarray,
    cfg: PhoenixEditConfig,
) -> tuple[np.ndarray, float]:
    theta = np.clip(np.asarray(start, dtype=float), low, high)
    value = float(obj_fn(theta))
    span = np.maximum(high - low, 1e-8)
    step = float(cfg.init_step_frac) * float(np.max(span))
    max_iters = int(max(1, cfg.pattern_max_iters))
    for _ in range(max_iters):
        improved = False
        for j in range(len(theta)):
            basis = np.zeros_like(theta)
            basis[j] = 1.0
            for sign in (+1.0, -1.0):
                cand = np.clip(theta + sign * step * basis, low, high)
                v = float(obj_fn(cand))
                if v + 1e-12 < value:
                    theta, value = cand, v
                    improved = True
        if not improved:
            step *= float(cfg.step_decay)
            if step < float(cfg.step_tol):
                break
    return theta, float(value)


def solve_theta_one_shot(
    *,
    theta_k: np.ndarray,
    phi: PhoenixPhi,
    cfg: PhoenixEditConfig,
    U_mc: np.ndarray,
    y_sim_batch,
    g_batch,
) -> tuple[np.ndarray, dict[str, float]]:
    theta_k = np.asarray(theta_k, dtype=float).reshape(-1)
    low = np.asarray(cfg.theta_low, dtype=float)
    high = np.asarray(cfg.theta_high, dtype=float)
    if theta_k.shape != low.shape or theta_k.shape != high.shape:
        raise ValueError(
            f"Theta bounds shape mismatch: theta={theta_k.shape}, "
            f"low={low.shape}, high={high.shape}."
        )

    def _obj(theta: np.ndarray) -> float:
        return _objective(
            theta,
            theta_ref=theta_k,
            phi=phi,
            cfg=cfg,
            U_mc=np.asarray(U_mc, dtype=float),
            y_sim_batch=y_sim_batch,
            g_batch=g_batch,
        )

    starts = [np.clip(theta_k, low, high)]
    halton = _halton_points(int(max(0, cfg.n_starts)), len(theta_k))
    if len(halton) > 0:
        starts.extend([low + row * (high - low) for row in halton])

    best_theta = starts[0]
    best_obj = float(_obj(best_theta))
    for start in starts:
        cand, val = _pattern_search(start, obj_fn=_obj, low=low, high=high, cfg=cfg)
        if val + 1e-12 < best_obj:
            best_theta, best_obj = cand, val

    obj_k = float(_obj(np.clip(theta_k, low, high)))
    pred_improvement = float(obj_k - best_obj)
    return np.asarray(best_theta, dtype=float), {
        "pred_obj_mc": float(best_obj),
        "pred_obj_mc_k": float(obj_k),
        "pred_obj_mc_improvement": float(pred_improvement),
    }


def select_alpha_predicted(
    *,
    theta_k: np.ndarray,
    theta_star: np.ndarray,
    phi: PhoenixPhi,
    cfg: PhoenixEditConfig,
    U_post: np.ndarray,
    y_sim_batch,
    g_batch,
    alpha_scale: float = 1.0,
    drop_alpha_one: bool = False,
) -> tuple[np.ndarray, dict[str, float | list[float]]]:
    theta_k = np.asarray(theta_k, dtype=float).reshape(-1)
    theta_star = np.asarray(theta_star, dtype=float).reshape(-1)
    low = np.asarray(cfg.theta_low, dtype=float)
    high = np.asarray(cfg.theta_high, dtype=float)
    U_post = np.asarray(U_post, dtype=float)

    cands_raw = [float(a) for a in cfg.alpha_candidates]
    cands = [float(a * float(alpha_scale)) for a in cands_raw]
    cands = [float(np.clip(a, 0.0, 1.0)) for a in cands]
    if bool(drop_alpha_one):
        cands = [a for a in cands if a < 0.999]
    if not cands:
        cands = [0.5]

    losses: list[float] = []
    thetas: list[np.ndarray] = []
    for alpha in cands:
        theta_alpha = np.clip(theta_k + alpha * (theta_star - theta_k), low, high)
        loss = _predicted_loss(
            U_post, theta=theta_alpha, phi=phi, y_sim_batch=y_sim_batch, g_batch=g_batch
        )
        thetas.append(theta_alpha)
        losses.append(float(loss))
    best_idx = int(np.argmin(np.asarray(losses, dtype=float)))
    theta_next = thetas[best_idx]
    return np.asarray(theta_next, dtype=float), {
        "alpha_chosen": float(cands[best_idx]),
        "post_pred_loss_chosen": float(losses[best_idx]),
        "alpha_candidates_used": [float(a) for a in cands],
        "alpha_losses_pred": [float(v) for v in losses],
    }
