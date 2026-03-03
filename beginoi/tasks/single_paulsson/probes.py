from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

TargetFn = Callable[[np.ndarray], np.ndarray]  # g_batch(U)->(N,)


def _latin_hypercube(rng: np.random.Generator, n: int, d: int = 2) -> np.ndarray:
    n = int(n)
    d = int(d)
    if n <= 0:
        return np.zeros((0, d), dtype=float)
    cut = np.linspace(0.0, 1.0, n + 1, dtype=float)
    out = np.zeros((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        u = rng.uniform(low=cut[:-1], high=cut[1:], size=(n,))
        out[:, j] = u[perm]
    return np.clip(out, 0.0, 1.0)


def _rejection_high_target(
    rng: np.random.Generator,
    *,
    g_batch: TargetFn,
    n: int,
    max_tries: int,
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    accepted: list[np.ndarray] = []
    tries = 0
    while len(accepted) < n and tries < max_tries:
        tries += 1
        U = rng.uniform(0.0, 1.0, size=(max(64, n), 2))
        g = np.clip(g_batch(U), 0.0, 1.0)
        keep = rng.uniform(0.0, 1.0, size=(len(U),)) < g
        if np.any(keep):
            accepted.append(U[keep])
    if not accepted:
        return rng.uniform(0.0, 1.0, size=(n, 2))
    U = np.vstack(accepted)
    if len(U) < n:
        pad = rng.uniform(0.0, 1.0, size=(n - len(U), 2))
        U = np.vstack([U, pad])
    return U[:n]


def _band_target(
    rng: np.random.Generator,
    *,
    g_batch: TargetFn,
    n: int,
    tau_low: float,
    tau_high: float,
    max_tries: int,
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    accepted: list[np.ndarray] = []
    tries = 0
    while len(accepted) < n and tries < max_tries:
        tries += 1
        U = rng.uniform(0.0, 1.0, size=(max(128, n), 2))
        g = g_batch(U)
        keep = (g >= float(tau_low)) & (g <= float(tau_high))
        if np.any(keep):
            accepted.append(U[keep])
    if not accepted:
        return rng.uniform(0.0, 1.0, size=(n, 2))
    U = np.vstack(accepted)
    if len(U) < n:
        pad = rng.uniform(0.0, 1.0, size=(n - len(U), 2))
        U = np.vstack([U, pad])
    return U[:n]


@dataclass(frozen=True)
class ProbeSamplingConfig:
    """Heuristic probe sampling parameters for audit/design sets."""

    audit_alpha_uniform: float = 0.5
    audit_beta_high: float = 0.3
    audit_gamma_boundary: float = 0.2
    boundary_tau_low: float = 0.45
    boundary_tau_high: float = 0.55
    max_rejection_tries: int = 64
    design_spacefill_frac: float = 0.7
    design_target_high_frac: float = 0.2
    design_target_boundary_frac: float = 0.1


def sample_audit(
    rng: np.random.Generator, *, cfg: ProbeSamplingConfig, g_batch: TargetFn, n: int
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    w = np.array(
        [cfg.audit_alpha_uniform, cfg.audit_beta_high, cfg.audit_gamma_boundary],
        dtype=float,
    )
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0:
        w = np.array([1.0, 0.0, 0.0], dtype=float)
    w = w / np.sum(w)
    counts = rng.multinomial(n, w)
    u1 = rng.uniform(0.0, 1.0, size=(int(counts[0]), 2))
    u2 = _rejection_high_target(
        rng, g_batch=g_batch, n=int(counts[1]), max_tries=int(cfg.max_rejection_tries)
    )
    u3 = _band_target(
        rng,
        g_batch=g_batch,
        n=int(counts[2]),
        tau_low=float(cfg.boundary_tau_low),
        tau_high=float(cfg.boundary_tau_high),
        max_tries=int(cfg.max_rejection_tries),
    )
    out = np.vstack([u1, u2, u3])
    rng.shuffle(out, axis=0)
    return np.clip(out[:n], 0.0, 1.0)


def sample_design_phase0(
    rng: np.random.Generator, *, cfg: ProbeSamplingConfig, g_batch: TargetFn, n: int
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    n_space = int(round(float(cfg.design_spacefill_frac) * n))
    n_high = int(round(float(cfg.design_target_high_frac) * n))
    n_band = max(0, n - n_space - n_high)
    u_space = _latin_hypercube(rng, n_space, d=2)
    u_high = _rejection_high_target(
        rng, g_batch=g_batch, n=n_high, max_tries=int(cfg.max_rejection_tries)
    )
    u_band = _band_target(
        rng,
        g_batch=g_batch,
        n=n_band,
        tau_low=float(cfg.boundary_tau_low),
        tau_high=float(cfg.boundary_tau_high),
        max_tries=int(cfg.max_rejection_tries),
    )
    out = np.vstack([u_space, u_high, u_band])
    rng.shuffle(out, axis=0)
    return np.clip(out[:n], 0.0, 1.0)
