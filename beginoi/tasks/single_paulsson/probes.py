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

    audit_stratified: bool = True
    audit_candidate_pool: int = 2048
    audit_alpha_uniform: float = 0.5
    audit_beta_high: float = 0.3
    audit_gamma_boundary: float = 0.2
    boundary_tau_low: float = 0.45
    boundary_tau_high: float = 0.55
    max_rejection_tries: int = 64
    design_spacefill_frac: float = 0.7
    design_target_high_frac: float = 0.2
    design_target_boundary_frac: float = 0.1
    design_anchor_count: int = 4


@dataclass(frozen=True)
class PhoenixProbeConfig:
    """Probe sampling parameters for PHOENIX."""

    audit_pool_size: int = 2048
    post_pool_size: int = 1024
    alpha_uniform: float = 0.5
    beta_high: float = 0.3
    gamma_boundary: float = 0.2
    boundary_tau_low: float = 0.45
    boundary_tau_high: float = 0.55
    max_rejection_tries: int = 64
    design_spacefill_frac_early: float = 0.55
    design_spacefill_frac_stable: float = 0.75
    design_target_high_frac: float = 0.20
    design_target_boundary_frac: float = 0.10
    design_anchor_count_early: int = 8
    design_anchor_count_stable: int = 2
    post_alpha_uniform: float = 0.4
    post_beta_high: float = 0.4
    post_gamma_boundary: float = 0.2


def _sample_design_anchors(rng: np.random.Generator, n: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    canonical = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    perm = rng.permutation(len(canonical))
    idx = np.resize(perm, n)
    return canonical[idx]


def sample_audit(
    rng: np.random.Generator, *, cfg: ProbeSamplingConfig, g_batch: TargetFn, n: int
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    if bool(cfg.audit_stratified):
        pool_n = max(int(cfg.audit_candidate_pool), n)
        w = np.array(
            [cfg.audit_alpha_uniform, cfg.audit_beta_high, cfg.audit_gamma_boundary],
            dtype=float,
        )
        w = np.maximum(w, 0.0)
        if float(np.sum(w)) <= 0:
            w = np.array([1.0, 0.0, 0.0], dtype=float)
        w = w / np.sum(w)
        counts = rng.multinomial(pool_n, w)
        u1 = rng.uniform(0.0, 1.0, size=(int(counts[0]), 2))
        u2 = _rejection_high_target(
            rng,
            g_batch=g_batch,
            n=int(counts[1]),
            max_tries=int(cfg.max_rejection_tries),
        )
        u3 = _band_target(
            rng,
            g_batch=g_batch,
            n=int(counts[2]),
            tau_low=float(cfg.boundary_tau_low),
            tau_high=float(cfg.boundary_tau_high),
            max_tries=int(cfg.max_rejection_tries),
        )
        pool = np.vstack([u1, u2, u3])
        if len(pool) < n:
            pad = rng.uniform(0.0, 1.0, size=(n - len(pool), 2))
            pool = np.vstack([pool, pad])
        g = np.asarray(g_batch(pool), dtype=float)
        order = np.argsort(g)
        if len(order) <= n:
            out = pool[order]
        else:
            idx = np.linspace(0, len(order) - 1, n, dtype=int)
            out = pool[order[idx]]
        rng.shuffle(out, axis=0)
        return np.clip(out, 0.0, 1.0)

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
    n_anchor = min(max(int(cfg.design_anchor_count), 0), n)
    n_remaining = max(0, n - n_anchor)
    n_space = int(round(float(cfg.design_spacefill_frac) * n_remaining))
    n_high = int(round(float(cfg.design_target_high_frac) * n_remaining))
    n_band = max(0, n_remaining - n_space - n_high)
    u_anchor = _sample_design_anchors(rng, n_anchor)
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
    out = np.vstack([u_anchor, u_space, u_high, u_band])
    rng.shuffle(out, axis=0)
    return np.clip(out[:n], 0.0, 1.0)


def _mixture_pool(
    rng: np.random.Generator,
    *,
    g_batch: TargetFn,
    n: int,
    alpha_uniform: float,
    beta_high: float,
    gamma_boundary: float,
    tau_low: float,
    tau_high: float,
    max_rejection_tries: int,
) -> np.ndarray:
    n = int(max(0, n))
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    w = np.array([alpha_uniform, beta_high, gamma_boundary], dtype=float)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0:
        w = np.array([1.0, 0.0, 0.0], dtype=float)
    w = w / float(np.sum(w))
    counts = rng.multinomial(n, w)
    u1 = rng.uniform(0.0, 1.0, size=(int(counts[0]), 2))
    u2 = _rejection_high_target(
        rng,
        g_batch=g_batch,
        n=int(counts[1]),
        max_tries=int(max_rejection_tries),
    )
    u3 = _band_target(
        rng,
        g_batch=g_batch,
        n=int(counts[2]),
        tau_low=float(tau_low),
        tau_high=float(tau_high),
        max_tries=int(max_rejection_tries),
    )
    out = np.vstack([u1, u2, u3])
    rng.shuffle(out, axis=0)
    return np.clip(out, 0.0, 1.0)


def build_audit_pool_mixture(
    rng: np.random.Generator, *, cfg: PhoenixProbeConfig, g_batch: TargetFn
) -> np.ndarray:
    return _mixture_pool(
        rng,
        g_batch=g_batch,
        n=int(cfg.audit_pool_size),
        alpha_uniform=float(cfg.alpha_uniform),
        beta_high=float(cfg.beta_high),
        gamma_boundary=float(cfg.gamma_boundary),
        tau_low=float(cfg.boundary_tau_low),
        tau_high=float(cfg.boundary_tau_high),
        max_rejection_tries=int(cfg.max_rejection_tries),
    )


def build_post_pool_mixture(
    rng: np.random.Generator, *, cfg: PhoenixProbeConfig, g_batch: TargetFn
) -> np.ndarray:
    return _mixture_pool(
        rng,
        g_batch=g_batch,
        n=int(cfg.post_pool_size),
        alpha_uniform=float(cfg.post_alpha_uniform),
        beta_high=float(cfg.post_beta_high),
        gamma_boundary=float(cfg.post_gamma_boundary),
        tau_low=float(cfg.boundary_tau_low),
        tau_high=float(cfg.boundary_tau_high),
        max_rejection_tries=int(cfg.max_rejection_tries),
    )


def rotate_pool_slice(
    pool: np.ndarray, *, start: int, n: int
) -> tuple[np.ndarray, int]:
    """Take a deterministic wrapped slice from a fixed pool."""
    pool = np.asarray(pool, dtype=float)
    n = int(max(0, n))
    if n == 0 or len(pool) == 0:
        return np.zeros((0, 2), dtype=float), int(start)
    idx = (np.arange(n, dtype=int) + int(start)) % int(len(pool))
    out = np.asarray(pool[idx], dtype=float)
    next_start = int((int(start) + n) % int(len(pool)))
    return out, next_start


def sample_design_phoenix(
    rng: np.random.Generator,
    *,
    cfg: PhoenixProbeConfig,
    g_batch: TargetFn,
    n: int,
    stable: bool,
) -> np.ndarray:
    n = int(max(0, n))
    if n == 0:
        return np.zeros((0, 2), dtype=float)

    if stable:
        n_anchor = min(max(int(cfg.design_anchor_count_stable), 0), n)
        frac_space = float(cfg.design_spacefill_frac_stable)
    else:
        n_anchor = min(max(int(cfg.design_anchor_count_early), 0), n)
        frac_space = float(cfg.design_spacefill_frac_early)

    n_remaining = max(0, n - n_anchor)
    n_space = int(round(frac_space * n_remaining))
    n_high = int(round(float(cfg.design_target_high_frac) * n_remaining))
    n_band = int(round(float(cfg.design_target_boundary_frac) * n_remaining))
    n_space = max(0, min(n_space, n_remaining))
    n_high = max(0, min(n_high, n_remaining - n_space))
    n_band = max(0, min(n_band, n_remaining - n_space - n_high))
    n_fill = max(0, n_remaining - n_space - n_high - n_band)
    n_space += n_fill

    u_anchor = _sample_design_anchors(rng, n_anchor)
    u_space = _latin_hypercube(rng, n_space, d=2)
    u_high = _rejection_high_target(
        rng,
        g_batch=g_batch,
        n=n_high,
        max_tries=int(cfg.max_rejection_tries),
    )
    u_band = _band_target(
        rng,
        g_batch=g_batch,
        n=n_band,
        tau_low=float(cfg.boundary_tau_low),
        tau_high=float(cfg.boundary_tau_high),
        max_tries=int(cfg.max_rejection_tries),
    )
    out = np.vstack([u_anchor, u_space, u_high, u_band])
    rng.shuffle(out, axis=0)
    return np.clip(out[:n], 0.0, 1.0)
