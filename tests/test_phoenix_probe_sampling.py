from __future__ import annotations

import numpy as np

from beginoi.tasks.single_paulsson.probes import (
    PhoenixProbeConfig,
    rotate_pool_slice,
    sample_design_phoenix,
    build_audit_pool_mixture,
)


def _target(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    return np.clip(1.0 - np.sum((U - 0.5) ** 2, axis=1) / 0.5, 0.0, 1.0)


def test_rotate_pool_slice_wraparound_deterministic() -> None:
    pool = np.array(
        [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype=float
    )
    s1, c1 = rotate_pool_slice(pool, start=3, n=4)
    s2, c2 = rotate_pool_slice(pool, start=3, n=4)
    assert np.allclose(s1, s2)
    assert c1 == c2
    expected = np.array([[0.3, 0.3], [0.4, 0.4], [0.0, 0.0], [0.1, 0.1]], dtype=float)
    assert np.allclose(s1, expected)
    assert c1 == 2


def test_build_audit_pool_is_in_unit_box() -> None:
    rng = np.random.default_rng(0)
    cfg = PhoenixProbeConfig(audit_pool_size=256)
    U = build_audit_pool_mixture(rng, cfg=cfg, g_batch=_target)
    assert U.shape == (256, 2)
    assert float(np.min(U)) >= 0.0
    assert float(np.max(U)) <= 1.0


def test_sample_design_phoenix_early_has_more_anchor_points() -> None:
    rng = np.random.default_rng(0)
    cfg = PhoenixProbeConfig(
        design_anchor_count_early=8,
        design_anchor_count_stable=1,
        design_spacefill_frac_early=0.4,
        design_spacefill_frac_stable=0.8,
    )
    early = sample_design_phoenix(rng, cfg=cfg, g_batch=_target, n=24, stable=False)
    stable = sample_design_phoenix(rng, cfg=cfg, g_batch=_target, n=24, stable=True)
    anchors = {
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.0),
        (0.5, 1.0),
        (0.0, 0.5),
        (1.0, 0.5),
        (0.5, 0.5),
    }
    early_anchor_hits = sum(tuple(np.round(u, 6)) in anchors for u in early)
    stable_anchor_hits = sum(tuple(np.round(u, 6)) in anchors for u in stable)
    assert early_anchor_hits >= stable_anchor_hits
