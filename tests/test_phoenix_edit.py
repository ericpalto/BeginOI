from __future__ import annotations

import numpy as np

from beginoi.tasks.single_paulsson.phoenix_edit import (
    PhoenixEditConfig,
    solve_theta_one_shot,
    select_alpha_predicted,
)
from beginoi.tasks.single_paulsson.phoenix_mismatch import PhoenixPhi


def _sim(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    th = np.asarray(theta, dtype=float)
    return th[0] * U[:, 0]


def _target(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    return 0.6 * U[:, 0]


def test_phoenix_one_shot_solver_is_deterministic_and_improves_objective() -> None:
    U_mc = np.array(
        [(x, y) for x in np.linspace(0, 1, 16) for y in np.linspace(0, 1, 4)]
    )
    phi = PhoenixPhi()
    cfg = PhoenixEditConfig(
        theta_low=np.array([0.0], dtype=float),
        theta_high=np.array([1.0], dtype=float),
        n_starts=16,
        pattern_max_iters=60,
    )
    theta_k = np.array([0.2], dtype=float)
    t1, d1 = solve_theta_one_shot(
        theta_k=theta_k,
        phi=phi,
        cfg=cfg,
        U_mc=U_mc,
        y_sim_batch=_sim,
        g_batch=_target,
    )
    t2, d2 = solve_theta_one_shot(
        theta_k=theta_k,
        phi=phi,
        cfg=cfg,
        U_mc=U_mc,
        y_sim_batch=_sim,
        g_batch=_target,
    )
    assert np.allclose(t1, t2)
    assert float(d1["pred_obj_mc"]) <= float(d1["pred_obj_mc_k"]) + 1e-12
    assert np.isclose(float(d1["pred_obj_mc"]), float(d2["pred_obj_mc"]))


def test_phoenix_alpha_selection_chooses_predicted_argmin() -> None:
    U_post = np.array([[0.2, 0.5], [0.7, 0.2], [1.0, 0.4]], dtype=float)

    def target(U: np.ndarray) -> np.ndarray:
        return 0.5 * np.asarray(U, dtype=float)[:, 0]

    cfg = PhoenixEditConfig(
        theta_low=np.array([0.0], dtype=float),
        theta_high=np.array([1.0], dtype=float),
        alpha_candidates=(1.0, 0.5, 0.25, 0.125),
    )
    theta_next, diag = select_alpha_predicted(
        theta_k=np.array([0.0], dtype=float),
        theta_star=np.array([1.0], dtype=float),
        phi=PhoenixPhi(),
        cfg=cfg,
        U_post=U_post,
        y_sim_batch=_sim,
        g_batch=target,
    )
    assert np.allclose(theta_next, np.array([0.5], dtype=float), atol=1e-6)
    assert np.isclose(float(diag["alpha_chosen"]), 0.5)
