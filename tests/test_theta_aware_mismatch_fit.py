from __future__ import annotations

import numpy as np

from beginoi.tasks.single_paulsson.types import ProbeObservation
from beginoi.tasks.single_paulsson.mismatch import MismatchFitConfig, fit_mismatch


def _linear_sim(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return theta[0] * U[:, 0] + theta[1] * U[:, 1]


def test_fit_mismatch_uses_per_observation_theta() -> None:
    U = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.1],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.9, 0.8],
        ],
        dtype=float,
    )
    thetas = np.array(
        [
            [0.5, 0.2],
            [0.8, -0.1],
            [1.1, 0.3],
            [0.7, 0.6],
            [1.3, -0.2],
        ],
        dtype=float,
    )
    obs: list[ProbeObservation] = []
    for u, th in zip(U, thetas):
        y = float(_linear_sim(u[None, :], th)[0])
        obs.append(
            ProbeObservation(
                u=u,
                theta=th,
                y_reps=np.array([y, y], dtype=float),
            )
        )

    cfg = MismatchFitConfig(
        use_affine_y=True,
        use_input_warp=False,
        use_rbf_residual=False,
        enforce_positive_a=False,
    )
    phi, diag = fit_mismatch(
        theta_k=np.array([9.0, 9.0], dtype=float),
        obs_design=obs,
        y_sim_batch=_linear_sim,
        cfg=cfg,
    )

    assert abs(float(phi.a) - 1.0) < 5e-4
    assert abs(float(phi.b)) < 1e-3
    assert float(diag["train_rmse"]) < 1e-3
    assert bool(diag["theta_fallback_used"]) is False


def test_fit_mismatch_falls_back_to_theta_k_when_theta_missing() -> None:
    U = np.array([[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]], dtype=float)
    theta_k = np.array([0.6, 0.4], dtype=float)
    obs = []
    for u in U:
        y = float(_linear_sim(u[None, :], theta_k)[0])
        obs.append(
            ProbeObservation(
                u=u,
                y_reps=np.array([y, y], dtype=float),
            )
        )

    cfg = MismatchFitConfig(
        use_affine_y=True,
        use_input_warp=False,
        use_rbf_residual=False,
        enforce_positive_a=False,
    )
    phi, diag = fit_mismatch(
        theta_k=theta_k,
        obs_design=obs,
        y_sim_batch=_linear_sim,
        cfg=cfg,
    )

    assert abs(float(phi.a) - 1.0) < 5e-4
    assert abs(float(phi.b)) < 1e-3
    assert float(diag["train_rmse"]) < 1e-3
    assert bool(diag["theta_fallback_used"]) is True
