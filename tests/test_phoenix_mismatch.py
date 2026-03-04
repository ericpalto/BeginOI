from __future__ import annotations

import numpy as np

from beginoi.tasks.single_paulsson.types import ProbeObservation
from beginoi.tasks.single_paulsson.phoenix_mismatch import (
    PhoenixMismatchConfig,
    y_hat_batch,
    fit_mismatch,
)


def _linear_sim(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    th = np.asarray(theta, dtype=float)
    return th[0] * U[:, 0] + th[1] * U[:, 1]


def test_phoenix_fit_uses_per_observation_theta() -> None:
    U = np.array(
        [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.9]],
        dtype=float,
    )
    thetas = np.array(
        [[0.5, 0.7], [0.4, 0.8], [0.6, 0.3], [0.7, 0.2], [0.2, 1.0]],
        dtype=float,
    )
    obs: list[ProbeObservation] = []
    for u, th in zip(U, thetas):
        y = float(1.3 * _linear_sim(u[None, :], th)[0] + 0.1)
        obs.append(
            ProbeObservation(
                u=u,
                y_reps=np.array([y, y + 1e-3], dtype=float),
                theta=th,
            )
        )

    phi, diag = fit_mismatch(
        obs_fit=obs,
        y_sim_batch=_linear_sim,
        cfg=PhoenixMismatchConfig(
            loss="wls",
            n_alt=5,
            n_warp_steps=1,
            n_ab_steps=40,
            warp_lr=0.05,
            lambda_s=1e-3,
            lambda_t=1e-3,
            lambda_ab=1e-4,
        ),
        theta_fallback=np.array([9.0, 9.0], dtype=float),
    )
    yhat = np.array(
        [
            y_hat_batch(u[None, :], theta=th, phi=phi, y_sim_batch=_linear_sim)[0]
            for u, th in zip(U, thetas)
        ],
        dtype=float,
    )
    ytrue = np.array([o.y_mean for o in obs], dtype=float)
    rmse = float(np.sqrt(np.mean((yhat - ytrue) ** 2)))
    assert rmse < 0.15
    assert bool(diag["theta_fallback_used"]) is False


def test_phoenix_fit_downweights_high_variance_outlier() -> None:
    obs: list[ProbeObservation] = []
    for x in np.linspace(0.1, 0.9, 8):
        y = float(x)
        obs.append(
            ProbeObservation(
                u=np.array([x, 0.0], dtype=float),
                y_reps=np.array([y, y + 1e-4], dtype=float),
                theta=np.array([1.0, 0.0], dtype=float),
            )
        )
    obs.append(
        ProbeObservation(
            u=np.array([0.5, 0.0], dtype=float),
            y_reps=np.array([2.0, 14.0], dtype=float),
            theta=np.array([1.0, 0.0], dtype=float),
        )
    )

    phi, _ = fit_mismatch(
        obs_fit=obs,
        y_sim_batch=_linear_sim,
        cfg=PhoenixMismatchConfig(
            loss="wls",
            n_alt=5,
            n_warp_steps=1,
            n_ab_steps=40,
            lambda_ab=1e-4,
            lambda_s=1e-3,
            lambda_t=1e-3,
        ),
    )
    pred = float(
        y_hat_batch(
            np.array([[0.5, 0.0]], dtype=float),
            theta=np.array([1.0, 0.0], dtype=float),
            phi=phi,
            y_sim_batch=_linear_sim,
        )[0]
    )
    assert abs(pred - 0.5) < 0.8


def test_phoenix_huber_wls_smoke() -> None:
    rng = np.random.default_rng(0)
    U = rng.uniform(0.0, 1.0, size=(24, 2))
    theta = np.array([0.7, 0.3], dtype=float)
    obs = []
    for u in U:
        y = float(_linear_sim(u[None, :], theta)[0] + rng.normal(0.0, 0.05))
        obs.append(
            ProbeObservation(
                u=u,
                y_reps=np.array([y, y + rng.normal(0.0, 0.02)], dtype=float),
                theta=theta,
            )
        )
    phi, diag = fit_mismatch(
        obs_fit=obs,
        y_sim_batch=_linear_sim,
        cfg=PhoenixMismatchConfig(loss="huber_wls", n_alt=4, n_warp_steps=1),
    )
    assert np.isfinite(float(phi.a))
    assert np.isfinite(float(phi.b))
    assert np.isfinite(float(diag["train_rmse_weighted"]))
