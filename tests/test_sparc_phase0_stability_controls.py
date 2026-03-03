from __future__ import annotations

import os
import sys
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from beginoi.policies.sparc_policy import (
    _effective_mismatch_cfg,
    _select_fit_observations,
)
from beginoi.tasks.single_paulsson.types import ProbeObservation
from beginoi.tasks.single_paulsson.mismatch import MismatchFitConfig, fit_mismatch


def _obs(u: tuple[float, float], y: float) -> ProbeObservation:
    return ProbeObservation(
        u=np.asarray(u, dtype=float),
        y_reps=np.asarray([y, y + 0.01], dtype=float),
    )


def test_history_fit_modes_choose_expected_design_observations() -> None:
    history = [
        [_obs((0.1, 0.1), 0.2), _obs((0.2, 0.2), 0.3)],
        [_obs((0.3, 0.3), 0.4), _obs((0.4, 0.4), 0.5), _obs((0.5, 0.5), 0.6)],
        [_obs((0.6, 0.6), 0.7)],
    ]

    cumulative_counts: list[int] = []
    for i in range(1, len(history) + 1):
        prefix = history[:i]
        cumulative = _select_fit_observations(
            prefix, mode="cumulative", window_rounds=2
        )
        per_round = _select_fit_observations(prefix, mode="per_round", window_rounds=2)
        windowed = _select_fit_observations(prefix, mode="window", window_rounds=2)

        cumulative_counts.append(len(cumulative))
        assert len(per_round) == len(prefix[-1])
        expected_window = sum(len(chunk) for chunk in prefix[max(0, i - 2) : i])
        assert len(windowed) == expected_window

    assert all(
        cumulative_counts[i] > cumulative_counts[i - 1]
        for i in range(1, len(cumulative_counts))
    )


def test_effective_mismatch_cfg_stages_warp_and_rbf() -> None:
    base = MismatchFitConfig(
        use_affine_y=False,
        use_input_warp=True,
        use_rbf_residual=True,
        min_design_points_for_warp=5,
        min_design_points_for_rbf=8,
    )

    low = _effective_mismatch_cfg(base, fit_n=4)
    mid = _effective_mismatch_cfg(base, fit_n=6)
    high = _effective_mismatch_cfg(base, fit_n=9)

    assert low.use_affine_y is True
    assert low.use_input_warp is False
    assert low.use_rbf_residual is False
    assert mid.use_input_warp is True
    assert mid.use_rbf_residual is False
    assert high.use_input_warp is True
    assert high.use_rbf_residual is True


def test_fit_mismatch_enforces_positive_a_constraint() -> None:
    U = np.array(
        [
            [0.0, 0.2],
            [0.15, 0.4],
            [0.3, 0.1],
            [0.45, 0.7],
            [0.6, 0.5],
            [0.75, 0.3],
            [0.9, 0.9],
        ],
        dtype=float,
    )
    obs = [
        ProbeObservation(u=u, y_reps=np.array([1.0 - u[0], 1.0 - u[0]], dtype=float))
        for u in U
    ]

    def y_sim_batch(inp: np.ndarray, theta: np.ndarray) -> np.ndarray:
        del theta
        return np.asarray(inp, dtype=float)[:, 0]

    theta_k = np.zeros((4,), dtype=float)
    unconstrained_cfg = MismatchFitConfig(
        use_affine_y=True,
        use_input_warp=False,
        use_rbf_residual=False,
        enforce_positive_a=False,
    )
    constrained_cfg = MismatchFitConfig(
        use_affine_y=True,
        use_input_warp=False,
        use_rbf_residual=False,
        enforce_positive_a=True,
        a_min=0.05,
        a_max=3.0,
    )

    phi_unconstrained, _ = fit_mismatch(
        theta_k=theta_k,
        obs_design=obs,
        y_sim_batch=y_sim_batch,
        cfg=unconstrained_cfg,
    )
    phi_constrained, diag = fit_mismatch(
        theta_k=theta_k,
        obs_design=obs,
        y_sim_batch=y_sim_batch,
        cfg=constrained_cfg,
    )

    assert phi_unconstrained.a < 0.0
    assert phi_constrained.a >= 0.05
    assert np.isfinite(phi_constrained.b)
    assert bool(diag["a_clamped"]) is True


def test_smoke_sparc_cumulative_diagnostics_and_positive_gain(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_sparc_stable"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd())
    cmd = [
        sys.executable,
        "-m",
        "beginoi.cli",
        "task=run",
        "regime=single_paulsson",
        "benchmark=paulsson_machine",
        "policy=sparc",
        "metrics=sparc",
        "tracking=local_only",
        "seed=0",
        "budget.total=10.0",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
    ]
    subprocess.check_call(cmd, env=env)

    rounds_path = run_dir / "sparc_rounds.jsonl"
    assert rounds_path.exists()
    rounds = [
        json.loads(line)
        for line in rounds_path.read_text().splitlines()
        if line.strip()
    ]
    assert rounds

    fit_ns = [int(rec["diagnostics"]["fit_n"]) for rec in rounds]
    assert all(fit_ns[i] >= fit_ns[i - 1] for i in range(1, len(fit_ns)))
    assert all(str(rec["diagnostics"]["fit_mode"]) == "cumulative" for rec in rounds)
    assert all(float(rec["phi_fit"]["a"]) >= 0.05 for rec in rounds)

    warp_flags = [bool(rec["diagnostics"]["warp_enabled"]) for rec in rounds]
    assert any(flag is False for flag in warp_flags)
    assert any(flag is True for flag in warp_flags)
    assert all(bool(rec["diagnostics"]["rbf_enabled"]) is False for rec in rounds)

    history_rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert history_rows
    first = history_rows[0]
    assert int(first["schema_version"]) == 2
    assert "round_id" in first
    assert "theta" in first
    assert isinstance(first["theta"], list)
    assert len(first["theta"]) == 4

    summary = json.loads((run_dir / "summary.json").read_text())
    assert "final_metrics" in summary
    assert math.isfinite(float(summary["final_metrics"]["sparc_audit_mse"]))
