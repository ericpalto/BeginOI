from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Budget, Program
from beginoi.core.runner import run_loop
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime
from beginoi.core.experiments.basic import BasicBatchExperiment
from beginoi.policies.phoenix_policy import PhoenixPolicy


@dataclass
class ToyTheta:
    theta: np.ndarray
    time: float = 0.0


class ToyPlant:
    def __init__(self, *, theta0: np.ndarray, post_bias: float) -> None:
        self.theta0 = np.asarray(theta0, dtype=float)
        self.post_bias = float(post_bias)

    def reset(self, seed: int) -> ToyTheta:
        del seed
        return ToyTheta(theta=self.theta0.copy())

    def simulate(self, program: Program, theta: ToyTheta) -> float:
        u = np.asarray(program.as_constant_inputs(), dtype=float)
        return float(theta.theta[0] * u[0] + theta.theta[1] * u[1])

    def observe(self, program: Program, theta: ToyTheta, rng) -> float:
        del rng
        y = self.simulate(program, theta)
        role = str(program.meta.get("phoenix_role", ""))
        if role == "post_step":
            y += float(self.post_bias)
        return float(y)

    def apply_intervention(self, theta: ToyTheta, intervention, dt):
        del dt
        if intervention is None:
            return theta
        if intervention.kind == "theta_edit":
            vec = np.asarray(intervention.payload["set"]["theta"], dtype=float)
            return ToyTheta(theta=vec.copy(), time=float(theta.time))
        return theta

    def evaluate_heatmap(self, theta: ToyTheta, grid_spec):
        del theta, grid_spec
        raise NotImplementedError


class ToyBenchmark:
    name = "toy_phoenix"
    grid = None
    theta_low = np.array([0.0, 0.0], dtype=float)
    theta_high = np.array([1.2, 1.2], dtype=float)

    def __init__(self, *, post_bias: float = 0.0) -> None:
        self.post_bias = float(post_bias)

    @staticmethod
    def target_g_batch(U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        return 0.8 * U[:, 0] + 0.2 * U[:, 1]

    @staticmethod
    def simulator_y_batch(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        theta = np.asarray(theta, dtype=float)
        return theta[0] * U[:, 0] + theta[1] * U[:, 1]

    def make_plant(self, *, regime, seed: int) -> ToyPlant:
        del regime, seed
        return ToyPlant(
            theta0=np.array([0.4, 0.6], dtype=float), post_bias=self.post_bias
        )


def _regime() -> Regime:
    return Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="paulsson_machine",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=128,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )


def _run(policy: PhoenixPolicy, *, benchmark: ToyBenchmark, budget_total: float = 4.0):
    regime = _regime()
    plant = benchmark.make_plant(regime=regime, seed=0)
    return run_loop(
        plant=plant,
        experiment=BasicBatchExperiment(max_programs=128, cost_per_unit=1.0),
        policy=policy,
        regime=regime,
        benchmark=benchmark,
        budget=Budget(total=float(budget_total)),
        metrics=[],
        rng=np.random.default_rng(0),
        loggers=[],
    )


def test_phoenix_fit_dataset_uses_design_only_by_default() -> None:
    policy = PhoenixPolicy(
        n_max=8,
        replicates=1,
        audit_frac=0.25,
        n_mc=64,
        post_step={"enabled": True, "n_post": 2, "fit_include_post_step": False},
        stop={"k_max": 3},
        edit={"theta_low": [0.0, 0.0], "theta_high": [1.2, 1.2], "n_starts": 8},
    )
    out = _run(policy, benchmark=ToyBenchmark(post_bias=0.0), budget_total=3.0)
    recs = list(out.history.extras.get("phoenix_rounds", []))
    design_total = int(sum(int(rec["U_design_n"]) for rec in recs))
    audit_total = int(sum(int(rec["U_audit_n"]) for rec in recs))
    assert len(policy._fit_dataset_design) == design_total
    assert len(policy._fit_dataset_post) == 0
    assert design_total < design_total + audit_total


def test_phoenix_post_step_overshoot_triggers_damping() -> None:
    policy = PhoenixPolicy(
        n_max=8,
        replicates=1,
        audit_frac=0.25,
        n_mc=64,
        post_step={
            "enabled": True,
            "n_post": 2,
            "fit_include_post_step": False,
            "damping_mode": "drop_alpha_one",
            "damping_rounds": 2,
            "overshoot_kappa": 1.0,
        },
        stop={"k_max": 4},
        edit={"theta_low": [0.0, 0.0], "theta_high": [1.2, 1.2], "n_starts": 8},
    )
    out = _run(policy, benchmark=ToyBenchmark(post_bias=1.0), budget_total=4.0)
    recs = list(out.history.extras.get("phoenix_rounds", []))
    overshoot_flags = [
        bool(rec.get("post_step_validation", {}).get("overshoot", False))
        for rec in recs
    ]
    assert any(overshoot_flags)
    assert any(int(rec.get("damping_rounds_left", 0)) > 0 for rec in recs)


def test_phoenix_stop_respects_kmax() -> None:
    policy = PhoenixPolicy(
        n_max=6,
        replicates=1,
        audit_frac=0.25,
        n_mc=32,
        stop={"k_max": 2},
        edit={"theta_low": [0.0, 0.0], "theta_high": [1.2, 1.2], "n_starts": 6},
    )
    out = _run(policy, benchmark=ToyBenchmark(post_bias=0.0), budget_total=10.0)
    assert len(out.history.actions) == 2


def test_phoenix_design_stability_flag_can_turn_on() -> None:
    policy = PhoenixPolicy(
        n_max=8,
        replicates=1,
        audit_frac=0.25,
        n_mc=64,
        stability={
            "eps_alpha": 1.0,
            "eps_b": 1.0,
            "eps_s": 1.0,
            "eps_t": 1.0,
            "rounds_required": 1,
        },
        stop={"k_max": 3},
        edit={"theta_low": [0.0, 0.0], "theta_high": [1.2, 1.2], "n_starts": 8},
    )
    out = _run(policy, benchmark=ToyBenchmark(post_bias=0.0), budget_total=3.0)
    recs = list(out.history.extras.get("phoenix_rounds", []))
    flags = [bool(rec.get("design_stable", False)) for rec in recs]
    assert any(flags[1:])
