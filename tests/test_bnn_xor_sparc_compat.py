from __future__ import annotations

import numpy as np

from beginoi.core.types import Budget
from beginoi.core.runner import run_loop
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime
from beginoi.benchmarks.bnn_ode import BNNXORFunctionBenchmark
from beginoi.metrics.sparc_audit import SparcAuditMetric
from beginoi.policies.sparc_policy import SparcPolicy
from beginoi.core.experiments.basic import BasicBatchExperiment
from beginoi.benchmarks.mismatch.noise import ObservationNoise


def test_bnn_xor_supports_single_paulsson_sparc_loop() -> None:
    regime = Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="paulsson_machine",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="timeseries",
            max_programs_per_unit=32,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )
    benchmark = BNNXORFunctionBenchmark(
        grid_n1=5,
        grid_n2=5,
        surface_n=11,
        noise=ObservationNoise(0.0),
        structured_residual=None,
    )
    plant = benchmark.make_plant(regime=regime, seed=0)
    policy = SparcPolicy(
        n_max=2,
        replicates=1,
        n_mc=64,
        sim_warmstart={"enabled": False},
        edit={"n_candidates": 32},
    )
    experiment = BasicBatchExperiment(
        max_programs=int(regime.max_programs_per_unit),
        cost_per_unit=1.0,
    )
    summary = run_loop(
        plant=plant,
        experiment=experiment,
        policy=policy,
        regime=regime,
        benchmark=benchmark,
        budget=Budget(total=1.0),
        metrics=[SparcAuditMetric()],
        rng=np.random.default_rng(0),
        loggers=[],
    )
    assert len(summary.history.observations) == 2
    assert "sparc_audit_mse" in summary.final_metrics
