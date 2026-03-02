from __future__ import annotations

import numpy as np

from beginoi.core.types import Budget
from beginoi.core.runner import run_loop
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime
from beginoi.core.experiments.basic import BasicBatchExperiment
from beginoi.policies.random_policy import RandomPolicy
from beginoi.benchmarks.ode_lotka_volterra import LotkaVolterraBenchmark


def test_determinism_same_seed_same_observations() -> None:
    seed = 123
    regime = Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="standard_lab",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=1,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )
    benchmark = LotkaVolterraBenchmark(grid_n1=5, grid_n2=5, noise=None)
    plant = benchmark.make_plant(regime=regime, seed=seed)
    experiment = BasicBatchExperiment(max_programs=1, cost_per_unit=1.0)
    policy1 = RandomPolicy(batch_size=1)
    policy2 = RandomPolicy(batch_size=1)

    out1 = run_loop(
        plant=plant,
        experiment=experiment,
        policy=policy1,
        regime=regime,
        benchmark=benchmark,
        budget=Budget(total=3.0),
        metrics=[],
        rng=np.random.default_rng(seed),
        loggers=[],
    )
    # Re-create plant to avoid shared state.
    plant2 = benchmark.make_plant(regime=regime, seed=seed)
    out2 = run_loop(
        plant=plant2,
        experiment=experiment,
        policy=policy2,
        regime=regime,
        benchmark=benchmark,
        budget=Budget(total=3.0),
        metrics=[],
        rng=np.random.default_rng(seed),
        loggers=[],
    )

    ys1 = [o.y for o in out1.history.observations]
    ys2 = [o.y for o in out2.history.observations]
    assert ys1 == ys2
