from __future__ import annotations

import numpy as np

from beginoi.core.types import Program
from beginoi.benchmarks.bnn_ode import BNNXORFunctionBenchmark
from beginoi.benchmarks.mismatch.noise import ObservationNoise


def test_bnn_xor_function_benchmark_builds_two_node_xor_plant() -> None:
    benchmark = BNNXORFunctionBenchmark(
        grid_n1=3,
        grid_n2=3,
        noise=ObservationNoise(0.0),
        structured_residual=None,
    )
    plant = benchmark.make_plant(regime=None, seed=0)

    assert plant.formulation == "moorman"
    assert plant.model == "two_node_xor"

    theta = plant.reset(seed=0)
    y = plant.simulate(
        Program(kind="constant", u=np.array([0.0, 1.0], dtype=float)),
        theta,
    )
    assert np.isfinite(y)
