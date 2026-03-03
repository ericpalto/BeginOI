from __future__ import annotations

import numpy as np

from beginoi.core.types import Program
from beginoi.benchmarks.bnn_ode import BNNXORFunctionBenchmark
from beginoi.benchmarks.mismatch.noise import ObservationNoise


def test_bnn_xor_function_benchmark_builds_two_node_xor_plant() -> None:
    benchmark = BNNXORFunctionBenchmark(
        grid_n1=3,
        grid_n2=3,
        surface_n=9,
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
    assert np.asarray(theta.theta, dtype=float).shape == (4,)


def test_bnn_xor_function_benchmark_exposes_sparc_batch_apis() -> None:
    benchmark = BNNXORFunctionBenchmark(
        grid_n1=3,
        grid_n2=3,
        surface_n=9,
        noise=ObservationNoise(0.0),
        structured_residual=None,
    )
    U = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    theta = np.array(benchmark.theta0, dtype=float)
    y_sim = benchmark.simulator_y_batch(U, theta)
    y_target = benchmark.target_g_batch(U)

    assert y_sim.shape == (4,)
    assert y_target.shape == (4,)
    assert np.all(np.isfinite(y_sim))
    assert np.all(np.isfinite(y_target))
