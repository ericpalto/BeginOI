from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Heatmap


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(np.abs(diff)))


@dataclass(frozen=True)
class HeatmapErrorMetric:
    """Compute RMSE/MAE between simulator and oracle heatmaps."""

    name_prefix: str = "heatmap"
    oracle_seed: int = 0

    def __call__(self, history, benchmark: Any, *, plant: Any) -> dict[str, Any]:
        theta = history.theta_snapshots[-1]
        sim_map: Heatmap = benchmark.simulator_heatmap(plant, theta)
        oracle_map: Heatmap = benchmark.oracle_heatmap(theta, seed=self.oracle_seed)
        return {
            f"{self.name_prefix}_rmse": _rmse(sim_map.y, oracle_map.y),
            f"{self.name_prefix}_mae": _mae(sim_map.y, oracle_map.y),
        }
