from __future__ import annotations

from typing import Any, Protocol
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Heatmap, GridSpec


@dataclass(frozen=True)
class BenchmarkContext:
    """Metadata container for a benchmark instance."""

    name: str
    grid: GridSpec
    meta: dict[str, Any]


class Benchmark(Protocol):
    """Benchmark protocol: constructs a plant and defines oracle evaluation."""

    name: str
    grid: GridSpec

    def make_plant(self, *, regime: Any, seed: int) -> Any:
        ...

    def oracle_heatmap(self, theta: Any, *, seed: int) -> Heatmap:
        ...

    def simulator_heatmap(self, plant: Any, theta: Any) -> Heatmap:
        ...


def make_grid(*, n1: int, n2: int, low: float = 0.0, high: float = 1.0) -> GridSpec:
    return GridSpec(
        x1=np.linspace(low, high, int(n1), dtype=float),
        x2=np.linspace(low, high, int(n2), dtype=float),
    )
