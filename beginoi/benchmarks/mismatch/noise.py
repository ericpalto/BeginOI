from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationNoise:
    """Gaussian observation noise applied to oracle outputs."""

    sigma: float = 0.0

    def apply(
        self, y: float, *, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        if self.sigma <= 0:
            return float(y), {"sigma": float(self.sigma)}
        eps = float(rng.normal(0.0, self.sigma))
        return float(y + eps), {"sigma": float(self.sigma), "eps": eps}
