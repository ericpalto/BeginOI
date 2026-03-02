from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuredResidualMismatch:
    """Additive residual on the oracle output as a smooth function of (x1,x2)."""

    weights: np.ndarray
    centers: np.ndarray
    lengthscale: float

    @staticmethod
    def random(
        *,
        seed: int,
        num_features: int = 16,
        lengthscale: float = 0.25,
        weight_scale: float = 0.05,
    ) -> "StructuredResidualMismatch":
        rng = np.random.default_rng(int(seed))
        centers = rng.uniform(0.0, 1.0, size=(int(num_features), 2))
        weights = rng.normal(0.0, float(weight_scale), size=(int(num_features),))
        return StructuredResidualMismatch(
            weights=weights.astype(float),
            centers=centers.astype(float),
            lengthscale=float(lengthscale),
        )

    def residual(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (2,):
            raise ValueError(f"Expected x shape (2,), got {x.shape}.")
        diff = self.centers - x[None, :]
        r2 = np.sum(diff**2, axis=1)
        feats = np.exp(-0.5 * r2 / (self.lengthscale**2 + 1e-12))
        return float(np.dot(self.weights, feats))

    def apply(self, y: float, *, x: np.ndarray) -> tuple[float, dict[str, Any]]:
        r = self.residual(x)
        return float(y + r), {"structured_residual": r}


@dataclass(frozen=True)
class RandomStructuredResidual:
    """Config wrapper that builds a randomized StructuredResidualMismatch."""

    seed: int = 0
    num_features: int = 16
    lengthscale: float = 0.25
    weight_scale: float = 0.05

    def build(self) -> StructuredResidualMismatch:
        return StructuredResidualMismatch.random(
            seed=int(self.seed),
            num_features=int(self.num_features),
            lengthscale=float(self.lengthscale),
            weight_scale=float(self.weight_scale),
        )
