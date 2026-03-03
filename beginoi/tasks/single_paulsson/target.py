from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class BandpassTarget:
    """Analytic difference-of-sigmoids ring target g(u) in [0,1]."""

    center: tuple[float, float] = (0.5, 0.5)
    r_mid: float = 0.45
    thickness: float = 0.18
    width: float = 0.03

    def g_batch(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[None, :]
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError(f"Expected U shape (N,2), got {U.shape}.")
        center = np.asarray(self.center, dtype=float)
        r = np.sqrt(np.sum((U - center[None, :]) ** 2, axis=1))
        thickness = float(max(self.thickness, 1e-6))
        r_mid = float(self.r_mid)
        r_in = max(0.0, r_mid - 0.5 * thickness)
        r_out = max(r_in + 1e-6, r_mid + 0.5 * thickness)
        w = float(max(self.width, 1e-6))
        band = _sigmoid((r - r_in) / w) - _sigmoid((r - r_out) / w)
        return np.clip(band, 0.0, 1.0)

    def g(self, u: np.ndarray) -> float:
        return float(self.g_batch(np.asarray(u, dtype=float))[0])

    def dense_grid(self, *, n: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x1, x2, g) where g has shape (n,n). For visualization only."""
        x1 = np.linspace(0.0, 1.0, int(n), dtype=float)
        x2 = np.linspace(0.0, 1.0, int(n), dtype=float)
        xs = np.array([(a, b) for a in x1 for b in x2], dtype=float)
        g = self.g_batch(xs).reshape((len(x1), len(x2)))
        return x1, x2, g
