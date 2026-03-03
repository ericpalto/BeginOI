from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np


@dataclass(frozen=True)
class Probe:
    """A single probe location `u` in [0,1]^2."""

    u: np.ndarray  # shape (2,)

    def __post_init__(self) -> None:
        u = np.asarray(self.u, dtype=float)
        if u.shape != (2,):
            raise ValueError(f"Probe.u must have shape (2,), got {u.shape}.")
        object.__setattr__(self, "u", u)


@dataclass(frozen=True)
class ProbeObservation:
    """Replicated observations at a single probe location."""

    u: np.ndarray  # shape (2,)
    y_reps: np.ndarray  # shape (R,)
    theta: np.ndarray | None = None  # shape (D,) if available
    y_mean: float = field(init=False)
    y_var: float = field(init=False)

    def __post_init__(self) -> None:
        u = np.asarray(self.u, dtype=float)
        if u.shape != (2,):
            raise ValueError(f"ProbeObservation.u must have shape (2,), got {u.shape}.")
        y = np.asarray(self.y_reps, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"ProbeObservation.y_reps must be 1D, got {y.shape}.")
        theta = self.theta
        if theta is not None:
            theta_arr = np.asarray(theta, dtype=float)
            if theta_arr.ndim != 1:
                raise ValueError(
                    f"ProbeObservation.theta must be 1D when present, "
                    f"got {theta_arr.shape}."
                )
            object.__setattr__(self, "theta", theta_arr)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "y_reps", y)
        object.__setattr__(self, "y_mean", float(np.mean(y)))
        object.__setattr__(
            self, "y_var", float(np.var(y, ddof=1) if len(y) > 1 else 0.0)
        )


@dataclass
class RoundData:
    """One SPARC round: probes, observations, fit params, and metrics."""

    theta_k: np.ndarray
    U_audit: np.ndarray
    U_design: np.ndarray
    obs_audit: list[ProbeObservation]
    obs_design: list[ProbeObservation]
    phi_fit: dict[str, Any]
    metrics: dict[str, Any]
    fit_diagnostics: dict[str, Any]


@dataclass
class ExperimentHistory:
    """SPARC round history container."""

    rounds: list[RoundData] = field(default_factory=list)
