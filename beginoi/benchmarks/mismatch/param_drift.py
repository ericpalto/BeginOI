from __future__ import annotations

from typing import Any, Mapping
from dataclasses import dataclass

import numpy as np

from beginoi.core.param_utils import apply_delta


@dataclass(frozen=True)
class ParamDriftMismatch:
    """Exposure-driven additive drift.

    Update rule: params[field] += rate[field] * level * dt
    """

    rates: Mapping[str, Any]

    def step(self, params: Any, *, level: float, dt: float) -> Any:
        if dt <= 0:
            return params
        deltas = {
            k: np.asarray(v) * float(level) * float(dt) for k, v in self.rates.items()
        }
        return apply_delta(params, deltas)
