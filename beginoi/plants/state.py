from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass


@dataclass
class SimToRealThetaState:
    """Holds simulator params and real/oracle params plus optional dynamics state."""

    sim_params: Any
    real_params: Any
    time: float = 0.0
    exposure_level: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
