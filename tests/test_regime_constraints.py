from __future__ import annotations

import numpy as np
import pytest

from beginoi.core.types import Program, Intervention, ProgramBatch, ControlAction
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime


def test_regime_rejects_over_capacity_batch() -> None:
    regime = Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="standard_lab",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=1,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )
    action = ControlAction(
        batch=ProgramBatch(
            programs=[
                Program(kind="constant", u=np.array([0.1, 0.2])),
                Program(kind="constant", u=np.array([0.2, 0.3])),
            ]
        )
    )
    with pytest.raises(ValueError):
        regime.validate_action(action)


def test_regime_rejects_wrong_program_kind() -> None:
    regime = Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="standard_lab",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=2,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )
    action = ControlAction(
        batch=ProgramBatch(programs=[Program(kind="timeseries", u=np.zeros((3, 2)))]),
    )
    with pytest.raises(ValueError):
        regime.validate_action(action)


def test_regime_rejects_disallowed_intervention() -> None:
    regime = Regime(
        cfg=RegimeConfig(
            copy_mode="single",
            instrument="standard_lab",
            theta_dynamics="discrete",
            feedback=False,
            program_kind="constant",
            max_programs_per_unit=1,
            observation_schedule="end_only",
            exposure_model="none",
        )
    )
    action = ControlAction(
        batch=ProgramBatch(programs=[Program(kind="constant", u=np.array([0.1, 0.2]))]),
        intervention=Intervention(kind="exposure_schedule", payload={"level": 1.0}),
    )
    with pytest.raises(ValueError):
        regime.validate_action(action)
