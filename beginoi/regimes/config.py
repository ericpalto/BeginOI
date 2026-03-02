from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

CopyMode = Literal["single", "multi"]
Instrument = Literal["standard_lab", "paulsson_machine"]
ThetaDynamics = Literal["discrete", "continuous"]
ProgramKind = Literal["constant", "timeseries"]
ObservationSchedule = Literal["end_only", "streaming"]
ExposureModel = Literal["none", "piecewise_constant", "continuous"]


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration describing one experimental regime ("quadrant")."""

    copy_mode: CopyMode
    instrument: Instrument
    theta_dynamics: ThetaDynamics
    feedback: bool
    program_kind: ProgramKind
    max_programs_per_unit: int
    observation_schedule: ObservationSchedule
    exposure_model: ExposureModel
