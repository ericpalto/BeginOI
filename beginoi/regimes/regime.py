from __future__ import annotations

from dataclasses import dataclass

from beginoi.core.types import ProgramKind, ControlAction, InterventionKind

from .config import RegimeConfig


@dataclass(frozen=True)
class Regime:
    """Runtime helper that enforces regime constraints on actions."""

    cfg: RegimeConfig

    @property
    def feedback(self) -> bool:
        return bool(self.cfg.feedback)

    @property
    def max_programs_per_unit(self) -> int:
        return int(self.cfg.max_programs_per_unit)

    @property
    def program_kind(self) -> ProgramKind:
        return self.cfg.program_kind

    def allowed_interventions(self) -> set[InterventionKind]:
        allowed: set[InterventionKind] = {"theta_edit"}
        if self.cfg.exposure_model != "none":
            allowed.add("exposure_schedule")
        return allowed

    def validate_action(self, action: ControlAction) -> None:
        programs = action.batch.programs
        if len(programs) > self.max_programs_per_unit:
            raise ValueError(
                f"Too many programs ({len(programs)}) for regime capacity "
                f"{self.max_programs_per_unit}."
            )
        for p in programs:
            if p.kind != self.program_kind:
                raise ValueError(
                    f"Program kind {p.kind!r} not allowed "
                    f"(requires {self.program_kind!r})."
                )
        if action.intervention is None:
            return
        if action.intervention.kind not in self.allowed_interventions():
            raise ValueError(
                f"Intervention {action.intervention.kind!r} not allowed in this regime."
            )
        if not self.cfg.feedback and action.intervention.kind == "exposure_schedule":
            # Exposure schedules can exist without feedback, but only as
            # per-unit actions.
            return
