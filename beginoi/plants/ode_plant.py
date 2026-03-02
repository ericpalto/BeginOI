from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass

import numpy as np

from beginoi.core.types import Heatmap, Program, GridSpec, Intervention
from beginoi.core.param_utils import apply_set, apply_delta
from beginoi.benchmarks.mismatch.noise import ObservationNoise
from beginoi.benchmarks.mismatch.param_drift import ParamDriftMismatch
from beginoi.benchmarks.mismatch.structured_residual import StructuredResidualMismatch

from .state import SimToRealThetaState

RhsFn = Callable[[float, np.ndarray, np.ndarray, Any], np.ndarray]


@dataclass(frozen=True)
class ODESpec:
    """Defines an ODE system to integrate to steady-state."""

    state_size: int
    rhs: RhsFn
    t_final: float = 10.0
    dt: float = 0.02


class GenericODESimToRealPlant:
    """Sim-to-real plant backed by a simple explicit Euler ODE simulator."""

    def __init__(
        self,
        *,
        ode: ODESpec,
        sim_params: Any,
        real_params: Any,
        output_index: int = -1,
        y0: np.ndarray | None = None,
        noise: ObservationNoise | None = None,
        structured_residual: StructuredResidualMismatch | None = None,
        param_drift: ParamDriftMismatch | None = None,
    ) -> None:
        self.ode = ode
        self.output_index = int(output_index)
        self.y0 = (
            np.zeros((ode.state_size,), dtype=float)
            if y0 is None
            else np.asarray(y0, dtype=float)
        )
        self.noise = ObservationNoise(0.0) if noise is None else noise
        self.structured_residual = structured_residual
        self.param_drift = param_drift
        self._initial = SimToRealThetaState(
            sim_params=sim_params, real_params=real_params
        )

    def reset(self, seed: int) -> SimToRealThetaState:
        del seed
        return SimToRealThetaState(
            sim_params=self._initial.sim_params,
            real_params=self._initial.real_params,
            time=0.0,
            exposure_level=0.0,
            meta={},
        )

    def _simulate_params(self, program: Program, params: Any) -> float:
        x = program.as_constant_inputs()
        state = np.asarray(self.y0, dtype=float).copy()
        t = 0.0
        while t < self.ode.t_final - 1e-12:
            state = state + self.ode.dt * self.ode.rhs(t, state, x, params)
            state = np.maximum(state, 0.0)
            t += self.ode.dt
        return float(state[self.output_index])

    def simulate(self, program: Program, theta: SimToRealThetaState) -> float:
        return self._simulate_params(program, theta.sim_params)

    def observe(self, program: Program, theta: SimToRealThetaState, rng: Any) -> float:
        if rng is None:
            rng = np.random.default_rng(0)
        x = program.as_constant_inputs()
        y = self._simulate_params(program, theta.real_params)
        noise_meta: dict[str, Any] = {}
        if self.structured_residual is not None:
            y, meta = self.structured_residual.apply(y, x=x)
            noise_meta.update(meta)
        y, meta = self.noise.apply(y, rng=rng)
        noise_meta.update(meta)
        program.meta.setdefault("noise_meta", noise_meta)
        return float(y)

    def apply_intervention(
        self,
        theta: SimToRealThetaState,
        intervention: Intervention,
        dt: float | None,
    ) -> SimToRealThetaState:
        if intervention.kind == "theta_edit":
            payload = intervention.payload or {}
            sim_params = theta.sim_params
            if "set" in payload:
                sim_params = apply_set(sim_params, payload["set"])
            if "delta" in payload:
                sim_params = apply_delta(sim_params, payload["delta"])
            return SimToRealThetaState(
                sim_params=sim_params,
                real_params=theta.real_params,
                time=theta.time,
                exposure_level=theta.exposure_level,
                meta=dict(theta.meta),
            )

        if intervention.kind == "exposure_schedule":
            payload = intervention.payload or {}
            level = float(payload.get("level", theta.exposure_level))
            duration = float(payload.get("duration", 0.0))
            step_dt = float(duration if dt is None else dt)
            real_params = theta.real_params
            if self.param_drift is not None:
                real_params = self.param_drift.step(
                    real_params, level=level, dt=step_dt
                )
            return SimToRealThetaState(
                sim_params=theta.sim_params,
                real_params=real_params,
                time=float(theta.time + step_dt),
                exposure_level=level,
                meta=dict(theta.meta),
            )

        raise ValueError(f"Unknown intervention kind: {intervention.kind!r}")

    def evaluate_heatmap(
        self, theta: SimToRealThetaState, grid_spec: GridSpec
    ) -> Heatmap:
        y = np.zeros((len(grid_spec.x1), len(grid_spec.x2)), dtype=float)
        for i, x1 in enumerate(grid_spec.x1):
            for j, x2 in enumerate(grid_spec.x2):
                program = Program(kind="constant", u=np.array([x1, x2], dtype=float))
                y[i, j] = self.simulate(program, theta)
        return Heatmap(grid=grid_spec, y=y)
