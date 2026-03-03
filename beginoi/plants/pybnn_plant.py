from __future__ import annotations

from typing import Any
from dataclasses import fields, replace, is_dataclass

import numpy as np
import pybnn
from pybnn.api import Circuit

from beginoi.core.types import Heatmap, Program, GridSpec, Intervention
from beginoi.core.param_utils import apply_set, apply_delta
from beginoi.benchmarks.mismatch.noise import ObservationNoise
from beginoi.benchmarks.mismatch.param_drift import ParamDriftMismatch
from beginoi.benchmarks.mismatch.structured_residual import StructuredResidualMismatch

from .state import SimToRealThetaState


def _constrained_to_raw(circuit: Any, constrained_params: Any) -> Any:
    transforms = getattr(circuit, "param_transforms", None)
    if not transforms:
        return constrained_params
    if not is_dataclass(constrained_params):
        return constrained_params

    converted: dict[str, Any] = {}
    for f in fields(constrained_params):
        name = f.name
        value = getattr(constrained_params, name)
        transform = transforms.get(name)
        if transform is None:
            converted[name] = value
        else:
            converted[name] = np.asarray(
                transform.inverse(np.asarray(value, dtype=float))
            )
    return replace(constrained_params, **converted)


def _steady_state_output(
    circuit: Circuit,
    x: np.ndarray,
    *,
    t_final: float,
    dt: float,
) -> float:
    return float(circuit.steady_state_output(x, t_final=float(t_final), dt=float(dt)))


class PyBNNSimToRealPlant:
    """Sim-to-real plant backed by a PyBNN circuit for simulator and oracle."""

    def __init__(
        self,
        *,
        formulation: str = "moorman",
        model: str = "perceptron",
        backend: str = "numpy",
        t_final: float = 5.0,
        dt: float = 0.02,
        noise: ObservationNoise | None = None,
        structured_residual: StructuredResidualMismatch | None = None,
        param_drift: ParamDriftMismatch | None = None,
    ) -> None:
        self.formulation = str(formulation)
        self.model = str(model)
        self.backend = str(backend)
        self.t_final = float(t_final)
        self.dt = float(dt)
        self.noise = ObservationNoise(0.0) if noise is None else noise
        self.structured_residual = structured_residual
        self.param_drift = param_drift

    def _make_circuit(self, params: Any) -> Any:
        if self.backend == "jax":
            return pybnn.create_circuit(
                self.formulation,
                self.model,
                backend="jax",
                params=params,
            )
        return pybnn.create_circuit(
            self.formulation,
            self.model,
            backend="numpy",
            params=params,
        )

    def reset(self, seed: int) -> SimToRealThetaState:
        rng = np.random.default_rng(int(seed))
        base_circuit = self._make_circuit(params=None)
        sim_params = getattr(base_circuit, "params", None)
        if sim_params is None:
            raise RuntimeError("PyBNN circuit does not expose constrained params.")
        real_params = sim_params
        # Small default mismatch: multiplicative jitter on numeric fields if dataclass.
        if is_dataclass(sim_params):
            updates: dict[str, Any] = {}
            for f in fields(sim_params):
                v = getattr(sim_params, f.name)
                if isinstance(v, (int, float, np.integer, np.floating)):
                    updates[f.name] = float(v) * float(
                        rng.lognormal(mean=0.0, sigma=0.05)
                    )
            real_params = replace(sim_params, **updates)
        return SimToRealThetaState(sim_params=sim_params, real_params=real_params)

    def simulate(self, program: Program, theta: SimToRealThetaState) -> float:
        x = program.as_constant_inputs()
        params = theta.sim_params
        circuit = self._make_circuit(params)
        return _steady_state_output(circuit, x, t_final=self.t_final, dt=self.dt)

    def observe(self, program: Program, theta: SimToRealThetaState, rng: Any) -> float:
        if rng is None:
            rng = np.random.default_rng(0)
        x = program.as_constant_inputs()
        params = theta.real_params
        circuit = self._make_circuit(params)
        y = _steady_state_output(circuit, x, t_final=self.t_final, dt=self.dt)
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


class MoormanXORFunctionPlant(PyBNNSimToRealPlant):
    """Specialized PyBNN plant for the Moorman two-node XOR function."""

    def __init__(
        self,
        *,
        backend: str = "numpy",
        t_final: float = 30.0,
        dt: float = 0.005,
        noise: ObservationNoise | None = None,
        structured_residual: StructuredResidualMismatch | None = None,
        param_drift: ParamDriftMismatch | None = None,
    ) -> None:
        super().__init__(
            formulation="moorman",
            model="two_node_xor",
            backend=backend,
            t_final=t_final,
            dt=dt,
            noise=noise,
            structured_residual=structured_residual,
            param_drift=param_drift,
        )
