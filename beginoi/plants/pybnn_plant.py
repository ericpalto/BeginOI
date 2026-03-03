from __future__ import annotations

from typing import Any
from functools import lru_cache
from dataclasses import field, fields, replace, dataclass, is_dataclass

import jax
import numpy as np
import pybnn
import diffrax
import jax.numpy as jnp
from pybnn.api import Circuit

from beginoi.core.types import Heatmap, Program, GridSpec, Intervention
from beginoi.core.param_utils import apply_set, apply_delta
from beginoi.benchmarks.mismatch.noise import ObservationNoise
from beginoi.benchmarks.mismatch.param_drift import ParamDriftMismatch
from beginoi.benchmarks.mismatch.structured_residual import StructuredResidualMismatch

from .state import SimToRealThetaState


def _steady_state_output(
    circuit: Circuit,
    x: np.ndarray,
    *,
    t_final: float | None,
    dt: float | None,
) -> float:
    kwargs: dict[str, float] = {}
    if t_final is not None:
        kwargs["t_final"] = float(t_final)
    if dt is not None:
        kwargs["dt"] = float(dt)
    return float(circuit.steady_state_output(np.asarray(x, dtype=float), **kwargs))


def _three_node_xor_output_batch_jax(
    U: np.ndarray,
    *,
    params: Any,
    t_final: float,
    dt: float,
) -> np.ndarray:
    U_batch = _as_u_batch(U)
    w1 = jnp.asarray(getattr(params, "w1"), dtype=jnp.float32)
    w2 = jnp.asarray(getattr(params, "w2"), dtype=jnp.float32)
    w3 = jnp.asarray(getattr(params, "w3"), dtype=jnp.float32)
    gamma = jnp.asarray(float(getattr(params, "gamma")), dtype=jnp.float32)
    phi = jnp.asarray(float(getattr(params, "phi")), dtype=jnp.float32)

    steps = int(np.ceil(float(t_final) / float(dt)))
    t1 = float(steps * float(dt))
    solve_batch = _get_three_node_xor_solver(steps=steps, t1=t1, dt=float(dt))
    y_batch = solve_batch(
        jnp.asarray(U_batch, dtype=jnp.float32),
        w1,
        w2,
        w3,
        gamma,
        phi,
    )
    return np.asarray(y_batch, dtype=float)


@lru_cache(maxsize=16)
def _get_three_node_xor_solver(*, steps: int, t1: float, dt: float):
    y0 = jnp.zeros((7,), dtype=jnp.float32)

    def solve_batch(
        batch_inputs: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w3: jax.Array,
        gamma: jax.Array,
        phi: jax.Array,
    ) -> jax.Array:
        def perceptron_pair(
            z1: jax.Array,
            z2: jax.Array,
            *,
            u: jax.Array,
            v: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            dz1 = u - gamma * z1 * z2 - phi * z1
            dz2 = v - gamma * z1 * z2 - phi * z2
            return dz1, dz2

        def solve_one(inputs: jax.Array) -> jax.Array:
            x1, x2 = inputs[0], inputs[1]

            def vector_field(
                _t_value: jax.Array,
                state: jax.Array,
                args: None,
            ) -> jax.Array:
                del args
                state = jnp.maximum(state, 0.0)
                n11_z1, n11_z2, n12_z1, n12_z2, n21_z1, n21_z2, _y = state

                dn11_z1, dn11_z2 = perceptron_pair(
                    n11_z1,
                    n11_z2,
                    u=w1[0],
                    v=w1[1] * x1 + w1[2] * x2,
                )
                dn12_z1, dn12_z2 = perceptron_pair(
                    n12_z1,
                    n12_z2,
                    u=w2[0] * x1 + w2[1] * x2,
                    v=w2[2],
                )
                dn21_z1, dn21_z2 = perceptron_pair(
                    n21_z1,
                    n21_z2,
                    u=w3[0] * n11_z1 + w3[1] * n12_z1,
                    v=w3[2],
                )
                dy = dn21_z1
                return jnp.array(
                    [dn11_z1, dn11_z2, dn12_z1, dn12_z2, dn21_z1, dn21_z2, dy],
                    dtype=state.dtype,
                )

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(vector_field),
                diffrax.Tsit5(),
                t0=0.0,
                t1=t1,
                dt0=dt,
                y0=y0,
                args=None,
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=diffrax.ConstantStepSize(),
                max_steps=steps + 1,
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
                throw=False,
            )
            final_state = jnp.maximum(solution.ys[0], 0.0)
            return final_state[6]

        return jax.vmap(solve_one)(batch_inputs)

    return jax.jit(solve_batch)


def _steady_state_output_batch(
    *,
    formulation: str,
    model: str,
    backend: str,
    params: Any,
    U: np.ndarray,
    t_final: float | None,
    dt: float | None,
) -> np.ndarray:
    U_batch = _as_u_batch(U)

    if backend == "jax" and formulation == "moorman" and model == "three_node_xor":
        if t_final is None or dt is None:
            raise ValueError(
                "JAX batched integration requires finite t_final and dt values."
            )
        if params is None:
            params = pybnn.create_circuit(
                formulation,
                model,
                backend="numpy",
            ).params
        return _three_node_xor_output_batch_jax(
            U_batch,
            params=params,
            t_final=float(t_final),
            dt=float(dt),
        )

    circuit = pybnn.create_circuit(
        formulation,
        model,
        backend=backend,  # type: ignore[arg-type]
        params=params,
    )
    y = np.empty((U_batch.shape[0],), dtype=float)
    for i, u in enumerate(U_batch):
        y[i] = _steady_state_output(
            circuit,
            u,
            t_final=t_final,
            dt=dt,
        )
    return y


def _jitter_numeric_value(value: Any, rng: np.random.Generator) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) * float(rng.lognormal(mean=0.0, sigma=0.05))
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return np.asarray(value, dtype=float) * rng.lognormal(
            mean=0.0,
            sigma=0.05,
            size=value.shape,
        )
    return value


def _as_u_batch(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError(f"Expected U with shape (N,2), got {U.shape}.")
    return U


def _warp_by_theta(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    U = _as_u_batch(U)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (4,):
        raise ValueError(f"Expected theta shape (4,), got {theta.shape}.")
    warped = np.empty_like(U, dtype=float)
    warped[:, 0] = theta[0] * U[:, 0] + theta[1]
    warped[:, 1] = theta[2] * U[:, 1] + theta[3]
    return np.clip(warped, 0.0, 1.0)


def _bilinear_interp(
    U: np.ndarray,
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    U = _as_u_batch(U)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(x1), len(x2)):
        raise ValueError(
            "Surface values shape must match grid sizes "
            f"({len(x1)}, {len(x2)}), got {values.shape}."
        )
    if len(x1) < 2 or len(x2) < 2:
        raise ValueError("Interpolation grid must have at least 2 points per axis.")

    u1 = np.clip(U[:, 0], float(x1[0]), float(x1[-1]))
    u2 = np.clip(U[:, 1], float(x2[0]), float(x2[-1]))

    i = np.clip(np.searchsorted(x1, u1, side="right") - 1, 0, len(x1) - 2)
    j = np.clip(np.searchsorted(x2, u2, side="right") - 1, 0, len(x2) - 2)

    x1_lo = x1[i]
    x1_hi = x1[i + 1]
    x2_lo = x2[j]
    x2_hi = x2[j + 1]
    tx = (u1 - x1_lo) / np.maximum(x1_hi - x1_lo, 1e-12)
    ty = (u2 - x2_lo) / np.maximum(x2_hi - x2_lo, 1e-12)

    q11 = values[i, j]
    q21 = values[i + 1, j]
    q12 = values[i, j + 1]
    q22 = values[i + 1, j + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * q11
        + tx * (1.0 - ty) * q21
        + (1.0 - tx) * ty * q12
        + tx * ty * q22
    )


class PyBNNSimToRealPlant:
    """Sim-to-real plant based on Moorman `three_node_xor` steady-state simulation."""

    def __init__(
        self,
        *,
        formulation: str = "moorman",
        model: str = "three_node_xor",
        backend: str = "jax",
        t_final: float | None = 30.0,
        dt: float | None = 0.005,
        noise: ObservationNoise | None = None,
        structured_residual: StructuredResidualMismatch | None = None,
        param_drift: ParamDriftMismatch | None = None,
    ) -> None:
        self.formulation = str(formulation)
        self.model = str(model)
        self.backend = str(backend)
        self.t_final = None if t_final is None else float(t_final)
        self.dt = None if dt is None else float(dt)
        self.noise = ObservationNoise(0.0) if noise is None else noise
        self.structured_residual = structured_residual
        self.param_drift = param_drift

    def _make_circuit(self, params: Any) -> Circuit:
        if self.backend not in {"jax", "numpy"}:
            raise ValueError(
                f"Unsupported backend {self.backend!r}; expected 'jax' or 'numpy'."
            )
        return pybnn.create_circuit(
            self.formulation,
            self.model,
            backend=self.backend,  # type: ignore[arg-type]
            params=params,
        )

    def reset(self, seed: int) -> SimToRealThetaState:
        rng = np.random.default_rng(int(seed))
        base_circuit = self._make_circuit(params=None)
        sim_params = getattr(base_circuit, "params", None)
        if sim_params is None:
            raise RuntimeError("PyBNN circuit does not expose constrained params.")

        real_params = sim_params
        if is_dataclass(sim_params):
            updates: dict[str, Any] = {}
            for f in fields(sim_params):
                updates[f.name] = _jitter_numeric_value(
                    getattr(sim_params, f.name), rng
                )
            real_params = replace(sim_params, **updates)
        return SimToRealThetaState(sim_params=sim_params, real_params=real_params)

    def simulate(self, program: Program, theta: SimToRealThetaState) -> float:
        x = program.as_constant_inputs()
        y = _steady_state_output_batch(
            formulation=self.formulation,
            model=self.model,
            backend=self.backend,
            params=theta.sim_params,
            U=x[None, :],
            t_final=self.t_final,
            dt=self.dt,
        )
        return float(y[0])

    def observe(self, program: Program, theta: SimToRealThetaState, rng: Any) -> float:
        if rng is None:
            rng = np.random.default_rng(0)
        x = program.as_constant_inputs()
        y = float(
            _steady_state_output_batch(
                formulation=self.formulation,
                model=self.model,
                backend=self.backend,
                params=theta.real_params,
                U=x[None, :],
                t_final=self.t_final,
                dt=self.dt,
            )[0]
        )
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
                    real_params,
                    level=level,
                    dt=step_dt,
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
        self,
        theta: SimToRealThetaState,
        grid_spec: GridSpec,
    ) -> Heatmap:
        x1g, x2g = np.meshgrid(
            np.asarray(grid_spec.x1, dtype=float),
            np.asarray(grid_spec.x2, dtype=float),
            indexing="ij",
        )
        U = np.stack([x1g.reshape(-1), x2g.reshape(-1)], axis=-1)
        y_flat = _steady_state_output_batch(
            formulation=self.formulation,
            model=self.model,
            backend=self.backend,
            params=theta.sim_params,
            U=U,
            t_final=self.t_final,
            dt=self.dt,
        )
        y = np.asarray(y_flat, dtype=float).reshape(
            len(grid_spec.x1), len(grid_spec.x2)
        )
        return Heatmap(grid=grid_spec, y=y)


_XOR_SURFACE_CACHE: dict[
    tuple[int, str, float | None, float | None],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}


@dataclass(frozen=True)
class MoormanXORSurface:
    """Cached dense `three_node_xor` steady-state surface with bilinear sampling."""

    x1: np.ndarray
    x2: np.ndarray
    y: np.ndarray

    @staticmethod
    def build(
        *,
        grid_n: int,
        t_final: float | None,
        dt: float | None,
        backend: str = "jax",
    ) -> "MoormanXORSurface":
        key = (int(grid_n), str(backend), t_final, dt)
        cached = _XOR_SURFACE_CACHE.get(key)
        if cached is not None:
            x1, x2, y = cached
            return MoormanXORSurface(x1=x1, x2=x2, y=y)

        x1 = np.linspace(0.0, 1.0, int(grid_n), dtype=float)
        x2 = np.linspace(0.0, 1.0, int(grid_n), dtype=float)
        x1g, x2g = np.meshgrid(x1, x2, indexing="ij")
        U = np.stack([x1g.reshape(-1), x2g.reshape(-1)], axis=-1)
        y_flat = _steady_state_output_batch(
            formulation="moorman",
            model="three_node_xor",
            backend=str(backend),
            params=None,
            U=U,
            t_final=t_final,
            dt=dt,
        )
        y = np.asarray(y_flat, dtype=float).reshape(len(x1), len(x2))

        _XOR_SURFACE_CACHE[key] = (x1, x2, y)
        return MoormanXORSurface(x1=x1, x2=x2, y=y)

    def evaluate(self, U: np.ndarray) -> np.ndarray:
        return np.asarray(
            _bilinear_interp(U, x1=self.x1, x2=self.x2, values=self.y),
            dtype=float,
        )


@dataclass
class MoormanXORThetaState:
    """Editable theta state for SPARC-style vector edits."""

    theta: np.ndarray
    time: float = 0.0
    exposure_level: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class MoormanXORFunctionPlant(PyBNNSimToRealPlant):
    """Compatibility wrapper using 4D input warp + three-node XOR steady state."""

    def __init__(
        self,
        *,
        backend: str = "jax",
        t_final: float | None = 30.0,
        dt: float | None = 0.005,
        surface: MoormanXORSurface | None = None,
        theta0: np.ndarray | None = None,
        theta_low: np.ndarray | None = None,
        theta_high: np.ndarray | None = None,
        real_a: float = 1.0,
        real_b: float = 0.0,
        real_s: np.ndarray | None = None,
        real_t: np.ndarray | None = None,
        noise: ObservationNoise | None = None,
        structured_residual: StructuredResidualMismatch | None = None,
        param_drift: ParamDriftMismatch | None = None,
    ) -> None:
        super().__init__(
            formulation="moorman",
            model="three_node_xor",
            backend=backend,
            t_final=t_final,
            dt=dt,
            noise=noise,
            structured_residual=structured_residual,
            param_drift=param_drift,
        )
        self._surface = (
            surface
            if surface is not None
            else MoormanXORSurface.build(
                grid_n=41,
                t_final=self.t_final,
                dt=self.dt,
                backend=self.backend,
            )
        )
        self._theta0 = (
            np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
            if theta0 is None
            else np.asarray(theta0, dtype=float)
        )
        self._theta_low = (
            np.array([0.7, -0.2, 0.7, -0.2], dtype=float)
            if theta_low is None
            else np.asarray(theta_low, dtype=float)
        )
        self._theta_high = (
            np.array([1.3, 0.2, 1.3, 0.2], dtype=float)
            if theta_high is None
            else np.asarray(theta_high, dtype=float)
        )
        self._real_a = float(real_a)
        self._real_b = float(real_b)
        self._real_s = (
            np.ones((2,), dtype=float)
            if real_s is None
            else np.asarray(real_s, dtype=float)
        )
        self._real_t = (
            np.zeros((2,), dtype=float)
            if real_t is None
            else np.asarray(real_t, dtype=float)
        )
        if self._theta0.shape != (4,):
            raise ValueError(f"theta0 must have shape (4,), got {self._theta0.shape}.")
        if self._theta_low.shape != (4,) or self._theta_high.shape != (4,):
            raise ValueError("theta_low/theta_high must have shape (4,).")
        if self._real_s.shape != (2,) or self._real_t.shape != (2,):
            raise ValueError("real_s/real_t must have shape (2,).")

    def _project_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (4,):
            raise ValueError(f"Expected theta shape (4,), got {theta.shape}.")
        return np.clip(theta, self._theta_low, self._theta_high)

    def simulator_y_batch(self, U: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta = self._project_theta(theta)
        Uw = _warp_by_theta(U, theta)
        return self._surface.evaluate(Uw)

    def reset(self, seed: int) -> MoormanXORThetaState:
        del seed
        return MoormanXORThetaState(theta=self._project_theta(self._theta0))

    def simulate(self, program: Program, theta: MoormanXORThetaState) -> float:
        u = program.as_constant_inputs()
        return float(self.simulator_y_batch(u[None, :], theta.theta)[0])

    def real_mean_batch(self, U: np.ndarray, *, theta: np.ndarray) -> np.ndarray:
        U = _as_u_batch(U)
        theta = self._project_theta(theta)
        U_sim = _warp_by_theta(U, theta)
        U_real = np.empty_like(U_sim, dtype=float)
        U_real[:, 0] = self._real_s[0] * U_sim[:, 0] + self._real_t[0]
        U_real[:, 1] = self._real_s[1] * U_sim[:, 1] + self._real_t[1]
        U_real = np.clip(U_real, 0.0, 1.0)
        y = self._real_a * self._surface.evaluate(U_real) + self._real_b
        if self.structured_residual is not None:
            y = y + np.array(
                [self.structured_residual.residual(u) for u in U],
                dtype=float,
            )
        return np.asarray(y, dtype=float)

    def observe(self, program: Program, theta: MoormanXORThetaState, rng: Any) -> float:
        if rng is None:
            rng = np.random.default_rng(0)
        u = program.as_constant_inputs()
        y = float(self.real_mean_batch(u[None, :], theta=theta.theta)[0])
        y, meta = self.noise.apply(y, rng=rng)
        program.meta.setdefault("noise_meta", dict(meta))
        return float(y)

    def apply_intervention(
        self,
        theta: MoormanXORThetaState,
        intervention: Intervention,
        dt: float | None,
    ) -> MoormanXORThetaState:
        if intervention.kind == "theta_edit":
            payload = intervention.payload or {}
            th = np.asarray(theta.theta, dtype=float)
            if "set" in payload and payload["set"] is not None:
                s = dict(payload["set"])
                if "theta" in s:
                    th = np.asarray(s["theta"], dtype=float)
            if "delta" in payload and payload["delta"] is not None:
                d = dict(payload["delta"])
                if "theta" in d:
                    th = th + np.asarray(d["theta"], dtype=float)
            th = self._project_theta(th)
            return MoormanXORThetaState(
                theta=th,
                time=float(theta.time),
                exposure_level=float(theta.exposure_level),
                meta=dict(theta.meta),
            )

        if intervention.kind == "exposure_schedule":
            payload = intervention.payload or {}
            level = float(payload.get("level", theta.exposure_level))
            duration = float(payload.get("duration", 0.0))
            step_dt = float(duration if dt is None else dt)
            return MoormanXORThetaState(
                theta=np.asarray(theta.theta, dtype=float),
                time=float(theta.time + step_dt),
                exposure_level=level,
                meta=dict(theta.meta),
            )

        raise ValueError(f"Unknown intervention kind: {intervention.kind!r}")

    def evaluate_heatmap(
        self, theta: MoormanXORThetaState, grid_spec: GridSpec
    ) -> Heatmap:
        y = np.zeros((len(grid_spec.x1), len(grid_spec.x2)), dtype=float)
        for i, x1 in enumerate(grid_spec.x1):
            for j, x2 in enumerate(grid_spec.x2):
                y[i, j] = float(
                    self.simulator_y_batch(
                        np.array([[x1, x2]], dtype=float),
                        theta.theta,
                    )[0]
                )
        return Heatmap(grid=grid_spec, y=y)
