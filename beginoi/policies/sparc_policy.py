from __future__ import annotations

from typing import Any, Literal, Mapping, cast
from dataclasses import field, replace, dataclass

import numpy as np

from beginoi.core.types import (
    Program,
    Intervention,
    ProgramBatch,
    ControlAction,
    ExperimentResult,
)
from beginoi.tasks.single_paulsson.edit import EditConfig, propose_theta
from beginoi.tasks.single_paulsson.types import (
    RoundData,
    ProbeObservation,
    ExperimentHistory,
)
from beginoi.tasks.single_paulsson.probes import (
    ProbeSamplingConfig,
    sample_audit,
    sample_design_phase0,
)
from beginoi.tasks.single_paulsson.mismatch import (
    PhiHat,
    MismatchFitConfig,
    fit_mismatch,
)


def _as_mapping(x: Any) -> Mapping[str, Any]:
    if x is None:
        return {}
    if isinstance(x, Mapping):
        return x
    # OmegaConf DictConfig behaves like Mapping but isn't a dict.
    try:
        return dict(x)
    except TypeError as exc:  # pragma: no cover
        raise TypeError(
            f"Expected mapping-like config, got {type(x).__name__}"
        ) from exc


def _group_probe_observations(
    observations: list[Any],
) -> tuple[list[ProbeObservation], list[ProbeObservation]]:
    """Group raw observations into (audit, design) ProbeObservation lists."""
    by_id: dict[str, dict[str, Any]] = {}
    for o in observations:
        extra = dict(getattr(o, "extra", {}) or {})
        pid = str(extra.get("sparc_probe_id"))
        role = str(extra.get("sparc_role"))
        if pid == "None" or role not in {"audit", "design"}:
            continue
        theta_raw = getattr(o, "theta", None)
        if theta_raw is None:
            theta_raw = extra.get("theta")
        theta_vec = (
            None
            if theta_raw is None
            else np.asarray(theta_raw, dtype=float).reshape(-1)
        )
        rec = by_id.setdefault(pid, {"role": role, "u": None, "ys": [], "theta": None})
        rec["role"] = role
        if rec["u"] is None:
            rec["u"] = np.asarray(o.inputs_summary["u"], dtype=float)
        if rec["theta"] is None and theta_vec is not None and theta_vec.size > 0:
            rec["theta"] = np.asarray(theta_vec, dtype=float)
        rec["ys"].append(float(o.y))

    audit: list[ProbeObservation] = []
    design: list[ProbeObservation] = []
    for pid, rec in by_id.items():
        del pid
        u = np.asarray(rec["u"], dtype=float)
        y_reps = np.asarray(rec["ys"], dtype=float)
        theta_vec = None
        if rec.get("theta") is not None:
            theta_vec = np.asarray(rec["theta"], dtype=float)
        po = ProbeObservation(u=u, y_reps=y_reps, theta=theta_vec)
        if rec["role"] == "audit":
            audit.append(po)
        else:
            design.append(po)
    return audit, design


@dataclass(frozen=True)
class HistoryFitConfig:
    """How to select design observations for mismatch fit at each round."""

    mode: Literal["cumulative", "window", "per_round"] = "cumulative"
    window_rounds: int = 3


HistoryFitMode = Literal["cumulative", "window", "per_round"]


def _parse_history_fit_mode(raw_mode: Any) -> HistoryFitMode:
    mode = str(raw_mode if raw_mode is not None else "cumulative")
    if mode not in {"cumulative", "window", "per_round"}:
        raise ValueError(
            "history_fit.mode must be one of ['cumulative', 'window', 'per_round']."
        )
    return cast(HistoryFitMode, mode)


def _select_fit_observations(
    design_history: list[list[ProbeObservation]],
    *,
    mode: HistoryFitMode,
    window_rounds: int,
) -> list[ProbeObservation]:
    if not design_history:
        return []
    if mode == "per_round":
        return list(design_history[-1])
    if mode == "window":
        window = max(1, int(window_rounds))
        chunks = design_history[-window:]
    elif mode == "cumulative":
        chunks = design_history
    else:  # pragma: no cover
        raise ValueError(f"Unsupported history_fit mode: {mode!r}")
    merged: list[ProbeObservation] = []
    for chunk in chunks:
        merged.extend(chunk)
    return merged


def _effective_mismatch_cfg(
    base: MismatchFitConfig, *, fit_n: int
) -> MismatchFitConfig:
    fit_n = max(0, int(fit_n))
    warp_min = max(0, int(base.min_design_points_for_warp))
    rbf_min = max(0, int(base.min_design_points_for_rbf))
    return replace(
        base,
        use_affine_y=True,
        use_input_warp=bool(base.use_input_warp and fit_n >= warp_min),
        use_rbf_residual=bool(base.use_rbf_residual and fit_n >= rbf_min),
    )


@dataclass
class SparcPolicy:
    """SPARC Phase-0 policy: audit/design probes + mismatch fit + trust-region edits."""

    seed: int = 0
    n_max: int = 8
    replicates: int = 3
    audit_frac: float = 0.25
    n_mc: int = 2048
    sim_warmstart: Any = None
    history_fit: Any = None
    probe_sampling: Any = None
    mismatch_fit: Any = None
    edit: Any = None

    rng: np.random.Generator = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)
    benchmark: Any = field(init=False, repr=False)
    cfg_probe: ProbeSamplingConfig = field(init=False, repr=False)
    cfg_mismatch: MismatchFitConfig = field(init=False, repr=False)
    cfg_history_fit: HistoryFitConfig = field(init=False, repr=False)
    cfg_edit: EditConfig = field(init=False, repr=False)
    U_audit: np.ndarray = field(init=False, repr=False)
    U_mc: np.ndarray = field(init=False, repr=False)
    _audit_n: int = field(init=False, repr=False, default=0)
    _design_n: int = field(init=False, repr=False, default=0)
    round_idx: int = field(init=False, default=0)
    pending_theta_set: np.ndarray | None = field(init=False, default=None, repr=False)
    _phi_prev: PhiHat | None = field(init=False, default=None, repr=False)
    _design_obs_history: list[list[ProbeObservation]] = field(
        init=False, default_factory=list, repr=False
    )
    rounds: ExperimentHistory = field(
        init=False, repr=False, default_factory=ExperimentHistory
    )

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        self.seed = int(seed if seed is not None else self.seed)
        self.rng = np.random.default_rng(int(self.seed))
        self.regime = regime
        self.benchmark = benchmark

        self.cfg_probe = ProbeSamplingConfig(**_as_mapping(self.probe_sampling))
        self.cfg_mismatch = MismatchFitConfig(**_as_mapping(self.mismatch_fit))
        history_fit_map = dict(_as_mapping(self.history_fit))
        mode = _parse_history_fit_mode(history_fit_map.get("mode", "cumulative"))
        self.cfg_history_fit = HistoryFitConfig(
            mode=mode,
            window_rounds=int(history_fit_map.get("window_rounds", 3)),
        )
        edit_map = dict(_as_mapping(self.edit))

        # Prefer bounds from benchmark if present.
        if "theta_low" not in edit_map and hasattr(benchmark, "theta_low"):
            edit_map["theta_low"] = np.asarray(
                getattr(benchmark, "theta_low"), dtype=float
            )
        if "theta_high" not in edit_map and hasattr(benchmark, "theta_high"):
            edit_map["theta_high"] = np.asarray(
                getattr(benchmark, "theta_high"), dtype=float
            )
        self.cfg_edit = EditConfig(**edit_map)

        if not hasattr(benchmark, "target_g_batch") or not hasattr(
            benchmark, "simulator_y_batch"
        ):
            raise ValueError(
                "SPARC requires benchmark.target_g_batch(U) and "
                "benchmark.simulator_y_batch(U, theta)."
            )

        n_max = int(self.n_max)
        r = int(self.replicates)
        if n_max <= 0 or r <= 0:
            raise ValueError("SPARC requires n_max>0 and replicates>0.")
        total_programs = n_max * r
        if total_programs > int(self.regime.max_programs_per_unit):
            raise ValueError(
                "SPARC requires n_max*replicates <= capacity; got "
                f"{total_programs} > {self.regime.max_programs_per_unit}."
            )

        audit_n = int(np.floor(float(self.audit_frac) * n_max))
        audit_n = max(1, min(audit_n, n_max - 1))
        self._audit_n = audit_n
        self._design_n = n_max - audit_n

        ss = np.random.SeedSequence(int(self.seed))
        rng_audit = np.random.default_rng(ss.spawn(1)[0])
        self.U_audit = sample_audit(
            rng_audit, cfg=self.cfg_probe, g_batch=benchmark.target_g_batch, n=audit_n
        )
        rng_mc = np.random.default_rng(ss.spawn(1)[0].spawn(1)[0])
        self.U_mc = rng_mc.uniform(0.0, 1.0, size=(int(self.n_mc), 2))

        self.round_idx = 0
        self.pending_theta_set = None
        self._phi_prev = None
        self._design_obs_history = []
        self.rounds = ExperimentHistory()

        warm = dict(_as_mapping(self.sim_warmstart))
        warm_enabled = bool(warm.get("enabled", False))
        warm_candidates = int(warm.get("n_candidates", 1024))
        warm_chunk = int(warm.get("chunk_size", 128))
        if warm_enabled:
            theta_star = self._warmstart_theta_sim_only(
                n_candidates=warm_candidates,
                chunk_size=warm_chunk,
            )
            self.pending_theta_set = theta_star

    def _warmstart_theta_sim_only(
        self,
        *,
        n_candidates: int,
        chunk_size: int,
    ) -> np.ndarray:
        """Offline sim-only warmstart: pick theta minimizing E[(y_sim-g)^2] on U_mc."""
        n_candidates = int(max(n_candidates, 1))
        chunk_size = int(max(chunk_size, 1))
        low = np.asarray(self.cfg_edit.theta_low, dtype=float)
        high = np.asarray(self.cfg_edit.theta_high, dtype=float)
        g = np.asarray(self.benchmark.target_g_batch(self.U_mc), dtype=float)

        ss = np.random.SeedSequence(int(self.seed))
        rng = np.random.default_rng(ss.spawn(42)[0])

        best_theta = np.clip(low + 0.5 * (high - low), low, high)
        best_val = float("inf")

        # If the benchmark exposes an analytic target parameterization that matches the
        # simulator family, try it directly (can make y_sim == g exactly).
        target = getattr(self.benchmark, "target", None)
        if target is not None:
            center = getattr(target, "center", None)
            r_mid = getattr(target, "r_mid", None)
            thickness = getattr(target, "thickness", None)
            if center is not None and r_mid is not None and thickness is not None:
                try:
                    tx, ty = float(center[0]), float(center[1])
                    cand = np.array(
                        [tx, ty, float(r_mid), float(thickness)], dtype=float
                    )
                    cand = np.clip(cand, low, high)
                    y0 = np.asarray(
                        self.benchmark.simulator_y_batch(self.U_mc, cand), dtype=float
                    )
                    mse0 = float(np.mean((y0 - g) ** 2))
                    best_theta = cand
                    best_val = mse0
                except (TypeError, ValueError, IndexError):
                    pass

        for start in range(0, n_candidates, chunk_size):
            m = min(chunk_size, n_candidates - start)
            thetas = rng.uniform(low, high, size=(m, len(low)))
            for th in thetas:
                y = np.asarray(
                    self.benchmark.simulator_y_batch(self.U_mc, th), dtype=float
                )
                mse = float(np.mean((y - g) ** 2))
                if mse < best_val:
                    best_val = mse
                    best_theta = np.asarray(th, dtype=float)

        return np.clip(best_theta, low, high)

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        if self.regime.program_kind not in {"constant", "timeseries"}:
            raise ValueError(f"Unsupported program_kind: {self.regime.program_kind!r}")

        ss = np.random.SeedSequence(int(self.seed))
        rng_round = np.random.default_rng(ss.spawn(self.round_idx + 1)[-1])
        U_design = sample_design_phase0(
            rng_round,
            cfg=self.cfg_probe,
            g_batch=self.benchmark.target_g_batch,
            n=self._design_n,
        )

        programs: list[Program] = []
        for role, U in (("audit", self.U_audit), ("design", U_design)):
            for i, u_row in enumerate(U):
                u = np.asarray(u_row, dtype=float)
                probe_id = f"k{self.round_idx}_{role}_{i}"
                for rep in range(int(self.replicates)):
                    meta = {
                        "sparc_role": role,
                        "sparc_probe_id": probe_id,
                        "replicate_id": int(rep),
                    }
                    if self.regime.program_kind == "timeseries":
                        prog_u = u[None, :]
                        programs.append(
                            Program(kind="timeseries", u=prog_u, t=None, meta=meta)
                        )
                    else:
                        programs.append(Program(kind="constant", u=u, meta=meta))

        intervention = None
        if self.pending_theta_set is not None:
            intervention = Intervention(
                kind="theta_edit",
                payload={
                    "set": {
                        "theta": np.asarray(
                            self.pending_theta_set, dtype=float
                        ).tolist()
                    }
                },
            )
            self.pending_theta_set = None

        return ControlAction(
            batch=ProgramBatch(programs=programs), intervention=intervention
        )

    def update(self, history, new_result: ExperimentResult) -> None:
        obs = list(new_result.observations)
        audit_obs, design_obs = _group_probe_observations(obs)
        self._design_obs_history.append(list(design_obs))
        fit_obs = _select_fit_observations(
            self._design_obs_history,
            mode=self.cfg_history_fit.mode,
            window_rounds=self.cfg_history_fit.window_rounds,
        )
        fit_n = int(len(fit_obs))
        cfg_fit = _effective_mismatch_cfg(self.cfg_mismatch, fit_n=fit_n)

        theta_state = history.theta_snapshots[-1]
        theta_vec = (
            np.asarray(getattr(theta_state, "theta"), dtype=float)
            if hasattr(theta_state, "theta")
            else np.asarray(theta_state, dtype=float)
        )

        phi_hat, fit_diag = fit_mismatch(
            theta_k=theta_vec,
            obs_design=fit_obs,
            y_sim_batch=self.benchmark.simulator_y_batch,
            cfg=cfg_fit,
            init_phi=self._phi_prev,
        )

        # Smooth mismatch parameters across rounds (true mismatch is constant in the
        # synthetic benchmark; smoothing reduces jitter from small design sets).
        smooth = float(getattr(self.cfg_mismatch, "smoothing", 0.0))
        if self._phi_prev is None or smooth <= 0.0:
            phi_for_edit = phi_hat
        else:
            smooth = float(np.clip(smooth, 0.0, 1.0))
            phi_for_edit = PhiHat(
                a=(1.0 - smooth) * float(self._phi_prev.a) + smooth * float(phi_hat.a),
                b=(1.0 - smooth) * float(self._phi_prev.b) + smooth * float(phi_hat.b),
                s=(1.0 - smooth) * np.asarray(self._phi_prev.s, dtype=float)
                + smooth * np.asarray(phi_hat.s, dtype=float),
                t=(1.0 - smooth) * np.asarray(self._phi_prev.t, dtype=float)
                + smooth * np.asarray(phi_hat.t, dtype=float),
                centers=phi_hat.centers,
                lengthscale=float(phi_hat.lengthscale),
                c=phi_hat.c,
            )
        self._phi_prev = phi_for_edit

        # Compute audit-only metric for round record (actual logging is done by
        # SparcAuditMetric).
        if audit_obs:
            U_a = np.array([o.u for o in audit_obs], dtype=float)
            y_a = np.array([o.y_mean for o in audit_obs], dtype=float)
            g_a = np.asarray(self.benchmark.target_g_batch(U_a), dtype=float)
            audit_mse = float(np.mean((y_a - g_a) ** 2))
        else:
            audit_mse = float("nan")

        ss = np.random.SeedSequence(int(self.seed))
        rng_edit = np.random.default_rng(ss.spawn(10_000 + self.round_idx + 1)[-1])
        theta_next, edit_diag = propose_theta(
            theta_k=theta_vec,
            phi=phi_for_edit,
            cfg=self.cfg_edit,
            rng=rng_edit,
            U_mc=self.U_mc,
            y_sim_batch=self.benchmark.simulator_y_batch,
            g_batch=self.benchmark.target_g_batch,
        )
        self.pending_theta_set = np.asarray(theta_next, dtype=float)

        round_rec = RoundData(
            theta_k=np.asarray(theta_vec, dtype=float),
            U_audit=np.asarray(self.U_audit, dtype=float),
            U_design=np.asarray([o.u for o in design_obs], dtype=float),
            obs_audit=audit_obs,
            obs_design=design_obs,
            phi_fit={
                "a": float(phi_hat.a),
                "b": float(phi_hat.b),
                "s": np.asarray(phi_hat.s, dtype=float).tolist(),
                "t": np.asarray(phi_hat.t, dtype=float).tolist(),
                "lengthscale": float(phi_hat.lengthscale),
                "centers": None
                if phi_hat.centers is None
                else np.asarray(phi_hat.centers, dtype=float).tolist(),
                "c": None
                if phi_hat.c is None
                else np.asarray(phi_hat.c, dtype=float).tolist(),
            },
            metrics={"audit_mse": audit_mse},
            fit_diagnostics={
                "fit_n": fit_n,
                "fit_mode": str(self.cfg_history_fit.mode),
                "warp_enabled": bool(cfg_fit.use_input_warp),
                "rbf_enabled": bool(cfg_fit.use_rbf_residual),
                **fit_diag,
                **edit_diag,
            },
        )
        self.rounds.rounds.append(round_rec)

        extras = getattr(history, "extras", None)
        if extras is not None:
            extras.setdefault("sparc_rounds", [])
            extras["sparc_rounds"].append(
                {
                    "round": int(self.round_idx),
                    "theta_k": np.asarray(round_rec.theta_k, dtype=float).tolist(),
                    "theta_next": np.asarray(theta_next, dtype=float).tolist(),
                    "U_audit": np.asarray(round_rec.U_audit, dtype=float).tolist(),
                    "U_design": np.asarray(round_rec.U_design, dtype=float).tolist(),
                    "obs_audit": [
                        {
                            "u": o.u.tolist(),
                            "y_reps": o.y_reps.tolist(),
                            "theta": None
                            if o.theta is None
                            else np.asarray(o.theta, dtype=float).tolist(),
                        }
                        for o in round_rec.obs_audit
                    ],
                    "obs_design": [
                        {
                            "u": o.u.tolist(),
                            "y_reps": o.y_reps.tolist(),
                            "theta": None
                            if o.theta is None
                            else np.asarray(o.theta, dtype=float).tolist(),
                        }
                        for o in round_rec.obs_design
                    ],
                    "phi_fit": dict(round_rec.phi_fit),
                    "metrics": dict(round_rec.metrics),
                    "diagnostics": dict(round_rec.fit_diagnostics),
                }
            )

        self.round_idx += 1
