from __future__ import annotations

from typing import Any, Mapping
from dataclasses import field, dataclass

import numpy as np

from beginoi.core.types import (
    Program,
    Intervention,
    ProgramBatch,
    ControlAction,
    ExperimentResult,
)
from beginoi.tasks.single_paulsson.types import ProbeObservation
from beginoi.tasks.single_paulsson.probes import (
    PhoenixProbeConfig,
    rotate_pool_slice,
    sample_design_phoenix,
    build_post_pool_mixture,
    build_audit_pool_mixture,
)
from beginoi.tasks.single_paulsson.phoenix_edit import (
    PhoenixEditConfig,
    solve_theta_one_shot,
    select_alpha_predicted,
)
from beginoi.tasks.single_paulsson.phoenix_mismatch import (
    PhoenixPhi,
    PhoenixMismatchConfig,
    y_hat_batch,
    fit_mismatch,
)


def _as_mapping(x: Any) -> Mapping[str, Any]:
    if x is None:
        return {}
    if isinstance(x, Mapping):
        return x
    try:
        return dict(x)
    except TypeError as exc:  # pragma: no cover
        raise TypeError(
            f"Expected mapping-like config, got {type(x).__name__}"
        ) from exc


@dataclass(frozen=True)
class PhoenixStopConfig:
    k_max: int = 30
    patience: int = 5
    tol: float = 1e-4
    min_pred_improvement: float = 1e-4
    post_gain_min: float = 0.0
    noise_floor_factor: float = 1.2


@dataclass(frozen=True)
class PhoenixStabilityConfig:
    eps_alpha: float = 0.01
    eps_b: float = 0.01
    eps_s: float = 0.02
    eps_t: float = 0.02
    rounds_required: int = 2


@dataclass(frozen=True)
class PhoenixPostStepConfig:
    enabled: bool = True
    n_post: int = 8
    fit_include_post_step: bool = False
    overshoot_kappa: float = 2.0
    damping_mode: str = "drop_alpha_one"  # or "scale_grid"
    damping_scale: float = 0.5
    damping_rounds: int = 2


def _group_probe_observations(
    observations: list[Any],
) -> tuple[list[ProbeObservation], list[ProbeObservation], list[ProbeObservation]]:
    """Group raw observations into (audit, design, post-step) ProbeObservation lists."""
    by_id: dict[str, dict[str, Any]] = {}
    for o in observations:
        extra = dict(getattr(o, "extra", {}) or {})
        role_raw = extra.get("phoenix_role", extra.get("sparc_role"))
        role = str(role_raw)
        probe_id_raw = extra.get("phoenix_probe_id", extra.get("sparc_probe_id"))
        pid = str(probe_id_raw)
        if pid == "None" or role not in {"audit", "design", "post_step"}:
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
    post_step: list[ProbeObservation] = []
    for rec in by_id.values():
        po = ProbeObservation(
            u=np.asarray(rec["u"], dtype=float),
            y_reps=np.asarray(rec["ys"], dtype=float),
            theta=None
            if rec["theta"] is None
            else np.asarray(rec["theta"], dtype=float),
        )
        if rec["role"] == "audit":
            audit.append(po)
        elif rec["role"] == "design":
            design.append(po)
        else:
            post_step.append(po)
    return audit, design, post_step


@dataclass
class PhoenixPolicy:
    """PHOENIX policy: probe-hard design + mismatch correction + one-shot edits."""

    seed: int = 0
    n_max: int = 64
    replicates: int = 2
    audit_frac: float = 0.25
    n_mc: int = 1024
    probe_sampling: Any = None
    mismatch_fit: Any = None
    edit: Any = None
    post_step: Any = None
    stop: Any = None
    stability: Any = None

    rng: np.random.Generator = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)
    benchmark: Any = field(init=False, repr=False)
    cfg_probe: PhoenixProbeConfig = field(init=False, repr=False)
    cfg_mismatch: PhoenixMismatchConfig = field(init=False, repr=False)
    cfg_edit: PhoenixEditConfig = field(init=False, repr=False)
    cfg_post: PhoenixPostStepConfig = field(init=False, repr=False)
    cfg_stop: PhoenixStopConfig = field(init=False, repr=False)
    cfg_stability: PhoenixStabilityConfig = field(init=False, repr=False)
    U_audit_pool: np.ndarray = field(init=False, repr=False)
    U_post_pool: np.ndarray = field(init=False, repr=False)
    U_mc: np.ndarray = field(init=False, repr=False)
    round_idx: int = field(init=False, default=0)
    pending_theta_set: np.ndarray | None = field(init=False, default=None, repr=False)
    _scheduled_post_eval: dict[str, Any] | None = field(
        init=False, default=None, repr=False
    )
    _audit_cursor: int = field(init=False, default=0, repr=False)
    _post_cursor: int = field(init=False, default=0, repr=False)
    _fit_dataset_design: list[ProbeObservation] = field(
        init=False, default_factory=list, repr=False
    )
    _fit_dataset_post: list[ProbeObservation] = field(
        init=False, default_factory=list, repr=False
    )
    _phi_prev: PhoenixPhi | None = field(init=False, default=None, repr=False)
    _stable_counter: int = field(init=False, default=0, repr=False)
    _design_stable: bool = field(init=False, default=False, repr=False)
    _damping_rounds_left: int = field(init=False, default=0, repr=False)
    _should_stop: bool = field(init=False, default=False, repr=False)
    _stop_reasons: list[str] = field(init=False, default_factory=list, repr=False)
    _audit_history: list[float] = field(init=False, default_factory=list, repr=False)
    _last_post_real_loss: float = field(init=False, default=float("nan"), repr=False)
    _last_post_gain: float = field(init=False, default=float("nan"), repr=False)
    _last_pred_improvement: float = field(init=False, default=float("nan"), repr=False)
    _last_noise_floor: float = field(init=False, default=float("nan"), repr=False)
    _last_train_rmse: float = field(init=False, default=float("nan"), repr=False)

    def init(self, seed: int, *, benchmark: Any, regime: Any) -> None:
        self.seed = int(seed if seed is not None else self.seed)
        self.rng = np.random.default_rng(int(self.seed))
        self.benchmark = benchmark
        self.regime = regime

        self.cfg_probe = PhoenixProbeConfig(**_as_mapping(self.probe_sampling))
        self.cfg_mismatch = PhoenixMismatchConfig(**_as_mapping(self.mismatch_fit))
        self.cfg_post = PhoenixPostStepConfig(**_as_mapping(self.post_step))
        self.cfg_stop = PhoenixStopConfig(**_as_mapping(self.stop))
        self.cfg_stability = PhoenixStabilityConfig(**_as_mapping(self.stability))

        edit_map = dict(_as_mapping(self.edit))
        if "theta_low" not in edit_map and hasattr(benchmark, "theta_low"):
            edit_map["theta_low"] = np.asarray(
                getattr(benchmark, "theta_low"), dtype=float
            )
        if "theta_high" not in edit_map and hasattr(benchmark, "theta_high"):
            edit_map["theta_high"] = np.asarray(
                getattr(benchmark, "theta_high"), dtype=float
            )
        self.cfg_edit = PhoenixEditConfig(**edit_map)

        if not hasattr(benchmark, "target_g_batch") or not hasattr(
            benchmark, "simulator_y_batch"
        ):
            raise ValueError(
                "PHOENIX requires benchmark.target_g_batch(U) and "
                "benchmark.simulator_y_batch(U, theta)."
            )

        n_max = int(self.n_max)
        if n_max <= 0 or int(self.replicates) <= 0:
            raise ValueError("PHOENIX requires n_max>0 and replicates>0.")
        if n_max * int(self.replicates) > int(self.regime.max_programs_per_unit):
            raise ValueError(
                "PHOENIX requires n_max*replicates <= max_programs_per_unit; got "
                f"{n_max * int(self.replicates)} > {self.regime.max_programs_per_unit}."
            )

        ss = np.random.SeedSequence(int(self.seed))
        rng_audit = np.random.default_rng(ss.spawn(1)[0])
        self.U_audit_pool = build_audit_pool_mixture(
            rng_audit, cfg=self.cfg_probe, g_batch=self.benchmark.target_g_batch
        )
        rng_post = np.random.default_rng(ss.spawn(1)[0].spawn(1)[0])
        self.U_post_pool = build_post_pool_mixture(
            rng_post, cfg=self.cfg_probe, g_batch=self.benchmark.target_g_batch
        )
        rng_mc = np.random.default_rng(ss.spawn(1)[0].spawn(1)[0].spawn(1)[0])
        self.U_mc = rng_mc.uniform(0.0, 1.0, size=(int(self.n_mc), 2))

        self.round_idx = 0
        self.pending_theta_set = None
        self._scheduled_post_eval = None
        self._audit_cursor = 0
        self._post_cursor = 0
        self._fit_dataset_design = []
        self._fit_dataset_post = []
        self._phi_prev = None
        self._stable_counter = 0
        self._design_stable = False
        self._damping_rounds_left = 0
        self._should_stop = False
        self._stop_reasons = []
        self._audit_history = []
        self._last_post_real_loss = float("nan")
        self._last_post_gain = float("nan")
        self._last_pred_improvement = float("nan")
        self._last_noise_floor = float("nan")
        self._last_train_rmse = float("nan")

    def should_stop(self, history, *, budget_remaining: float) -> bool:
        del history, budget_remaining
        if int(self.round_idx) >= int(self.cfg_stop.k_max):
            return True
        return bool(self._should_stop)

    def _n_audit(self) -> int:
        n_max = int(self.n_max)
        return int(np.floor(float(self.audit_frac) * n_max))

    def _n_post(self) -> int:
        if not bool(self.cfg_post.enabled):
            return 0
        n_audit = self._n_audit()
        budget_left = max(0, int(self.n_max) - int(n_audit))
        return min(int(max(0, self.cfg_post.n_post)), budget_left)

    def act(self, history, *, budget_remaining: float) -> ControlAction:
        del history, budget_remaining
        if self.regime.program_kind not in {"constant", "timeseries"}:
            raise ValueError(
                f"Unsupported program_kind for PHOENIX: {self.regime.program_kind!r}"
            )

        n_audit = self._n_audit()
        U_audit, self._audit_cursor = rotate_pool_slice(
            self.U_audit_pool, start=self._audit_cursor, n=n_audit
        )

        U_post = np.zeros((0, 2), dtype=float)
        scheduled = self._scheduled_post_eval
        if scheduled is not None and int(scheduled.get("round_id", -1)) == int(
            self.round_idx
        ):
            U_post = np.asarray(scheduled.get("U_post"), dtype=float)

        n_design = max(0, int(self.n_max) - len(U_audit) - len(U_post))
        ss = np.random.SeedSequence(int(self.seed))
        rng_design = np.random.default_rng(ss.spawn(self.round_idx + 1)[-1])
        U_design = sample_design_phoenix(
            rng_design,
            cfg=self.cfg_probe,
            g_batch=self.benchmark.target_g_batch,
            n=n_design,
            stable=bool(self._design_stable),
        )

        programs: list[Program] = []
        probe_sets = [
            ("audit", U_audit),
            ("design", U_design),
            ("post_step", U_post),
        ]
        for role, U in probe_sets:
            for i, u_row in enumerate(np.asarray(U, dtype=float)):
                u = np.asarray(u_row, dtype=float)
                probe_id = f"k{self.round_idx}_{role}_{i}"
                for rep in range(int(self.replicates)):
                    meta = {
                        "phoenix_role": role,
                        "phoenix_probe_id": probe_id,
                        # Keep SPARC fields for compatibility with existing metric.
                        "sparc_role": role,
                        "sparc_probe_id": probe_id,
                        "replicate_id": int(rep),
                    }
                    if self.regime.program_kind == "timeseries":
                        programs.append(
                            Program(kind="timeseries", u=u[None, :], t=None, meta=meta)
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

    def _stability_update(self, phi: PhoenixPhi) -> None:
        if self._phi_prev is None:
            self._stable_counter = 0
            self._design_stable = False
            return
        d_alpha = abs(float(phi.alpha) - float(self._phi_prev.alpha))
        d_b = abs(float(phi.b) - float(self._phi_prev.b))
        d_s = float(np.max(np.abs(np.asarray(phi.s) - np.asarray(self._phi_prev.s))))
        d_t = float(np.max(np.abs(np.asarray(phi.t) - np.asarray(self._phi_prev.t))))
        stable_now = (
            d_alpha <= float(self.cfg_stability.eps_alpha)
            and d_b <= float(self.cfg_stability.eps_b)
            and d_s <= float(self.cfg_stability.eps_s)
            and d_t <= float(self.cfg_stability.eps_t)
        )
        if stable_now:
            self._stable_counter += 1
        else:
            self._stable_counter = 0
        self._design_stable = bool(
            self._stable_counter >= int(max(1, self.cfg_stability.rounds_required))
        )

    def _update_stop_flags(self) -> tuple[bool, dict[str, float | bool]]:
        reasons: list[str] = []
        if int(self.round_idx) + 1 >= int(self.cfg_stop.k_max):
            reasons.append("k_max")

        plateau = False
        patience = int(max(1, self.cfg_stop.patience))
        if len(self._audit_history) >= patience + 1:
            baseline = float(self._audit_history[-(patience + 1)])
            best_recent = float(np.min(np.asarray(self._audit_history[-patience:])))
            improvement = baseline - best_recent
            plateau = bool(improvement < float(self.cfg_stop.tol))
            if plateau:
                reasons.append("audit_plateau")

        pred_low = np.isfinite(self._last_pred_improvement) and (
            float(self._last_pred_improvement)
            < float(self.cfg_stop.min_pred_improvement)
        )
        post_not_improving = np.isfinite(self._last_post_gain) and (
            float(self._last_post_gain) <= float(self.cfg_stop.post_gain_min)
        )
        if pred_low and post_not_improving:
            reasons.append("low_pred_and_no_post_gain")

        fit_good = (
            np.isfinite(self._last_train_rmse)
            and np.isfinite(self._last_noise_floor)
            and float(self._last_train_rmse)
            <= float(self.cfg_stop.noise_floor_factor) * float(self._last_noise_floor)
        )
        if plateau and fit_good:
            reasons.append("plateau_and_noise_floor")

        self._stop_reasons = reasons
        self._should_stop = len(reasons) > 0
        return self._should_stop, {
            "stop_flag": bool(self._should_stop),
            "stop_reason_count": float(len(reasons)),
            "stop_plateau": bool(plateau),
            "fit_good": bool(fit_good),
            "pred_low": bool(pred_low),
            "post_not_improving": bool(post_not_improving),
        }

    def update(self, history, new_result: ExperimentResult) -> None:
        obs = list(new_result.observations)
        obs_audit, obs_design, obs_post = _group_probe_observations(obs)

        # Post-step validation compares predicted loss from prior round with realized loss
        # measured now under the already-applied edited theta.
        overshoot = False
        post_real_loss = float("nan")
        post_pred_loss = float("nan")
        post_sigma = float("nan")
        post_gain = float("nan")
        if obs_post:
            U_post_obs = np.asarray([o.u for o in obs_post], dtype=float)
            y_post = np.asarray([o.y_mean for o in obs_post], dtype=float)
            y_post_var = np.asarray([o.y_var for o in obs_post], dtype=float)
            g_post = np.asarray(self.benchmark.target_g_batch(U_post_obs), dtype=float)
            post_real_loss = float(np.mean((y_post - g_post) ** 2))
            post_sigma = float(
                np.sqrt(
                    np.mean(y_post_var + float(self.cfg_mismatch.var_epsilon))
                    / max(1, len(obs_post))
                )
            )
            if self._scheduled_post_eval is not None and int(
                self._scheduled_post_eval.get("round_id", -1)
            ) == int(self.round_idx):
                post_pred_loss = float(
                    self._scheduled_post_eval.get("pred_loss", np.nan)
                )
                if np.isfinite(post_pred_loss):
                    overshoot = bool(
                        post_real_loss
                        > post_pred_loss
                        + float(self.cfg_post.overshoot_kappa) * post_sigma
                    )
            if np.isfinite(self._last_post_real_loss):
                post_gain = float(self._last_post_real_loss - post_real_loss)
            self._last_post_real_loss = post_real_loss
            self._last_post_gain = post_gain

        if overshoot:
            self._damping_rounds_left = int(max(1, self.cfg_post.damping_rounds))

        self._fit_dataset_design.extend(obs_design)
        if bool(self.cfg_post.fit_include_post_step):
            self._fit_dataset_post.extend(obs_post)
        fit_obs = list(self._fit_dataset_design)
        fit_obs.extend(self._fit_dataset_post)

        theta_state = history.theta_snapshots[-1]
        theta_vec = (
            np.asarray(getattr(theta_state, "theta"), dtype=float)
            if hasattr(theta_state, "theta")
            else np.asarray(theta_state, dtype=float)
        )

        phi, fit_diag = fit_mismatch(
            obs_fit=fit_obs,
            y_sim_batch=self.benchmark.simulator_y_batch,
            cfg=self.cfg_mismatch,
            init_phi=self._phi_prev,
            theta_fallback=theta_vec,
        )
        self._stability_update(phi)
        self._phi_prev = phi

        # Diagnostics on current round design probes.
        design_round_rmse = float("nan")
        if obs_design:
            U_d = np.asarray([o.u for o in obs_design], dtype=float)
            y_d = np.asarray([o.y_mean for o in obs_design], dtype=float)
            yhat_d = y_hat_batch(
                U_d,
                theta=theta_vec,
                phi=phi,
                y_sim_batch=self.benchmark.simulator_y_batch,
            )
            design_round_rmse = float(np.sqrt(np.mean((yhat_d - y_d) ** 2)))

        audit_mse = float("nan")
        if obs_audit:
            U_a = np.asarray([o.u for o in obs_audit], dtype=float)
            y_a = np.asarray([o.y_mean for o in obs_audit], dtype=float)
            g_a = np.asarray(self.benchmark.target_g_batch(U_a), dtype=float)
            audit_mse = float(np.mean((y_a - g_a) ** 2))
            self._audit_history.append(audit_mse)

        theta_star, edit_diag = solve_theta_one_shot(
            theta_k=theta_vec,
            phi=phi,
            cfg=self.cfg_edit,
            U_mc=self.U_mc,
            y_sim_batch=self.benchmark.simulator_y_batch,
            g_batch=self.benchmark.target_g_batch,
        )

        n_post_next = self._n_post()
        U_post_next, self._post_cursor = rotate_pool_slice(
            self.U_post_pool, start=self._post_cursor, n=n_post_next
        )
        use_damping = self._damping_rounds_left > 0
        drop_alpha_one = bool(
            use_damping and str(self.cfg_post.damping_mode) == "drop_alpha_one"
        )
        alpha_scale = (
            float(self.cfg_post.damping_scale)
            if use_damping and str(self.cfg_post.damping_mode) == "scale_grid"
            else 1.0
        )
        theta_next, alpha_diag = select_alpha_predicted(
            theta_k=theta_vec,
            theta_star=theta_star,
            phi=phi,
            cfg=self.cfg_edit,
            U_post=U_post_next if len(U_post_next) > 0 else self.U_mc[:32],
            y_sim_batch=self.benchmark.simulator_y_batch,
            g_batch=self.benchmark.target_g_batch,
            alpha_scale=alpha_scale,
            drop_alpha_one=drop_alpha_one,
        )

        if use_damping:
            self._damping_rounds_left = max(0, self._damping_rounds_left - 1)

        self.pending_theta_set = np.asarray(theta_next, dtype=float)
        pred_loss_next = float(alpha_diag["post_pred_loss_chosen"])
        self._scheduled_post_eval = {
            "round_id": int(self.round_idx + 1),
            "U_post": np.asarray(U_post_next, dtype=float),
            "pred_loss": pred_loss_next,
            "theta_next": np.asarray(theta_next, dtype=float),
        }

        self._last_pred_improvement = float(edit_diag["pred_obj_mc_improvement"])
        self._last_train_rmse = float(fit_diag.get("train_rmse_weighted", np.nan))
        self._last_noise_floor = float(fit_diag.get("noise_floor_rmse", np.nan))
        stop_flag, stop_diag = self._update_stop_flags()

        extras = getattr(history, "extras", None)
        if extras is not None:
            extras.setdefault("phoenix_rounds", [])
            extras["phoenix_rounds"].append(
                {
                    "round": int(self.round_idx),
                    "theta_k": np.asarray(theta_vec, dtype=float).tolist(),
                    "theta_star": np.asarray(theta_star, dtype=float).tolist(),
                    "theta_next": np.asarray(theta_next, dtype=float).tolist(),
                    "U_audit_n": int(len(obs_audit)),
                    "U_design_n": int(len(obs_design)),
                    "U_post_n": int(len(obs_post)),
                    "audit_mse": float(audit_mse),
                    "design_round_rmse": float(design_round_rmse),
                    "phi_fit": {
                        "a": float(phi.a),
                        "alpha": float(phi.alpha),
                        "b": float(phi.b),
                        "s": np.asarray(phi.s, dtype=float).tolist(),
                        "t": np.asarray(phi.t, dtype=float).tolist(),
                    },
                    "fit_diag": dict(fit_diag),
                    "edit_diag": dict(edit_diag),
                    "alpha_diag": dict(alpha_diag),
                    "post_step_validation": {
                        "real_loss": float(post_real_loss),
                        "pred_loss": float(post_pred_loss),
                        "sigma": float(post_sigma),
                        "overshoot": bool(overshoot),
                        "post_gain": float(post_gain),
                    },
                    "damping_rounds_left": int(self._damping_rounds_left),
                    "design_stable": bool(self._design_stable),
                    "stop": {
                        "flag": bool(stop_flag),
                        "reasons": list(self._stop_reasons),
                        **stop_diag,
                    },
                }
            )

        self.round_idx += 1
