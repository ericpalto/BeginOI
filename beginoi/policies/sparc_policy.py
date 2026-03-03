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
from beginoi.tasks.single_paulsson.mismatch import MismatchFitConfig, fit_mismatch


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
        rec = by_id.setdefault(pid, {"role": role, "u": None, "ys": []})
        rec["role"] = role
        if rec["u"] is None:
            rec["u"] = np.asarray(o.inputs_summary["u"], dtype=float)
        rec["ys"].append(float(o.y))

    audit: list[ProbeObservation] = []
    design: list[ProbeObservation] = []
    for pid, rec in by_id.items():
        del pid
        u = np.asarray(rec["u"], dtype=float)
        y_reps = np.asarray(rec["ys"], dtype=float)
        po = ProbeObservation(u=u, y_reps=y_reps)
        if rec["role"] == "audit":
            audit.append(po)
        else:
            design.append(po)
    return audit, design


@dataclass
class SparcPolicy:
    """SPARC Phase-0 policy: audit/design probes + mismatch fit + trust-region edits."""

    seed: int = 0
    n_max: int = 8
    replicates: int = 3
    audit_frac: float = 0.25
    n_mc: int = 2048
    probe_sampling: Any = None
    mismatch_fit: Any = None
    edit: Any = None

    rng: np.random.Generator = field(init=False, repr=False)
    regime: Any = field(init=False, repr=False)
    benchmark: Any = field(init=False, repr=False)
    cfg_probe: ProbeSamplingConfig = field(init=False, repr=False)
    cfg_mismatch: MismatchFitConfig = field(init=False, repr=False)
    cfg_edit: EditConfig = field(init=False, repr=False)
    U_audit: np.ndarray = field(init=False, repr=False)
    U_mc: np.ndarray = field(init=False, repr=False)
    _audit_n: int = field(init=False, repr=False, default=0)
    _design_n: int = field(init=False, repr=False, default=0)
    round_idx: int = field(init=False, default=0)
    pending_theta_set: np.ndarray | None = field(init=False, default=None, repr=False)
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
        self.rounds = ExperimentHistory()

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

        theta_state = history.theta_snapshots[-1]
        theta_vec = (
            np.asarray(getattr(theta_state, "theta"), dtype=float)
            if hasattr(theta_state, "theta")
            else np.asarray(theta_state, dtype=float)
        )

        phi_hat, fit_diag = fit_mismatch(
            theta_k=theta_vec,
            obs_design=design_obs,
            y_sim_batch=self.benchmark.simulator_y_batch,
            cfg=self.cfg_mismatch,
        )

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
            phi=phi_hat,
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
            fit_diagnostics={**fit_diag, **edit_diag},
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
                        {"u": o.u.tolist(), "y_reps": o.y_reps.tolist()}
                        for o in round_rec.obs_audit
                    ],
                    "obs_design": [
                        {"u": o.u.tolist(), "y_reps": o.y_reps.tolist()}
                        for o in round_rec.obs_design
                    ],
                    "phi_fit": dict(round_rec.phi_fit),
                    "metrics": dict(round_rec.metrics),
                    "diagnostics": dict(round_rec.fit_diagnostics),
                }
            )

        self.round_idx += 1
