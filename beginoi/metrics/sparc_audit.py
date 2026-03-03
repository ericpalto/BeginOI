from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    av = a - np.mean(a)
    bv = b - np.mean(b)
    sa = float(np.sqrt(np.sum(av**2)))
    sb = float(np.sqrt(np.sum(bv**2)))
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    denom = float(sa * sb)
    return float(np.sum(av * bv) / denom)


@dataclass(frozen=True)
class SparcAuditMetric:
    """Audit-only metric for SPARC: MSE(y_mean, g(u)) on most recent unit."""

    name_prefix: str = "sparc"
    threshold: float = 0.5

    def __call__(self, history, benchmark: Any) -> dict[str, Any]:
        if not history.observations:
            return {f"{self.name_prefix}_audit_mse": float("nan")}

        unit_id = max(int(o.unit_id) for o in history.observations)
        obs = [o for o in history.observations if int(o.unit_id) == unit_id]
        audit = [o for o in obs if o.extra.get("sparc_role") == "audit"]
        if not audit:
            return {f"{self.name_prefix}_audit_mse": float("nan")}

        groups: dict[str, list[float]] = {}
        u_by: dict[str, np.ndarray] = {}
        for o in audit:
            pid = str(o.extra.get("sparc_probe_id"))
            groups.setdefault(pid, []).append(float(o.y))
            if pid not in u_by:
                u_by[pid] = np.asarray(o.inputs_summary["u"], dtype=float)

        U = np.array([u_by[k] for k in sorted(u_by)], dtype=float)
        y_mean = np.array([np.mean(groups[k]) for k in sorted(groups)], dtype=float)
        y_var = np.array(
            [
                np.var(groups[k], ddof=1) if len(groups[k]) > 1 else 0.0
                for k in sorted(groups)
            ],
            dtype=float,
        )

        if hasattr(benchmark, "target_g_batch"):
            g = np.asarray(benchmark.target_g_batch(U), dtype=float)
        else:
            raise ValueError(
                "Benchmark must expose target_g_batch(U) for SparcAuditMetric."
            )

        mse = float(np.mean((y_mean - g) ** 2))
        frac_y = float(np.mean(y_mean > float(self.threshold)))
        frac_g = float(np.mean(g > float(self.threshold)))
        return {
            f"{self.name_prefix}_audit_mse": mse,
            f"{self.name_prefix}_audit_corr": _corr(y_mean, g),
            f"{self.name_prefix}_audit_frac_y_above": frac_y,
            f"{self.name_prefix}_audit_frac_g_above": frac_g,
            f"{self.name_prefix}_audit_mean_var": float(np.mean(y_var)),
        }
