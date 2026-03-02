from __future__ import annotations

import sys
import json
import platform
import subprocess
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import field, dataclass


def _safe_git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


@dataclass
class LocalRunLogger:
    """Local artifact writer for reproducible `runs/<id>/` directories."""

    run_dir: Path
    _history_f: Any = field(init=False, repr=False, default=None)
    _metrics_f: Any = field(init=False, repr=False, default=None)
    _written_obs: int = field(init=False, repr=False, default=0)

    def open(self, *, config: Any) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "heatmaps").mkdir(parents=True, exist_ok=True)

        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": _safe_git_head(),
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )

        # Store resolved config as JSON for stable downstream parsing
        # (Hydra also stores YAML under `.hydra/`).
        (self.run_dir / "config_resolved.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str)
        )

        self._history_f = (self.run_dir / "history.jsonl").open("a", encoding="utf-8")
        self._metrics_f = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._written_obs = 0

    def log_unit(self, *, unit_id: int, history: Any, metrics: dict[str, Any]) -> None:
        if self._history_f is None or self._metrics_f is None:
            raise RuntimeError("LocalRunLogger not opened.")

        # Append newly observed records.
        obs = history.observations[self._written_obs :]
        for o in obs:
            rec = {
                "unit_id": int(o.unit_id),
                "program_id": o.program_id,
                "inputs_summary": o.inputs_summary,
                "y": float(o.y),
                "t_obs": float(o.t_obs),
                "noise_meta": dict(o.noise_meta),
                "replicate_id": int(o.replicate_id),
                "extra": dict(o.extra),
            }
            self._history_f.write(json.dumps(rec) + "\n")
        self._history_f.flush()
        self._written_obs = len(history.observations)

        mrec = {
            "unit_id": int(unit_id),
            "budget_spent": float(history.budget_spent),
            **metrics,
        }
        self._metrics_f.write(json.dumps(mrec, default=str) + "\n")
        self._metrics_f.flush()

    def close(self, *, summary: dict[str, Any], history: Any) -> None:
        del history
        (self.run_dir / "final_metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str)
        )
        if self._history_f is not None:
            self._history_f.close()
        if self._metrics_f is not None:
            self._metrics_f.close()
