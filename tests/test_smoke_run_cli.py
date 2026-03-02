from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path


def test_smoke_run_cli_creates_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd())
    cmd = [
        sys.executable,
        "-m",
        "beginoi.cli",
        "task=run",
        "regime=single_standard",
        "benchmark=ode_lotka_volterra",
        "policy=random",
        "seed=0",
        "budget.total=2.0",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
    ]
    subprocess.check_call(cmd, env=env)

    assert (run_dir / "meta.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "final_metrics" in summary
