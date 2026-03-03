from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path


def test_smoke_run_sparc_single_paulsson_creates_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_sparc"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd())
    cmd = [
        sys.executable,
        "-m",
        "beginoi.cli",
        "task=run",
        "regime=single_paulsson",
        "benchmark=paulsson_machine",
        "policy=sparc",
        "metrics=sparc",
        "seed=0",
        "budget.total=3.0",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
    ]
    subprocess.check_call(cmd, env=env)

    assert (run_dir / "meta.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "sparc_rounds.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "final_metrics" in summary
