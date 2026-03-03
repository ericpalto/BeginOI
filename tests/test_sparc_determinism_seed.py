from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def _run(tmp: Path, *, seed: int) -> Path:
    run_dir = tmp
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
        f"seed={seed}",
        "budget.total=3.0",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
    ]
    subprocess.check_call(cmd, env=env)
    return run_dir


def test_sparc_same_seed_same_round_records(tmp_path: Path) -> None:
    d1 = _run(tmp_path / "r1", seed=123)
    d2 = _run(tmp_path / "r2", seed=123)

    rec1 = (d1 / "sparc_rounds.jsonl").read_text()
    rec2 = (d2 / "sparc_rounds.jsonl").read_text()
    assert rec1 == rec2
