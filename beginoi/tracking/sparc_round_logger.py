from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass


@dataclass
class SparcRoundLogger:
    """Writes per-round SPARC records from History.extras['sparc_rounds']."""

    run_dir: Path
    _f: Any = field(init=False, repr=False, default=None)
    _written: int = field(init=False, repr=False, default=0)

    def open(self, *, config: Any) -> None:
        del config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._f = (self.run_dir / "sparc_rounds.jsonl").open("a", encoding="utf-8")
        self._written = 0

    def log_unit(self, *, unit_id: int, history: Any, metrics: dict[str, Any]) -> None:
        del unit_id, metrics
        if self._f is None:
            raise RuntimeError("SparcRoundLogger not opened.")
        extras = getattr(history, "extras", None) or {}
        rounds = list(extras.get("sparc_rounds", []))
        new = rounds[self._written :]
        for rec in new:
            self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()
        self._written = len(rounds)

    def close(self, *, summary: dict[str, Any], history: Any) -> None:
        del summary, history
        if self._f is not None:
            self._f.close()
