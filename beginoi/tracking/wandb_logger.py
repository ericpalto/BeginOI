from __future__ import annotations

from typing import Any
from dataclasses import dataclass

# pylint: disable=import-outside-toplevel


@dataclass
class WandbRunLogger:
    """Weights & Biases logger (optional dependency)."""

    project: str
    mode: str = "offline"  # disabled|offline|online
    group: str | None = None
    tags: list[str] | None = None

    _run: Any = None

    def open(self, *, config: Any) -> None:
        if self.mode == "disabled":
            return
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "W&B tracking selected but wandb is not installed. "
                "Install with: `uv sync --extra tracking`."
            ) from exc

        self._run = wandb.init(
            project=self.project,
            mode="offline" if self.mode == "offline" else "online",
            group=self.group,
            tags=self.tags,
            config=config,
        )

    def log_unit(self, *, unit_id: int, history: Any, metrics: dict[str, Any]) -> None:
        if self.mode == "disabled" or self._run is None:
            return
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover
            return
        wandb.log(
            {"budget_spent": float(history.budget_spent), **metrics}, step=int(unit_id)
        )

    def close(self, *, summary: dict[str, Any], history: Any) -> None:
        del history
        if self.mode == "disabled" or self._run is None:
            return
        self._run.summary.update(dict(summary))
        self._run.finish()
