from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # pylint: disable=wrong-import-position

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


def _grid_points(grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.asarray(grid.x1, dtype=float)
    x2 = np.asarray(grid.x2, dtype=float)
    U = np.array([(a, b) for a in x1 for b in x2], dtype=float)
    return x1, x2, U


def _plot_heatmap(
    *,
    out_path: Path,
    title: str,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    U_audit: np.ndarray,
    U_design: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(6, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        y.T,
        origin="lower",
        extent=(float(x1[0]), float(x1[-1]), float(x2[0]), float(x2[-1])),
        aspect="auto",
        cmap="viridis",
    )
    ax.scatter(
        U_design[:, 0],
        U_design[:, 1],
        s=30,
        c="white",
        marker="x",
        label="design",
    )
    ax.scatter(
        U_audit[:, 0],
        U_audit[:, 1],
        s=35,
        facecolors="none",
        edgecolors="red",
        marker="o",
        linewidths=1.5,
        label="audit",
    )
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class SparcRoundLogger:
    """Writes per-round SPARC records from History.extras['sparc_rounds']."""

    run_dir: Path
    plant: Any | None = None
    benchmark: Any | None = None
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
            self._maybe_plot_round(rec)
        self._f.flush()
        self._written = len(rounds)

    def _maybe_plot_round(self, rec: dict[str, Any]) -> None:
        if self.benchmark is None or self.plant is None:
            return
        if not hasattr(self.benchmark, "grid") or not hasattr(
            self.benchmark, "simulator_y_batch"
        ):
            return
        if not hasattr(self.plant, "real_mean_batch"):
            return

        round_id = int(rec.get("round", 0))
        theta_k = np.asarray(rec.get("theta_k", []), dtype=float)
        U_audit = np.asarray(rec.get("U_audit", []), dtype=float)
        U_design = np.asarray(rec.get("U_design", []), dtype=float)
        if U_audit.ndim != 2 or U_audit.shape[1] != 2:
            return
        if U_design.ndim != 2 or U_design.shape[1] != 2:
            return

        x1, x2, U_grid = _grid_points(self.benchmark.grid)
        y_sim = np.asarray(
            self.benchmark.simulator_y_batch(U_grid, theta_k), dtype=float
        ).reshape((len(x1), len(x2)))
        y_real = np.asarray(
            self.plant.real_mean_batch(U_grid, theta=theta_k), dtype=float
        ).reshape((len(x1), len(x2)))

        out_dir = self.run_dir / "sparc_heatmaps"
        sim_path = out_dir / f"round_{round_id:03d}_y_sim.png"
        real_path = out_dir / f"round_{round_id:03d}_y_real.png"
        _plot_heatmap(
            out_path=sim_path,
            title=f"y_sim round {round_id}",
            x1=x1,
            x2=x2,
            y=y_sim,
            U_audit=U_audit,
            U_design=U_design,
        )
        _plot_heatmap(
            out_path=real_path,
            title=f"y_real round {round_id}",
            x1=x1,
            x2=x2,
            y=y_real,
            U_audit=U_audit,
            U_design=U_design,
        )
        self._maybe_log_wandb_images(
            round_id=round_id,
            sim_path=sim_path,
            real_path=real_path,
        )

    @staticmethod
    def _maybe_log_wandb_images(
        *,
        round_id: int,
        sim_path: Path,
        real_path: Path,
    ) -> None:
        if wandb is None:
            return
        if getattr(wandb, "run", None) is None:
            return
        wandb.log(
            {
                "sparc_heatmaps/y_sim": wandb.Image(
                    str(sim_path), caption=f"y_sim round {round_id}"
                ),
                "sparc_heatmaps/y_real": wandb.Image(
                    str(real_path), caption=f"y_real round {round_id}"
                ),
            },
            step=int(round_id),
        )

    def close(self, *, summary: dict[str, Any], history: Any) -> None:
        del summary, history
        if self._f is not None:
            self._f.close()
