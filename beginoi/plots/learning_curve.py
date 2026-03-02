from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # pylint: disable=wrong-import-position


def plot_learning_curve(
    run_dir: str | Path, *, metric_key: str = "heatmap_rmse"
) -> Path:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    xs: list[float] = []
    ys: list[float] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            rec = json.loads(line)
            xs.append(float(rec.get("budget_spent", len(xs))))
            ys.append(float(rec.get(metric_key, float("nan"))))

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Budget spent")
    ax.set_ylabel(metric_key)
    ax.set_title("Learning curve")
    ax.grid(True, alpha=0.3)
    out = run_dir / "figures" / f"learning_curve_{metric_key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
