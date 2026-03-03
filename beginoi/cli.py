from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, cast
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from beginoi.core.types import Budget
from beginoi.core.runner import run_loop
from beginoi.regimes.config import RegimeConfig
from beginoi.regimes.regime import Regime
from beginoi.metrics.heatmap import HeatmapErrorMetric
from beginoi.plots.learning_curve import plot_learning_curve
from beginoi.tracking.local_writer import LocalRunLogger
from beginoi.tracking.wandb_logger import WandbRunLogger
from beginoi.tracking.sparc_round_logger import SparcRoundLogger


def _make_regime(cfg: DictConfig) -> Regime:
    def _require(value: str, *, allowed: set[str], field_name: str) -> str:
        if value not in allowed:
            allowed_list = sorted(allowed)
            raise ValueError(f"Invalid {field_name}={value!r}. Allowed: {allowed_list}")
        return value

    rc = RegimeConfig(
        copy_mode=cast(
            Any,
            _require(
                str(cfg.copy_mode),
                allowed={"single", "multi"},
                field_name="copy_mode",
            ),
        ),
        instrument=cast(
            Any,
            _require(
                str(cfg.instrument),
                allowed={"standard_lab", "paulsson_machine"},
                field_name="instrument",
            ),
        ),
        theta_dynamics=cast(
            Any,
            _require(
                str(cfg.theta_dynamics),
                allowed={"discrete", "continuous"},
                field_name="theta_dynamics",
            ),
        ),
        feedback=bool(cfg.feedback),
        program_kind=cast(
            Any,
            _require(
                str(cfg.program_kind),
                allowed={"constant", "timeseries"},
                field_name="program_kind",
            ),
        ),
        max_programs_per_unit=int(cfg.max_programs_per_unit),
        observation_schedule=cast(
            Any,
            _require(
                str(cfg.observation_schedule),
                allowed={"end_only", "streaming"},
                field_name="observation_schedule",
            ),
        ),
        exposure_model=cast(
            Any,
            _require(
                str(cfg.exposure_model),
                allowed={"none", "piecewise_constant", "continuous"},
                field_name="exposure_model",
            ),
        ),
    )
    return Regime(cfg=rc)


def _make_metrics(cfg: DictConfig, *, plant: Any) -> list[Any]:
    metrics: list[Any] = []
    for m_cfg in cfg:
        metric = instantiate(m_cfg)
        if isinstance(metric, HeatmapErrorMetric):

            def _wrapped(history, benchmark, metric=metric):
                return metric(history, benchmark, plant=plant)

            metrics.append(_wrapped)
        else:
            metrics.append(metric)
    return metrics


def _make_loggers(cfg: DictConfig, *, run_dir: Path) -> list[Any]:
    loggers: list[Any] = []
    loggers.append(LocalRunLogger(run_dir=run_dir))
    loggers.append(SparcRoundLogger(run_dir=run_dir))

    tracking = cfg.tracking
    if tracking.name == "wandb":
        loggers.append(
            WandbRunLogger(
                project=str(tracking.project),
                mode=str(tracking.mode),
                group=str(tracking.group) if tracking.group is not None else None,
                tags=list(tracking.tags) if tracking.tags is not None else None,
            )
        )
    return loggers


def _task_run(cfg: DictConfig) -> dict[str, Any]:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    regime = _make_regime(cfg.regime)
    benchmark = instantiate(cfg.benchmark)
    plant = benchmark.make_plant(regime=regime, seed=int(cfg.seed))

    experiment = instantiate(
        cfg.experiment,
        max_programs=int(regime.max_programs_per_unit),
        streaming=(str(regime.cfg.observation_schedule) == "streaming"),
    )
    policy = instantiate(cfg.policy)

    rng = np.random.default_rng(int(cfg.seed))
    metrics = _make_metrics(cfg.metrics, plant=plant)
    loggers = _make_loggers(cfg, run_dir=run_dir)

    for logger in loggers:
        logger.open(config=OmegaConf.to_container(cfg, resolve=True))

    summary = run_loop(
        plant=plant,
        experiment=experiment,
        policy=policy,
        regime=regime,
        benchmark=benchmark,
        budget=Budget(total=float(cfg.budget.total)),
        metrics=metrics,
        rng=rng,
        loggers=loggers,
    )
    out = {
        "final_metrics": summary.final_metrics,
        "budget_spent": summary.history.budget_spent,
    }
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def _task_plot(cfg: DictConfig) -> dict[str, Any]:
    run_dir = Path(cfg.task.run_dir)
    out = plot_learning_curve(run_dir, metric_key=str(cfg.task.metric_key))
    return {"figure": str(out)}


def _task_summarize(cfg: DictConfig) -> dict[str, Any]:
    base = Path(cfg.task.runs_dir)
    rows: list[dict[str, Any]] = []
    for p in sorted(base.glob("**/summary.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, JSONDecodeError):
            continue
    out_path = base / "summary_aggregate.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    return {"count": len(rows), "out": str(out_path)}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> Any:
    task = str(cfg.task.name)
    if task == "run":
        return _task_run(cfg)
    if task == "plot":
        return _task_plot(cfg)
    if task == "summarize":
        return _task_summarize(cfg)
    raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
