from __future__ import annotations

from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np

from .types import Budget, History, ControlAction, ExperimentResult
from .interfaces import Plant, Policy, MetricFn, Experiment


@dataclass(frozen=True)
class RunSummary:
    """Final outputs from a run: history + last metric values."""

    history: History
    final_metrics: dict[str, Any]


def _maybe_call_on_observation(policy: Policy, obs) -> Any | None:
    handler = getattr(policy, "on_observation", None)
    if handler is None:
        return None
    return handler(obs)


def run_loop(
    *,
    plant: Plant,
    experiment: Experiment,
    policy: Policy,
    regime: Any,
    benchmark: Any,
    budget: Budget,
    metrics: Iterable[MetricFn],
    rng: np.random.Generator,
    loggers: list[Any] | None = None,
) -> RunSummary:
    history = History()
    theta = plant.reset(int(rng.integers(0, 2**31 - 1)))
    history.theta_snapshots.append(theta)

    policy.init(int(rng.integers(0, 2**31 - 1)), benchmark=benchmark, regime=regime)

    unit_id = 0
    loggers = [] if loggers is None else list(loggers)

    while budget.remaining > 0.0:
        action: ControlAction = policy.act(history, budget_remaining=budget.remaining)
        regime.validate_action(action)

        cost = float(experiment.cost_of(action.batch))
        if not budget.can_afford(cost):
            break

        feedback_handler = None
        if regime.feedback:

            def _handler(obs):
                return _maybe_call_on_observation(policy, obs)

            feedback_handler = _handler

        result: ExperimentResult = experiment.run_budget_unit(
            plant,
            theta,
            action,
            unit_id=unit_id,
            feedback_handler=feedback_handler,
            rng=rng,
        )
        if result.consumed_cost < 0:
            raise ValueError("Experiment consumed negative cost.")

        budget.spent += float(result.consumed_cost)
        history.budget_spent = float(budget.spent)
        history.actions.append(action)
        history.observations.extend(result.observations)
        theta = result.final_theta
        history.theta_snapshots.append(theta)
        policy.update(history, result)

        metric_values: dict[str, Any] = {}
        for metric_fn in metrics:
            metric_values.update(metric_fn(history, benchmark))

        for logger in loggers:
            logger.log_unit(unit_id=unit_id, history=history, metrics=metric_values)

        unit_id += 1

    final_metrics: dict[str, Any] = {}
    for metric_fn in metrics:
        final_metrics.update(metric_fn(history, benchmark))

    for logger in loggers:
        logger.close(summary=final_metrics, history=history)

    return RunSummary(history=history, final_metrics=final_metrics)
