# BeginOI

Research repo for **budgeted sim-to-real calibration of steady-state I/O maps** (2D input → 1D output heatmap) with misspecified simulators.

## Quickstart
- Install deps: `uv sync`
- Run a small experiment (default config): `uv run beginoi task=run`
- Switch regimes (“quadrants”): `uv run beginoi task=run regime=multi_paulsson`
- Switch benchmark/policy: `uv run beginoi task=run benchmark=ode_brusselator policy=batch_active`
- Plot a learning curve from a run dir: `uv run beginoi task=plot task.run_dir=runs/<...>`

## Tracking
- Local artifacts always land in `runs/...` (ignored by git).
- Optional W&B: `uv sync --extra tracking` then `uv run beginoi tracking=wandb tracking.mode=offline`
