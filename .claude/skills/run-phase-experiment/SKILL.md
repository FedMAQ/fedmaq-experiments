---
name: run-phase-experiment
description: Run a phase experiment with Hydra overrides and verify WandB logging
---

# Run Phase Experiment

1. Identify phase (`phase1_env` … `phase4_benchmark`) and script under `scripts/`.
2. Run via `uv run` with Hydra overrides, e.g. `uv run python scripts/run.py dataset=cifar10`.
3. Confirm WandB logs metrics from [evaluation-metrics.md](../../rules/evaluation-metrics.md).
4. Record run in [.claude/project/experiment_registry.md](../../project/experiment_registry.md) (phase, dataset, alpha, WandB run ID).
