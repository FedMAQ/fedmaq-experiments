---
name: run-benchmark
description: >-
  Executes baseline sweeps or pilot studies and logs to WandB. Use when the
  user asks to run an experiment, a benchmark sweep, or a multirun.
---

# Run Benchmark

To run a benchmark sweep:

1. Ensure `WANDB_API_KEY` is set in your local `.env`.
2. Determine if running a staging/preliminary dry-run or production training:
   - Staging (10 rounds): `uv run python scripts/run.py experiment=preliminary`
   - Production (100 rounds): `uv run python scripts/run.py experiment=default`
3. Execute multirun sweeps for multiple algorithms, seeds, and alphas using Hydra:
   - Example (production): `uv run python scripts/run.py --multirun dataset=cifar10,cifar100 heterogeneity=dirichlet_alpha_0.1,dirichlet_alpha_1.0 algorithm=fedavg,fedprox,fedmaq seed=0,42,123`
4. Monitor metrics (accuracy, losses, communication overhead, simulated training times) on the WandB dashboard.
5. Record completed runs in [.claude/project/experiment_registry.md](../../project/experiment_registry.md).
