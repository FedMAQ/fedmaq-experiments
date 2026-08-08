---
name: run-minitest
description:
  Runs a minified sanity-check sweep across all baseline algorithms and the
  main FedMAQ formulation to eyeball whether implementations behave and whether FedMAQ
  is trending correctly. Not a benchmark run — no methodology rigor, single seed, local
  telemetry only. Use when asked for a quick sanity/pilot check across algorithms.
---

# Run Minitest

Quick, repeatable sanity sweep — not a citable benchmark result. Uses `experiment=preliminary`
(50 clients, 50 rounds, 0.2 client participation rate) instead of full-scale `experiment=default` (100/100), single seed,
local-only telemetry.

1. Do **NOT** use Hydra's `--multirun` CLI flag directly due to PyTorch/Ray process memory leaks causing CUDA Out-of-Memory (OOM) errors.
2. Run the process-isolated smoke test sweep instead: `uv run python scripts/run_smoke_test.py`. This executes 10 rounds for each of the 9 algorithms on CIFAR-10 with alpha=1.0.
3. Inspect per-run `experiment_log.jsonl`/`.csv` under `multirun/<date>/<time>/<job_num>/` for accuracy, loss, and communication-overhead trends.
4. For a faster single-run check of one algorithm/alpha, you can run a single job directly (without `--multirun`): `uv run python scripts/run.py experiment=preliminary dataset=cifar10 heterogeneity=dirichlet_alpha_1.0 algorithm=fedmaq seed=0`.
5. Re-run anytime after implementation changes to re-check trend direction.
