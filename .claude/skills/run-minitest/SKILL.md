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

1. Run the sweep:
   ```
   uv run python scripts/run.py --multirun \
     experiment=preliminary \
     dataset=cifar10 \
     heterogeneity=dirichlet_alpha_0.1,dirichlet_alpha_1.0 \
     algorithm=fedavg,fedprox,fedpaq,dadaquant,fedmd,fedkd,fedmaq,feddistill,cfd \
     seed=0
   ```
2. Inspect per-run `experiment_log.jsonl`/`.csv` under `multirun/<date>/<time>/<job_num>/` for
   accuracy, loss, and communication-overhead trends — confirm FedMAQ's accuracy-vs-bytes
   curve beats or tracks toward beating the pure-quantization and hybrid baselines.
3. Optionally narrow to one algorithm/alpha for a faster single-run check by dropping
   `--multirun` and fixing `algorithm=`/`heterogeneity=` to a single value.
4. Do not log these to `.claude/project/experiment_registry.md` (that registry is for
   methodology-compliant benchmark runs) — sanity runs are throwaway and not tracked there.
5. Re-run anytime after implementation changes to re-check trend direction.
