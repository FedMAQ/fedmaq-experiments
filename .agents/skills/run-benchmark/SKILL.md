---
name: run-benchmark
description: >-
  Executes baseline sweeps or pilot studies and logs to WandB. Use when the
  user asks to run an experiment, a benchmark sweep, or a multirun.
---

# Run Benchmark

To run a benchmark sweep:

1. Ensure `WANDB_API_KEY` is set in your local `.env`.
2. Do **NOT** use Hydra's `--multirun` CLI flag directly due to PyTorch/Ray process memory leaks causing CUDA Out-of-Memory (OOM) errors.
3. Instead, execute sweeps using the process-isolated python runners in `scripts/` (which run each job in a separate subprocess and clean up Ray):
   - **Full Benchmark Sweep (100 Rounds, 2 Datasets, 2 Alphas, 9 Algos, 3 Seeds):**
     `uv run python scripts/run_benchmark_grid.py`
   - **FEMNIST Writer Sweep (100 Rounds, 9 Algos, 3 Seeds):**
     `uv run python scripts/run_femnist_grid.py`
   - **Pilot Formulation Study (100 Rounds, 2 Alphas, FedMAQ Formulations 0-4, 3 Seeds):**
     `uv run python scripts/run_formulation_ablation.py`
4. Monitor metrics on the WandB dashboard.
