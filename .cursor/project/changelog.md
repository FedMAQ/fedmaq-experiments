# FedMAQ Agent Changelog

Archive of session-to-session changelog entries for the FedMAQ thesis codebase.

## Historical Entries

### 2026-07-02 — Manuscript Alignment Rule Creation and Test Expansion

- **Canonical Alignment Rules**: Created `manuscript-alignment.mdc` under `.cursor/rules/` to enforce hyperparameter synchronization (Table 4.1), soft quality formulations (0-4), decoupled simulated delays, and test requirements.
- **Formulation Test Expansion**: Expanded `test_compute_fedmaq_q_k_t` in `tests/test_environment.py` to assert correct bit-width allocations and boundary conditions for all five soft quality target formulations (Resource-Only, Linear Sum, Multiplicative, Gradient-Primary, and Threshold-Based).
- **Test Suite Robustness**: Verified all 20 test cases pass cleanly after expansion.

### 2026-07-02 — Uniform System Simulation and Preliminary Config Updates

- **Uniform System Parameters:** Removed heterogeneous bandwidth and compute parameters (from `conf/experiment/default.yaml` and `tests/test_environment.py`). Migrated to uniform system simulation using configurable `bandwidth_mbps` (default 10.0 Mbps) and `compute_samples_per_sec` (default 200 samples/sec) across all clients in `TelemetryFedAvg`.
- **Preliminary Test Setup:** Configured `num_clients: 50` and `total_rounds: 10` for preliminary iterative comparisons.
- **Decoupled simulated runtime:** Implemented decoupled simulated client training time and server processing/distillation time in `TelemetryFedAvg` strategy (`strategy.py`) and `TelemetryManager` (`telemetry.py`), writing them to separate metrics fields (`system/client_sim_time_sec` and `system/server_sim_time_sec`).
- **FedKD compute scaling:** Applied a $2.5\times$ computational penalty scale factor to client compute speeds for `fedkd` to model the increased overhead of training both student and teacher models concurrently.
- **FedMD pre-training delay:** Adjusted the client training delay simulation for `fedmd` during Round 1 to include the 10 public pre-training and 10 private pre-training epochs.
- **FedMAQ server delay:** Formulated and added the server-side proxy ensemble distillation delay ($T_{server}$) for `fedmaq` based on public proxy dataset size, epoch count, and active teachers.
- **Test suite flakiness resolved:** Updated the `get_properties` mock in `test_dadaquant_strategy_allocation` inside `test_environment.py` to deterministically map clients to partitions, preventing hash-seed random fallback failures.
- **5-Round benchmark simulation:** Executed full 5-round MNIST training runs for all 7 FL algorithms (`fedavg`, `fedprox`, `fedpaq`, `dadaquant`, `fedmd`, `fedkd`, and `fedmaq`) using GPU training, validating stable convergence and correct logging.
- **Manuscript alignment:** Updated subsubsection items in `chapter_4.tex` to clean up brackets and synchronize the decoupled simulated time formulation.

### 2026-07-01 — Manuscript Table 4.1 Hyperparameter Synchronization and Gitignore Cleanup

- **Hyperparameter alignment:** Modified `chapter_4.tex` and `.cursor/rules/hyperparameters.mdc` to split weight decay and momentum, add learning rate decay ($\gamma = 0.99$), and correct weight decay to $\lambda = 10^{-4}$ (0.0001).
- **Learning rate decay:** Implemented `_get_decayed_lr` in `GenericClient` in `client.py` and integrated it across standard, FedMD, and FedKD training loops to apply exponential per-round decay.
- **Default parameters corrected:** Updated parameter defaults for public dataset size from 500 to 200 in `partitioning.py` and `strategy.py` to match the manuscript default value.
- **Gitignore and untracked logs:** Added `wandb/` and local logs `experiment_log.csv` and `experiment_log.jsonl` to `.gitignore` in `fedmaq-experiments`, and ran a cached git remove to untrack them.
- **Rule alignment:** Updated `.cursor/rules/` (`baselines.mdc`, `datasets-simulation.mdc`, `evaluation-metrics.mdc`, `hyperparameters.mdc`) to match the streamlined baseline count (8), corrected Dirichlet alpha values (0.1, 1.0, 10.0), and added auxiliary metrics.

### 2026-07-01 — Full Manuscript Audit (Ch. 1--4) and Codebase Hardening

**Audit scope:** All four released chapters compared line-by-line against the experiment codebase.

- Identified and fixed 7 issues across config, source, and scripts; all 19 tests continue to pass post-changes.
- **Bug: heterogeneous compute speed** — `client_comp_speed` was always pinned to `comp_max`; changed to `rng.uniform(comp_min, comp_max)` per §2.2 ("variable CPU frequencies").
- **Bug: unseeded stochastic rounding** — `DAdaQuantCompressionHook.compress()` used the global `np.random.rand`, breaking reproducibility; replaced with an injectable `np.random.Generator`.
- **Documented 5 canonical seeds** `[0, 42, 123, 456, 789]` in `conf/config.yaml` with multirun sweep command; `run.py` now passes `np.random.default_rng(cfg.seed)` to the hook (§4.3 statistical controls).
- **Config separation (CI vs. production):** `conf/experiment/default.yaml` restored to `total_rounds: 100`; new `conf/experiment/ci.yaml` provides `total_rounds: 2` override (`+experiment=ci`).
- **Dead config removed:** `kd_weight: 0.5` stripped from `fedmaq.yaml`; replaced with `kd_epochs: 1`, `server_kd_lr`, `server_kd_momentum` as explicit, named KD parameters.
- **Control group config added:** `conf/heterogeneity/uniform_memory.yaml` (8192 MB fixed) for §4.1 ablation; wired into `TelemetryFedAvg.__init__` via `uniform_memory_mb` key.
- **`run_server_side_kd` gains `epochs` param:** distillation passes over proxy dataset now configurable; body correctly nested inside both epoch and batch loops.
- **Stale default fixed:** `num_public_samples` hardcoded default in `aggregate_fit` corrected from 500 → 200.
- **Safe OmegaConf device resolution:** `cfg.get("device", DEVICE)` → `OmegaConf.select(cfg, "device", default=None)` in `evaluate_fn`.
- **Potential manuscript error flagged:** Table 4.1 `Weight Decay / Momentum ρ = 0.99` conflates two hyperparameters with a numerically suspect value; recommend splitting into `Momentum ρ = 0.9` / `Weight Decay = 0.0`.
- **FedDistill and CFD confirmed as zero-implementation stubs** — on schedule for Sep--Oct 2026 per Gantt.

### 2026-07-01 — Revised FedMAQ methodology, auxiliary metrics, and local telemetry logging

- Simplified local client training for FedMAQ in `client.py` to strictly perform task-loss cross-entropy minimization ($L_{local} = CE(\hat{y}, y)$), removing student-teacher mutual learning and model checkpoint persistence on the client.
- Redesigned server-side ensemble distillation for FedMAQ in `strategy.py` to dynamically construct client teacher models in memory from uploaded parameter updates, completely bypassing disk checkpoint files.
- Configured a uniform client computation speed (`comp_max`) in `TelemetryFedAvg.__init__` to simulate uniform hardware across the federation.
- Integrated macro-averaged Precision, Recall, and F1-score evaluation metrics on the server in `run.py`'s global and client-averaging evaluation paths.
- Setup local telemetry logging to generate isolated `experiment_log.jsonl` and `experiment_log.csv` run artifacts within each Hydra execution directory using `HydraConfig`.
- Successfully validated the implementation with the complete test suite and end-to-end 2-round MNIST simulation runs for FedAvg and FedMAQ.
