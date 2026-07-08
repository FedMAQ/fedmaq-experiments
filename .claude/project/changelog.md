# FedMAQ Agent Changelog

Archive of session-to-session changelog entries for the FedMAQ thesis codebase.

## Historical Entries

### 2026-07-09 — Manuscript Alignment: Proxy Pool Size, Discrete Bit-Widths, FedDistill Spec (experiments + manuscript)

- **Manuscript is canon:** Reconciled `fedmaq-experiments` against user-updated `chapter_1.tex`/`chapter_4.tex` (Ch. 1 Introduction, Ch. 4 Methodology). Where the two disagreed, code was changed to match the manuscript; manuscript-only issues were reported back rather than silently resolved.
- **Proxy/public pool size 200 -> 1600:** Updated `conf/experiment/{default,femnist,preliminary}.yaml`; fixed a general (not FEMNIST-specific) remainder-allocation bug in `partitioning.py` so the pool always totals exactly `num_public_samples` regardless of class count.
- **Discrete bit-width set:** `compute_fedmaq_q_k_t` (`core/strategy_hooks/fedmaq.py`) now snaps both the Tier-1 memory hard cap and the Tier-2 soft quality target to the manuscript's permissible set `Q = {1,2,3,4,5,6,7,8,16,32}` instead of an arbitrary continuous integer; added `bit_widths` to `conf/algorithm/fedmaq.yaml`.
- **Compressor semantics fix:** FedMAQ's client-side compressor switched from `DAdaQuantCompressionHook` (levels-per-sign) to `FedPAQCompressionHook` (true bit-width) — needed once 16/32-bit tiers were reachable, since the old hook would have badly misinterpreted `q=16` as only 33 levels.
- **`baseline_registry.md`:** FedDistill entry updated to the FEDDISTILL+/FedGen spec (Jeong et al.) per manuscript Sec 4.3.1, pointing at `references/feddistill/`; CFD entry notes no reference implementation exists yet.
- **Manuscript edits (in `fedmaq-manuscript`, uncommitted — user's repo to commit):** `chapter_4.tex` updated per the plan's report items: FedKD SVD threshold now shown as a `0.1 -> 0.95` schedule instead of a fixed constant; KD temperature split into two rows (FedMAQ/FedAvgKD `T=1.0` vs. FedKD internal `T=2.0`); the five candidate quantization formulations transcribed into a real table ahead of the still-pending pilot-study results placeholder; `|D_pub|`/`|D_proxy|` symbol duplication resolved to `|D_pub|=1600`; dataset-overview placeholder caption no longer lists MNIST/FMNIST (out of benchmark scope); the "516-run experimental grid" claim replaced with a flagged `[VERIFY: ...]` note showing the actual computed total (~165 runs) from the documented `conf/config.yaml` sweeps, left for the user to reconcile.
- **Tests:** Added coverage for exact-size proxy pools under remainder conditions and for bit-width snapping (including the 16/32-bit escape tiers); full suite passes (22/22).

### 2026-07-09 — CLAUDE.md Relocation to Repo Root and AGENTS.md Removal

- **Entry point moved:** `.claude/CLAUDE.md` → root `CLAUDE.md` in both `fedmaq-experiments` and `fedmaq-manuscript` — more discoverable location, and `@import` paths now resolve relative to repo root instead of `.claude/`.
- **Lean CLAUDE.md:** In experiments, only the 4 rule files that were `alwaysApply: true` under the old Cursor setup are `@import`ed (`project-overview.md`, `repo-preferences.md`, `manuscript-alignment.md`, `agent-delegation.md`), plus `HANDOFF.md`. The remaining 7 task-specific rule files are listed in a routing table (path + one-line trigger) instead of being unconditionally imported, restoring the conditional-loading behavior Cursor's `globs`/`alwaysApply: false` used to provide.
- **`AGENTS.md` removed** (experiments only — manuscript never had one): its resource-index content is now redundant since skills/commands are auto-discovered by Claude Code and registries are linked directly from the skills/commands that use them. The workspace map's "Agent entry" for experiments now points at `CLAUDE.md` instead.

### 2026-07-09 — Cursor to Claude Code Migration (experiments + manuscript)

- **Tooling migration:** Replaced Cursor config with Claude Code equivalents in `fedmaq-experiments` and `fedmaq-manuscript`. `.cursor/rules/*.mdc` → `.claude/rules/*.md` (frontmatter stripped, `agent-workflows.mdc` rewritten as `agent-delegation.md` with Claude-Code-native delegation guidance instead of Cursor subagent names), `.cursor/skills/*` → `.claude/skills/*`, `.cursor/project/*` → `.claude/project/*`, `.agents/workflows/*.md` → `.claude/commands/*.md` slash commands.
- **New entry point:** Both repos now have `.claude/CLAUDE.md`, which `@import`s the modular rule files (imports are unconditional, unlike Cursor's `alwaysApply`/`globs` scoping — noted explicitly in each CLAUDE.md; later superseded same-day, see entry above).
- **Cross-repo docs updated:** `HANDOFF.md` and `README.md` now point at `.claude/` locations for experiments and manuscript (`AGENTS.md` itself was later removed from experiments, see the entry above). `fedmaq-literature`, `fedmaq-analyses`, `fedmaq-presentations` are unmigrated; their `thesis-context.mdc` pointers to the old `fedmaq-experiments/.cursor/rules/` are now stale until their own future migration.
- **Deleted:** `.cursor/` and `.agents/` in `fedmaq-experiments` and `fedmaq-manuscript` (Cursor config fully replaced, not kept in parallel).
- **Adjacent cleanup:** Fixed stale manuscript status in the workspace map (was "template pending", corrected to reflect Ch 1-4 integrated/Ch 5 drafted/Ch 6 pending); fixed manuscript README's chapter list to include Ch 5-6.

### 2026-07-03 — Codebase Hardening, Optimization & Correctness (Refactor Session)

- **Partition Resolution Optimization**: Bypassed synchronous `client.get_properties()` RPC queries and 5s timeouts in strategy hooks (`dadaquant.py` and `fedmaq.py`) by checking `cid_str.isdigit()` and validating against client counts.
- **Ensemble Evaluation Memory Optimization**: Reduced PyTorch allocation and memory overhead in FedMD evaluation (`evaluation.py`) by instantiating the model once outside the checkpoint loop and reusing it.
- **Stochastic Rounding Correctness**: Seeded each client's random number generator using `cfg.seed + partition_id` in `run.py`, ensuring mathematically independent stochastic rounding across clients while remaining fully reproducible.
- **Telemetry Robustness**: Guarded `import wandb` in `telemetry.py` to prevent import crashes when the library is not installed, and pre-populated the CSV header with a stable canonical column order.
- **Safe Device Config Resolution**: Handled explicit `null` / `None` device definitions in configurations across client and strategy hooks.
- **Validation**: Executed test suite (20/20 tests passed) and ran end-to-end simulation dry runs to verify changes.

### 2026-07-03 — Core Codebase Refactoring & Hardening

- **Strategy Modularization**: Extracted algorithm-specific logic from `TelemetryFedAvg` inside `strategy.py` into distinct modular strategy hooks (`core/strategy_hooks/`), significantly reducing complexity (953 → 430 lines) and preparing the ground for Task 10/11 (FedDistill/CFD).
- **Correctness Fix**: Fixed a critical estimation error in `DAdaQuantCompressionHook` where quantization levels were treated directly as bits, correcting upload bandwidth metrics.
- **Performance Optimizations**: Added `lru_cache` to torchvision dataset loading to eliminate redundant disk reads across clients, and vectorised F1 evaluation metrics using `scikit-learn`.
- **Typing & Robustness**: Aligned API signatures with Flower types, resolved bare exception swallows, locked CSV schemas, and resolved mutable closure risks in `evaluate_fn`.
- **Test Validation**: Confirmed that all 20 environment, simulation, and hook unit tests pass cleanly after the refactoring.

### 2026-07-02 — Workspace Agent Context Pruning & Slash Workflows Setup

- **Context Optimization**: Pruned redundant tables, literature specifications, and logs from `HANDOFF.md`, linking to active registries and stack readmes.
- **Workflow Automation**: Defined and automated project-scoped triggers for `/align-manuscript`, `/add-baseline`, and `/run-benchmark` under `.agents/workflows/`.
- **Rule Consolidation**: Deleted redundant `.cursor/rules/hyperparameters.mdc`, deferring default configs and constraints to `manuscript-alignment.mdc`.
- **Test Integrity**: Executed `uv run pytest` to ensure all 20 environment and simulation tests pass successfully.

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
