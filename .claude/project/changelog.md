# FedMAQ Agent Changelog

Archive of session-to-session changelog entries for the FedMAQ thesis codebase.

## Historical Entries

### 2026-07-11 — CFD Baseline Port, Paper-Faithful (experiments)

Ports the last unported baseline (P11): CFD (Compressed Federated Distillation, Sattler
et al. 2022). Exchanges quantized soft-labels on the shared public proxy set instead of
weights/gradients; the manuscript (ch4 §Baselines) names it one of only two reproducible
Category-D (Hybrid Q+KD) comparators, so the 195-run grid was blocked on this port.

- **`core/softlabel_codec.py`** (new, pure numpy): `constrained_quantize` — b-bit uniform
  quantization onto the probability simplex via largest-remainder (Hamilton apportionment),
  exactly preserving `codes.sum(axis=1) == 2**b - 1` per row (paper eq. 10); reduces exactly
  to argmax one-hot at b=1 (eq. 11), verified rather than special-cased. `dequantize`,
  `encode_bytes` (delta + zlib, mirrors `postprocess.py`'s pattern), `codes_to_bytes`/
  `codes_from_bytes` for the `FitIns.config`/`parameters` transport channels.
- **Server `CFDHook`** (`core/strategy_hooks/cfd.py`): holds a persistent `server_model`
  (the hook is a long-lived object, unlike ephemeral simulated clients) refined every round
  via dual distillation on the aggregated client soft-labels; its predictions become next
  round's downstream target. `pre_aggregate_fit` returns the `(None, {})` tuple (not bare
  `None`) to bypass FedAvg weight-averaging the int64 soft-label codes. Round 1 skips the
  downstream broadcast (server model untrained, no delta reference) but clients still upload
  soft-labels, seeding dual distillation from round 1.
- **Client `CFDFit`** (`core/client_hooks/cfd.py`): fresh model init each round (paper's
  design — clients hold no persistent state); digests server labels via KL (round >= 2, gated
  like FedMD), trains private CE, returns quantized soft-label codes as `parameters` (not
  weights). Only the upstream delta-reference codes persist, via `client.state`
  (`Context.state`, now threaded into `GenericClient` itself, not just
  `get_compressor_hook`).
- **Circular-import fix:** `softlabel_codec.py` lives under `core/`, not `baselines/`
  (deviating from the original plan) — `client_hooks/cfd.py` needing it from `baselines/`
  would import that package's `__init__.py`, which needs `fedmaq.core.client.CompressionHook`,
  while `core.client` is still mid-import via its own `client_hooks` import. The codec has no
  `fedmaq` dependencies, so `core/` sidesteps the cycle cleanly.
- **Fidelity caveats (documented in-code):** zlib substitutes for CABAC/arithmetic coding;
  public proxy is the 1600-sample FedMAQ pool, not the paper's ~80k-sample STL-10.
- **Tests:** `tests/test_cfd.py` (new, 11 tests: quantizer math, delta/zlib codec, server
  dual-distillation across 2 rounds, FedAvg-bypass, client round-gating, upstream delta-state
  persistence); `tests/test_environment.py` (registry test now constructs `CFDHook` instead of
  asserting `NotImplementedError`); `tests/test_simulation.py` (2-round `run(cfg)` smoke, like
  FedDistill's); `tests/test_timing_golden.py` (2 new goldens for the client digest-phase
  timing contribution). 78 tests green, ruff clean, mypy no new errors. CPU end-to-end smoke
  on CIFAR-10 (2 rounds, 6 clients) completes and produces finite eval accuracy.
- **Docs:** `baseline_registry.md` row 14 → `[Complete]`; `HANDOFF.md` P11 → `[x]`.

### 2026-07-09 — Literature OKF Restructure (literature)

Branch `okf-restructure` in `fedmaq-literature`. Restructured the literature repo from a local-first vector-RAG pipeline into an **Open Knowledge Format** knowledge graph, chosen over Karpathy's LLM-Wiki pattern for long-term LLM-agent integration and conformant to the user-provided `SPEC.md` (OKF v0.1). Executed in phases; the advisor was consulted at phase boundaries.

- **Decision + scope (three approved calls):** Remove the Chroma/LlamaIndex vector-RAG stack entirely (keep only Docling/Marker PDF→markdown conversion); Phase 1 = papers only (Method/Concept/Finding/Gap dirs scaffolded but empty); drop the draft→approve gate (nodes authored directly, reviewed via `git diff`).
- **Two-layer architecture:** raw layer `markdown/{slug}/paper.md` (verbatim, citable) + knowledge layer `kg/` (curated OKF nodes with YAML frontmatter, root-absolute intra-bundle links, `index.md`/`log.md` reserved files). No vector DB — grep + read replaces retrieval.
- **Conversions:** Confirmed 10 of 39 canon papers were unconverted (acar, karimireddy, li-moon, tan, wang-fednova, alistarh, bernstein, jeong, lin, zhu); converted all via the pipeline. Fixed the recursive PDF resolver (`resolve_pdf_path` non-recursive `glob` → `rglob`, since PDFs live in batch subfolders).
- **Migration:** Built a reproducible generator (`scripts/build_kg_papers.py` + hand-authored `scripts/kg_bodies/`, kept as provenance) emitting all **39** `type: Paper` nodes to `kg/papers/{slug}.md` (28 migrated from prior summaries + 11 hand-authored from `paper.md`). Fixes along the way: YAML-scalar quoting for titles/descriptions with colons; PDF-filename-derived titles + a `TITLE_OVERRIDES` map for Windows-dropped colons/truncation; batch-tag recovery; CJK cleanup; method-trigger cross-linking that avoids fabricating edges from shared surnames. All nodes OKF-conformant, 116 intra-bundle links resolve.
- **Stack removal:** Dropped `chromadb`/`llama-index-*`/`openai` from `pyproject.toml`; removed `ingest/`, `workflows/`, `summaries/`, `syntheses/`, `storage/` and their tests; rewrote the CLI to `convert` + `list-slugs` only.
- **Docs/rules/skills (Phase 4):** Retargeted `AGENTS.md`/`README.md` to the bundle. Rules: added `kg-conventions.mdc` (replaces `rag-boundaries`), `okf-paper-template.mdc` (replaces `summary-template`), `okf-finding-template.mdc` (replaces `synthesis-template`); updated `naming-conventions`/`no-pdf-read`. Skills: added `convert-paper` (replaces `ingest-paper`) and `author-node` (replaces `summarize-paper`); removed `query-literature` and `approve-summary`. Slimmed `paper_registry.md` to conversion status (dropped dead Indexing/Summary columns) with matching `registry.py` + test changes. 14 tests green. (Literature remains on `.cursor/` tooling — not yet migrated to `.claude/`.)
- **Cross-repo:** Updated `HANDOFF.md` (workspace map, locked decisions, per-repo status, queue, env vars, what-not-to-do, recommendation) to reflect the restructure and to mark the now-resolved experiments-refactor (PR #1) and manuscript `chapter_4.tex` (`b6c08bc`) threads.

### 2026-07-09 — Deep Cleanup/Refactor + FedDistill+ Port (experiments)

Branch `refactor/cleanup-and-feddistill` (10 commits, not yet merged to `main`). 54 tests green, ruff clean, mypy non-blocking. Executed a 6-phase plan; the advisor was consulted at each phase boundary.

- **Phase 0 — safety net + tooling:** Extracted a decorator-free `run(cfg)` into `src/fedmaq/simulation.py` (thin `scripts/run.py` wrapper) so the real `client_fn`/`server_fn`/`evaluate_fn` wiring is testable in-process. Added `tests/test_simulation.py` (Hydra composition of all 14 algorithm configs + `run(cfg)` smoke) and `tests/test_timing_golden.py` (golden characterization of the simulated time model). Moved `scikit-learn` to runtime deps; added mypy + `[tool.mypy]`; widened ruff to `B`; added `.pre-commit-config.yaml` and `.github/workflows/ci.yml` (ruff blocking, mypy non-blocking, pytest local-only due to the cu132 torch pin).
- **Phase 1 — correctness (each with a test):** FedMAQ gradient norms now computed on the KD-student architecture — the probe used `get_model()` (SimpleCNN/ResNet18GN) while the global/client model is the KD student, so on CIFAR a tensor-count mismatch was swallowed by a bare `except` and zeroed every client's `tilde_g`, silently disabling gradient-awareness. 1-bit FedPAQ is now valid sign-quantization (was `(0/0)*scale` = NaN, reachable via the Tier-1 memory cap). DAdaQuant per-client `q` clamped to `[q_min, q_max]`. `set_model_parameters` raises on count/shape mismatch instead of silently truncating.
- **Phase 2 — dedup into `core/`:** `kd_utils.distill_ensemble_into_global` (shared FedMAQ/FedAvgKD server-KD body), `strategy_hooks/_partition.py` (`resolve_partition_id` + `partition_dataset_size`), `compression.svd_compressed_nbytes`, and `quantization._quantize_deltas` (shared FedPAQ/DAdaQuant skeleton).
- **Phase 3 — deep architectural refactor (removed all `alg_name` string-dispatch):** Client-side `ClientFitStrategy` hooks in `core/client_hooks/` (Standard/DAdaQuant/FedMAQ/FedMD/FedKD; `client.py` 518->125 lines). Server-side: new `StrategyHook` methods `download_size_bytes`, `compute_speed_scale`, `local_train_sample_count`, `server_sim_time` replace the `alg_name` branches in `strategy.py`/`NetworkSimulator`; magic numbers `2.5`/`2000.0`/pretrain-`10` moved to config keys (`fedkd.compute_penalty`, `{fedmaq,fedavg_kd}.server_compute_speed`, `fedmd.{public,private}_pretrain_epochs`). Dict hook registry with a WARNING fallback and a config-time guard for unported baselines. State hygiene (init `last_round_*` in `__init__`; dropped the dead `strategy.cumulative_bytes` — `TelemetryManager` is the sole owner). Typing/dead-code cleanups.
- **Phase 4 — FedDistill+ port (P10):** FEDDISTILL+ (Zhu et al. 2021 / FedGen) sharing FedAvg weights AND label-wise logits. Client `FedDistillFit` + `LogitTracker` (counts init to ones so missing classes give a finite zero row, not NaN, under Dirichlet 0.1); loss `CE + reg_alpha*KLDiv(log_softmax(z), softmax(global_logits[y]))`. Server `FedDistillHook` averages client logit matrices (FedAvg still averages weights) and rebroadcasts. Logits travel as bytes via `FitRes.metrics`/`FitIns.config`. Deviation from reference: per-round tracker (Flower recreates clients each round), documented. Verified by unit tests + a 2-round `run(cfg)` smoke exercising the reg path.
- **Phase 5 — config hygiene:** `cifar10.yaml` `in_channels: 3`; explicit `bit_widths` on the 4 FedMAQ ablation configs; fixed the broken `conf/experiment/ci.yaml` overlay (was `@package _global_` sending keys the code never reads + a group collision on `+experiment=ci`) — now inherits `default` and is selected with `experiment=ci`.
- **Code review:** Ran a high-effort review pass over the branch; only finding (a redundant partition-ID resolution on the unreachable None-`client_indices_dict` path) was fixed by restoring the short-circuit.
- **Deferred debt (documented, not blocking):** mypy left non-blocking (27 errors, mostly Hydra OmegaConf->dict variance + torch `Dataset.__len__` stubs); DAdaQuant unit-test params differ from `dadaquant.yaml` (reconcile via config-composing tests).

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
