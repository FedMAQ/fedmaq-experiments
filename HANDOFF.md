# Handoff Context: FedMAQ Experiments

**Purpose**: Focused handoff for the next agent. All historical experiment details, audit findings, and remediation plans live in `docs/`. This document provides orientation, current state, and immediate action items only.

**Last updated**: 2026-07-16

---

## 1. Project Overview

FedMAQ is a communication-efficient federated learning algorithm that uses dual-tier multi-adaptive quantization and knowledge distillation. This is a Master's thesis codebase (Flower, Hydra, PyTorch) implementing FedMAQ and 7 baseline algorithms.

- **Current state document**: [docs/STATUS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/STATUS.md) — best configs, accuracy standings, open decisions.
- **Glossary**: [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md) — canonical terminology shared across 5 repos.
- **Experiment registry**: [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md) — 9 archived ResNet18GN smoke tests + index for new MobileNetV2GN runs.

---

## 2. What Has Been Done (Summary)

Nine exploratory smoke-test experiments (40–50 round, single-seed) were conducted on CIFAR-10 between July 13–15. **These are not formal thesis results** — they were run to validate the algorithm direction and identify which hyperparameters matter. See the full chronological table in [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md).

**FedMAQ has two variants (ResNet18GN-era naming; see below for current status):**

- **FedMAQ-Lite** (`fedmaq_lite`, SimpleCNN ~2.16M params): Smoke-test tuning complete on ResNet18GN-era baselines. Best result beat FedProx by +3.12pp with 8.6x comm savings under severe non-IID (α=0.1). **Dropped from the formal thesis** (DECISIONS.md Decision 4) — its size-contrast story is confounded now that main FedMAQ is also ~2.24M params (MobileNetV2GN). Smoke results live on as an exploration-appendix reference only.
- **FedMAQ** (`fedmaq`): Now trains **MobileNetV2GN** (~2.24M params, switched 2026-07-15 for edge realism — DECISIONS.md Decision 1), not ResNet18GN. Best ResNet18GN-era result was 47.43% (vs. FedProx 49.71%, gap of −2.28pp) with only 1.7x comm savings; hyperparameters were transferred from SimpleCNN and are likely suboptimal. No MobileNetV2GN results exist yet — this is the active formal-thesis variant.

A comprehensive algorithm audit was conducted: [docs/audits/fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md). The audit found the core algorithm sound (✅) with some defensible caveats (⚠️). Actionable recommendations: [docs/audits/fedmaq-audit-recos.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit-recos.md).

A separate **code-level** audit (craftsmanship + FL engineering) followed: [docs/audits/fedmaq-code-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-code-audit.md). Findings F2/F4–F8 fixed on branch `fix/code-audit-findings` ([PR #5](https://github.com/FedMAQ/fedmaq-experiments/pull/5), **merged**); F1 accepted (wontfix-thesis); **F9 deferred** — see Priority 1 caveat below. No experimental results are affected (behavior-neutral cleanups + telemetry/robustness only).

A follow-on **architecture** pass (branch `feat/architecture`, off merged `main`) is in progress — see §5.5.

---

## 3. Critical Decisions — RESOLVED (2026-07-16)

All 13 framing/methodology decisions plus 4 docs-management decisions from 2026-07-16 grilling sessions are logged in **[docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)** — read that first. Grid design detail: [docs/plans/formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md).

**Action for next agent**: Continue refining experiment details via grilling, then execute the codebase refactors/changes the plan implies (config-as-code registry, baseline matched-tuning setup, seed-determinism check). See plan §3 for deferred sub-details.

---

## 4. Key Technical Context

### CUDA OOM Mitigation

Hydra `--multirun` causes VRAM accumulation across sequential jobs. All experiments use **process-isolated runner scripts** in `scripts/` that spawn each job as a fresh subprocess. Between runs, the script calls `ray stop` to clean up workers. This is mandatory for all future experiment scripts.

### Experiment Data Paths

Runner scripts output to `experiments/` by default, but the actual data is manually transferred to `multirun/` after completion. Paths referencing `experiments/` in older documents may be outdated. Experiment analysis documents live in `docs/experiments/<experiment-name>/` with `results.md` and `comments.md`.

### Dual-Variant Architecture

Both variants share the same strategy hook (`FedMAQHook`) with dynamic model dispatch:

- `fedmaq`: Clients train MobileNetV2GN (default CIFAR model), server KD is self-distillation (MobileNetV2GN → MobileNetV2GN)
- `fedmaq_lite`: Clients train SimpleCNN, server KD is self-distillation (SimpleCNN → SimpleCNN) — dropped from formal thesis, exploration-appendix only (DECISIONS.md Decision 4)

The model factory selection is driven by algorithm name in [models.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/models.py) → `get_client_model()`.

### Key Novel Findings

1. **Capacity-EMA Duality**: Student EMA helps small models but _hurts_ large models under severe skew. See [archive/fedmaq-normal-no-ema-50r/comments.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/fedmaq-normal-no-ema-50r/comments.md).
2. **Heterogeneity-EMA Inverse Relationship**: Optimal β is strongly α-dependent (β=0.7 for α=0.1, β=0.1 for α=1.0).
3. **SimpleCNN→ResNet18GN Hyperparameter Transfer Failure**: `entropy_weight=4.0` caused voter exclusion on ResNet18GN under severe skew. Other transferred params may be similarly suboptimal.

---

## 5. Immediate Next Actions for the Next Agent

> Framing/grid decisions are **already resolved** (§3, and [formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md)). Remaining work is refinement + implementation.

### Priority 1: Exploration Phase (MobileNetV2GN)

1. **Run the adaptive exploration campaign** on CIFAR-10 (primary): re-sweep soft-voting (`entropy_weight` × `precision_weight`), validate Formulation 3, resolve capacity-EMA (on/off for MobileNetV2 — open question), client KD reg. Mechanisms are guides — keep/drop/revise per results.
   - **Before enabling `client_kd_reg=true`** (code-audit F9): `ClientKDLossHook.on_train_begin` deep-copies the whole client model **every round** because Flower/Ray re-instantiates `client_fn` per round. Persist the reference model in `context.state` first if the KD-reg sweep proves hot, otherwise it pays a full-model deepcopy per client per round.
2. **Matched light tuning** of the 8 baselines (one key HP each) on the same explore-α.
3. **Pre-register + git-tag** the frozen FedMAQ config + baseline HP table + fixed mechanism set. This ends exploration.

### Priority 2: Confirmation Infrastructure

4. **Build the config-as-code registry**: manifest enumerating every formal run (algo × dataset × α × seed), hashed frozen configs, driving process-isolated runners.
5. **Seed-determinism check** ✅ **DONE**: partition generation is a pure function of `(dataset, num_clients, alpha, num_public_samples, seed)` with **no algorithm input** (single call site `simulation.py:124` passes the global `cfg.experiment.num_public_samples`, not a per-algorithm value), so paired arms at a matched seed see byte-identical partitions by construction. Locked by regression test `test_partition_seed_invariant_for_paired_arms` (`tests/test_environment.py`): regenerates from scratch into separate cache dirs (proving generation determinism, not just the JSON cache round-trip the older test covered) — same seed identical, distinct seed differs. Together with the deterministic ClientManager this closes the reproducibility oracle (data partition + client sampling + per-worker training all seed-pinned).

### Priority 2.5: Architecture Branch (`feat/architecture`) — in progress

Behavior-changing improvements made *before* the formal baseline freezes (safe window; no formal runs started). Landed commits (all tested, 92 green):

- **Determinism (partial)**: torch RNG + deterministic-kernel flags now re-pinned **per Ray worker** inside `client_fn` (Ray actors don't inherit driver flags); DataLoader `generator`/`worker_init_fn` seeded; `experiment.strict_determinism` knob (default true). Training is now **bit-identical given a fixed sampled client**.
- **Phase 2** — centralized server model-factory dispatch (`models.get_server_model_factory`), removing the duplicated inline rule in `FedMAQHook`.
- **Phase 4** — centralized cross-hook fallback defaults (`core/config_defaults.py`) with a regression test; flagged two stale fallbacks (`num_public_samples` 200 vs conf 3000; `weight_decay` 0.0 vs conf 1e-4) left inline rather than enshrined.
- **Phase 5** — split `FedMAQHook.configure_fit` god-method into named helpers + `_QuantParams` (F8 fail-loud preserved).

**Deferred — do these before the confirmatory grid (they gate Priority 2's seed-determinism):**

- **Deterministic ClientManager** ✅ **DONE** (`core/client_manager.py`, `SeededPartitionClientManager`): samples by **partition-id** with a per-round-seeded RNG, resolving each `ClientProxy` node-id → partition-id via `get_properties` (cached) and waiting for the full population before drawing. Wired into `server_fn` via `ServerAppComponents(client_manager=...)`; `TelemetryFedAvg.configure_fit` calls `set_round_seed(server_round)` before sampling. **Measured**: same config run twice at `client_fraction=0.1` now yields bit-identical per-round selection (was nondeterministic — Flower's default draws global-`random` over timing-ordered node-ids). Note: `ClientProxy.cid` is a random node-id in flwr 1.32 sim, **not** the partition-id — partition-id is only recoverable via `get_properties`. Regression test: `tests/test_client_manager.py`.
- **Phase 3 = Priority 2 step 4** (config-as-code run registry): still owed; must encode `post_process=true` for primary-grid FedMAQ cells (headline §4.3 comm mechanism; default is `false`, a footgun the registry must own).
- **Phase 6** (optional, only if cheap): decouple `strategy.py` `isinstance(DAdaQuantHook)` proxies; cross-layer imports.

Branch is unpushed — ready to push + PR when the above scope decisions are settled.

### Priority 3: Ablations & Deferred Details

6. **Gradient-norm-smoothing isolation ablation** (`grad_norm_ema=false`) — still owed.
7. **Ablation table** (additive ladder + leave-one-out) on CIFAR-10 at α ∈ {0.1, 1.0}.
8. **Deferred sub-details** — see plan §3 (single-config selection rule, baseline key-HP list, Pareto matched-bit-budget comparison).

### Priority 4: Docs Structure (resolved 2026-07-16)

9. **Docs cleanup — conventions now settled**, logged as decisions 14–17 in `docs/DECISIONS.md`: single canonical experiment registry (`docs/experiments/README.md`; `.claude/project/experiment_registry.md` deleted), `docs/plans/` is active-only (delete on resolution, merge into `DECISIONS.md`), archive pattern is per-directory `archive/` subfolder (no top-level `docs/archive/`). Enforcement now lives in [.claude/rules/docs-management.md](.claude/rules/docs-management.md) (always-loaded conventions) and the `docs-audit` skill (on-demand full sweep, auto-fixes mechanical drift).
10. **Next agent: run the `docs-audit` skill** as a fresh-eyes review of the whole docs system (STATUS.md, DECISIONS.md, plans, experiments registry, HANDOFF.md itself) against the newly settled conventions — confirm nothing was missed in this session's fixes, and check `docs/plans/formal-experiment-plan.md` for further trim once the confirmatory grid is pre-registered (§4 of that doc still flags its own structure as tentative).

---

## 6. Reference Documents

| Document                                                                                                                                      | What It Contains                                                                         |
| :-------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)                                           | **Start here.** The 13 resolved decisions (2026-07-16 grilling)                          |
| [docs/plans/formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md) | Formal grid design, mechanisms under deliberation, deferred sub-details                  |
| [docs/STATUS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/STATUS.md)                                                 | Current best configs, accuracy standings, remaining work                                 |
| [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)                         | Chronological experiment registry with per-experiment links                              |
| [docs/audits/fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md)                       | Full algorithm audit (math, implementation, literature, defense Q&A)                     |
| [docs/audits/fedmaq-audit-recos.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit-recos.md)           | Actionable audit recommendations with priority table                                     |
| [docs/audits/fedmaq-code-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-code-audit.md)             | Code-level audit (craftsmanship + FL engineering); resolution status in summary table    |
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                         | Canonical glossary (resolves naming drift between repos)                                 |

### Per-Experiment Deep Dives

All 9 smoke-test experiments are ResNet18GN-era and now deprecated (see §3). They're archived under `docs/experiments/archive/<name>/` — see [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md) for the full index. New MobileNetV2GN experiments will land as fresh top-level entries in `docs/experiments/`.
