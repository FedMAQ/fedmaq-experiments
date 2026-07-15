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

**FedMAQ has two variants:**

- **FedMAQ-Lite** (`fedmaq_lite`, SimpleCNN ~2.16M params): Smoke-test tuning complete. Best result beats FedProx by +3.12pp with 8.6x comm savings under severe non-IID (α=0.1). These configs are validated.
- **FedMAQ** (`fedmaq`, ResNet18GN ~11.17M params): Partially tuned. Best result is 47.43% (vs. FedProx 49.71%, gap of −2.28pp) with only 1.7x comm savings. Hyperparameters were transferred from SimpleCNN and are likely suboptimal.

A comprehensive algorithm audit was conducted: [docs/audits/fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md). The audit found the core algorithm sound (✅) with some defensible caveats (⚠️). Actionable recommendations: [docs/audits/fedmaq-audit-recos.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit-recos.md).

---

## 3. Critical Decisions — RESOLVED (2026-07-16)

All 13 framing/methodology decisions from the 2026-07-16 grilling session are logged in **[docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)** — read that first. Grid design detail: [docs/plans/formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md).

**Action for next agent**: Continue refining experiment details via grilling, then execute the codebase refactors/changes the plan implies (config-as-code registry, baseline matched-tuning setup, seed-determinism check). See plan §3 for deferred sub-details.

---

## 4. Key Technical Context

### CUDA OOM Mitigation

Hydra `--multirun` causes VRAM accumulation across sequential jobs. All experiments use **process-isolated runner scripts** in `scripts/` that spawn each job as a fresh subprocess. Between runs, the script calls `ray stop` to clean up workers. This is mandatory for all future experiment scripts.

### Experiment Data Paths

Runner scripts output to `experiments/` by default, but the actual data is manually transferred to `multirun/` after completion. Paths referencing `experiments/` in older documents may be outdated. Experiment analysis documents live in `docs/experiments/<experiment-name>/` with `results.md` and `comments.md`.

### Dual-Variant Architecture

Both variants share the same strategy hook (`FedMAQHook`) with dynamic model dispatch:

- `fedmaq`: Clients train ResNet18GN, server KD is self-distillation (ResNet18GN → ResNet18GN)
- `fedmaq_lite`: Clients train SimpleCNN, server KD is cross-architecture (SimpleCNN → SimpleCNN)

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
2. **Matched light tuning** of the 8 baselines (one key HP each) on the same explore-α.
3. **Pre-register + git-tag** the frozen FedMAQ config + baseline HP table + fixed mechanism set. This ends exploration.

### Priority 2: Confirmation Infrastructure

4. **Build the config-as-code registry**: manifest enumerating every formal run (algo × dataset × α × seed), hashed frozen configs, driving process-isolated runners.
5. **Seed-determinism check**: ensure partitions are identical across paired arms (required for the paired test).

### Priority 3: Ablations & Deferred Details

6. **Gradient-norm-smoothing isolation ablation** (`grad_norm_ema=false`) — still owed.
7. **Ablation table** (additive ladder + leave-one-out) on CIFAR-10 at α ∈ {0.1, 1.0}.
8. **Deferred sub-details** — see plan §3 (single-config selection rule, baseline key-HP list, Pareto matched-bit-budget comparison).

### Priority 4: Docs Structure (in progress, 2026-07-16)

9. **Continue docs cleanup**. Done this session: `docs/DECISIONS.md` created as single decision log (killed 3-way duplication across STATUS/HANDOFF/plan); STATUS.md §7 (stale, self-contradicting) removed; `docs/plans/fedmaq-audit-remediation.md` and `docs/plans/client-regularization.md` deleted (completed/superseded, recoverable via git history); all 9 ResNet18GN experiment dirs moved to `docs/experiments/archive/`; audit docs got staleness caveats. **Still open**: decide long-term semantics for `docs/plans/` (active-only vs. mixed), confirm `docs/archive/` vs. per-directory `archive/` subfolders is the right pattern going forward (currently only `docs/experiments/archive/` exists, no top-level `docs/archive/` was created), and check `docs/plans/formal-experiment-plan.md` for further trim once the confirmatory grid is pre-registered (§4 of that doc flags its own structure as tentative).

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
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                         | Canonical glossary (resolves naming drift between repos)                                 |

### Per-Experiment Deep Dives

All 9 smoke-test experiments are ResNet18GN-era and now deprecated (see §3). They're archived under `docs/experiments/archive/<name>/` — see [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md) for the full index. New MobileNetV2GN experiments will land as fresh top-level entries in `docs/experiments/`.
