# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                         |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-10                                                                    |
| **Last session focus** | Codebase-consistency audit + fixes in `fedmaq-experiments` against the finalized manuscript (branch `fix/codebase-manuscript-consistency`, not yet PR'd): `c_unit` 2048→512 MB (5 yamls), combine-then-floor-once bit-width snap logic (was independently nearest-snapping Tier-1/Tier-2 then min'ing, could violate the memory cap), and a new post-processing pipeline (error-feedback + diff-coding + lossless zlib) for FedMAQ's winning formulation on the primary CIFAR-10/100 + FEMNIST grid only (`src/fedmaq/baselines/postprocess.py`, gated via `post_process` in all 14 algorithm yamls). 73 tests green (9 new), full CPU smoke run pending verification. |
| **Active repo**        | fedmaq-experiments (open PR for `fix/codebase-manuscript-consistency`, then back to fedmaq-manuscript: final polishing pass — task detail incoming from user) |
| **Blockers**           | None                                                                          |

---

## 1. Quick start (new agent)

1. Open the **multi-root workspace** with all five `fedmaq-*` repos.
2. Read this file end-to-end, then the **active task** in [Section 6](#6-implementation-queue).
3. Load domain rules from [`fedmaq-experiments/.claude/rules/`](.claude/rules/) (canonical thesis context).
4. Work in **one primary repo** per task; use the "Agent entry" column in the workspace map below for entrypoints (`CLAUDE.md` for experiments/manuscript, `AGENTS.md` for unmigrated siblings).
5. Before ending a session, update this handoff file with changelog entries and recommendations for clean context.

**Candidate:** Christian Joseph Bunyi | **Institution:** De La Salle University | **Advisor:** Fritz Flores

---

## 2. Workspace map

| Repo                                             | Role                                                                 | Agent entry                                    | Domain rules                       |
| ------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------- |
| [fedmaq-experiments](../fedmaq-experiments/)     | FedMAQ code, Hydra, Flower, WandB                                    | [CLAUDE.md](../fedmaq-experiments/CLAUDE.md)   | **Owner:** `.claude/rules/`        |
| [fedmaq-literature](../fedmaq-literature/)       | PDFs, markdown conversions, OKF knowledge graph (`kg/`)              | [CLAUDE.md](../fedmaq-literature/CLAUDE.md)    | `thesis-context.md` → experiments  |
| [fedmaq-analyses](../fedmaq-analyses/)           | Notebooks, thesis figures                                            | [AGENTS.md](../fedmaq-analyses/AGENTS.md)      | `thesis-context.mdc` → experiments |
| [fedmaq-manuscript](../fedmaq-manuscript/)       | LaTeX thesis (Active; Ch 1–6 drafted, de-overclaim pass applied)    | [README.md](../fedmaq-manuscript/README.md)    | **Owner:** `.claude/rules/`        |
| [fedmaq-presentations](../fedmaq-presentations/) | Beamer slides                                                        | [AGENTS.md](../fedmaq-presentations/AGENTS.md) | `thesis-context.mdc` → experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content. Index via `../fedmaq-experiments/.claude/rules/`.

**Agent tooling layout (mixed state):** `fedmaq-experiments`, `fedmaq-manuscript`, and `fedmaq-literature` have migrated to Claude Code — each owns `.claude/rules/`, `.claude/project/` (experiments also has `.claude/skills/` and `.claude/commands/`; literature entrypoint is `CLAUDE.md` and its `thesis-context.md` points at `fedmaq-experiments/.cursor/rules/`, still stale). `fedmaq-analyses` and `fedmaq-presentations` have not migrated yet and still use `.cursor/rules/`, `.cursor/skills/`, `.cursor/project/`, with `thesis-context.mdc` pointers to `fedmaq-experiments/.cursor/rules/` that are likewise stale until each repo's own migration pass. No shared parent config directory across repos (may add later).

---

## 3. Locked architectural decisions

| Topic              | Decision                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| Thesis context     | `fedmaq-experiments/.claude/rules/` (decomposed from `context.md`; `context.md` is human snapshot only) |
| Experiments layout | uv monorepo, code under `src/fedmaq/core/` and `src/fedmaq/baselines/`                                  |
| Tooling            | Preferred stack in `tech-stack.md`; adopt extra libs (pandas, sklearn, etc.) when justified             |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                         |
| Literature KG      | OKF bundle at `kg/` (see `fedmaq-literature/SPEC.md`); two layers — raw `markdown/` (citable) + curated OKF nodes. No vector store (grep + read); nodes authored directly, no approve gate (review via `git diff`); no cross-repo auto-edits |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                          |

---

## 4. Per-repo status

### [fedmaq-experiments](../fedmaq-experiments/) — [Deep-refactored, hook-based; FedDistill+ ported]

- **Status details:** See completed baselines in [baseline_registry.md](.claude/project/baseline_registry.md). This session (branch `refactor/cleanup-and-feddistill`, not yet merged to `main`) removed all `alg_name` string-dispatch: client-side local training now goes through `ClientFitStrategy` hooks (`core/client_hooks/`), and server-side time/comms modelling through new `StrategyHook` methods (`download_size_bytes`, `compute_speed_scale`, `local_train_sample_count`, `server_sim_time`) — `strategy.py`/`NetworkSimulator` carry no per-algorithm branches. Magic numbers (fedkd `2.5`, server KD `2000.0`, fedmd pretrain `10`) moved to config. Deduped KD/partition/SVD/quantization helpers into `core/`. Correctness fixes: FedMAQ grad-norm now on the KD-student architecture (was silently zeroed on CIFAR), 1-bit FedPAQ sign-quantization (was NaN), DAdaQuant per-client `q` clamp, `set_model_parameters` shape guard. **FedDistill+ ported** (FedAvg weights + label-wise logit KD; `core/{client_hooks,strategy_hooks}/feddistill.py`). Safety net added: `simulation.run(cfg)`, composition + golden time/bytes tests, 54 tests green, ruff clean.
- **Pending:** `refactor/cleanup-and-feddistill` **merged to `main`** (PR #1, `7b05f17`). Port CFD baseline (P11, ~Oct 2026 — still a config-time-guarded stub). Docker integration. Minor debt: mypy non-blocking (27 errors, mostly Hydra OmegaConf->dict variance + torch `Dataset.__len__` stubs); DAdaQuant unit-test params (phi=3,q_min=4) differ from `dadaquant.yaml` (phi=5,q_min=1) — reconcile by composing real configs in tests. **P19 (this session, branch `fix/codebase-manuscript-consistency`, not yet PR'd):** `c_unit` corrected to 512 MB (was 2048) across `fedmaq.yaml` + 4 ablation-arm yamls; `compute_fedmaq_q_k_t` refactored from independent nearest-snap-then-min to combine-raw-then-floor-once (matches manuscript §4.2; regression test added for the divergence case, e.g. q_hat=13 now floors to 8 instead of rounding to 16); new `src/fedmaq/baselines/postprocess.py` (`FedMAQPostProcessCompressionHook`) implements error-feedback + diff-coding + real zlib-measured bytes for the primary CIFAR-10/100 + FEMNIST grid, gated off by default everywhere else via `post_process` (14 algorithm yamls); `Context.state` threaded through `simulation.py` → `get_compressor_hook`. 73 tests green (64 pre-existing + 9 new in `tests/test_postprocess.py`).

### [fedmaq-manuscript](../fedmaq-manuscript/) — [Active — next task]

- **Completed:** Chapter 1–6 drafted (Ch5 is largely results placeholders awaiting the run grid; Ch6 has content with `[PLACEHOLDER]` slots). **Grilling-pass de-overclaim revision** applied across Ch1–6 on branch `manuscript-deoverclaim-communication-primary` (PR pending, compiles clean, 65 pp): communication-primary problem statement with memory as a hard feasibility *ceiling*; two-axis objectives; central §3.5 rewrite crediting quantization-noise attenuation to unbiased **parameter averaging** (~1/K_active) and KD to non-IID **drift reconciliation** (not noise cancellation — Jensen/nonlinear-softmax), with a falsifiable α-prediction wired to the ablation; unbiased-quantizer note in §3.3; "pilot study" → "formulation study" across both α∈{0.1,1.0}; **195-run grid** (108 main + 27 FEMNIST + 30 formulation + 30 ablation); DynFed-style reference arm; hardware reframed (lab datacenter primary, RTX 5060 dev/smoke); removed "mathematically unifying"/"resolves the conflicting demands" over-claims. The earlier flagged experimental-grid-size note is now reconciled to 195 runs.
- **Pending (NEXT TASK):** **Final polishing pass over the manuscript** — user will detail scope for the next agent. Ch5/Ch6 still hold `[PLACEHOLDER]` slots that fill once results land. Incorporate proposal-panel feedback post-defense.

### [fedmaq-literature](../fedmaq-literature/) — [OKF bundle fully populated]

- **Status details:** OKF bundle at `kg/` (two layers: raw `markdown/{slug}/paper.md` citable + curated nodes). **Knowledge layer now fully populated (87 nodes):** 39 papers, 24 methods, 10 concepts, 8 findings, 6 gaps (Phases 2–3 merged to `main`, PR #2). Migrated to Claude Code (`.claude/rules|skills|project/`; `CLAUDE.md` is the entrypoint). Node authoring is direct-to-`kg/`, reviewed via `git diff`; no vector store (grep + read). This session's **grilling de-overclaim** revised 6 gap/finding nodes + `log.md` on branch `kg-deoverclaim-multisignal` (PR pending) — retired the "round × client × layer" framing for the multi-signal combination contribution; FedMAQ now explicitly does **not** close the skew-aware-precision gap (distillation carries the statistical-het load).
- **Pending:** None blocking. Deferred polish: conversion-pipeline QA tuning; math rendering in converted bodies.

### [fedmaq-analyses](../fedmaq-analyses/) — [Scaffold complete]

- **Status details:** See active figures and notebook status in [figure_registry.md](../fedmaq-analyses/.cursor/project/figure_registry.md).
- **Pending:** WandB/Hydra ingest implementations, Real ablation + thesis figure notebooks.

### [fedmaq-presentations](../fedmaq-presentations/) — [Complete]

- **Status details:** Slide mapping and metadata updated in [slide_registry.md](../fedmaq-presentations/.cursor/project/slide_registry.md).
- **Pending:** None.

---

## 5. Literature knowledge-graph reference

For the two-layer structure (raw `markdown/` + OKF `kg/`), the conversion
pipeline, and how agents traverse the bundle, refer to
[fedmaq-literature/README.md](../fedmaq-literature/README.md) and
[AGENTS.md](../fedmaq-literature/AGENTS.md). The format itself is specified in
[fedmaq-literature/SPEC.md](../fedmaq-literature/SPEC.md); bundle conventions live
in `fedmaq-literature/.cursor/rules/kg-conventions.mdc`. There is no vector
store — retrieval is grep + read over `markdown/` and `kg/`.

---

## 6. Implementation queue

Priority order for upcoming work. Mark items `[x]` when done; add new items at the bottom with date.

| P   | Task                                                       | Repo        | Status |
| --- | ---------------------------------------------------------- | ----------- | ------ |
| 1   | Implement PDF convert (Docling + Marker QA)                | literature  | [x]    |
| 2   | LlamaIndex + Chroma ingest with Qwen3-4B _(removed in OKF restructure)_ | literature  | [x]    |
| 3   | `fedmaq-lit` summarize + approve + OpenRouter _(removed in OKF restructure)_ | literature  | [x]    |
| 4   | Phase 1 FL environment (data partition, bandwidth, Flower) | experiments | [x]    |
| 5   | FedAvg / FedProx / FedPAQ / DAdaQuant baselines            | experiments | [x]    |
| 6   | Full manuscript audit + codebase hardening (Ch. 1--4)      | experiments | [x]    |
| 7   | WandB + Hydra ingest utilities                             | analyses    | [ ]    |
| 8   | Review & approve remaining 10 draft summaries (remediate) _(summaries removed; content re-authored as OKF nodes in Task 14)_ | literature  | [x]    |
| 9   | Consolidate cross-paper findings (OKF Phase 2: `kg/findings/`) _(reframed from the removed `syntheses/` workflow)_ | literature  | [ ]    |
| 10  | Port FedDistill baseline                                   | experiments | [x]    |
| 11  | Port CFD baseline                                          | experiments | [ ]    |
| 13  | Deep cleanup/refactor: hook-based client/server dispatch, dedup, correctness fixes | experiments | [x] |
| 12  | Align experiments code + Ch. 1/4 manuscript (proxy pool, discrete bit-widths, FedDistill spec); apply manuscript-side fixes to chapter_4.tex | experiments + manuscript | [x] |
| 14  | Restructure literature to OKF: remove vector-RAG, convert 10 missing papers, migrate 39 papers into `kg/` bundle | literature | [x] |
| 15  | Populate OKF knowledge layer (methods, concepts, findings, gaps → 87 nodes) | literature | [x] |
| 16  | Grilling-pass de-overclaim: KG gap/finding nodes + manuscript Ch1–6 (multi-signal framing, communication-primary, two-stage aggregation, 195-run grid) | literature + manuscript | [x] |
| 17  | **Final polishing pass over the manuscript** (scope detail incoming from user)   | manuscript  | [ ] |
| 18  | Section B — align experiments code: switch FedPAQ quantizer to **stochastic rounding** (reuse `DAdaQuantCompressionHook._quantize_elem`) + wire 195-run/both-α ablation. Hard dependency for the α-prediction. | experiments | [ ] |
| 19  | Codebase-consistency fixes: `c_unit` 512, combine-then-floor-once bit-width snap, post-processing pipeline (error-feedback + diff-coding + zlib) gated by `post_process` on the primary grid only | experiments | [x] |

**Current focus:** The grilling-pass de-overclaim (Task 16) is complete across
both repos, each on its own branch with a **PR pending** (`kg-deoverclaim-multisignal`
in literature, `manuscript-deoverclaim-communication-primary` in manuscript — open
via the compare links; `gh` CLI is not installed on this machine). Next substantive
work is **P17 — final polishing pass over the manuscript**; the user will detail the
scope for the next agent. Also open: **P18** (Section B — stochastic-quantizer switch
+ both-α ablation in experiments; hard dependency for the α-prediction, deferred to
its own experiments-scoped session), **P7** (WandB + Hydra ingest, analyses), and
**P11** (CFD, ~Oct 2026 stub).

---

## 7. Environment and secrets

| Variable                      | Used by               | Notes                                         |
| ----------------------------- | --------------------- | --------------------------------------------- |
| `FEDMAQ_QA_MIN_MEAN_GRADE`    | literature            | Default `good` (Docling mean_grade threshold) |
| `FEDMAQ_QA_MIN_LOW_GRADE`     | literature            | Default `fair`                                |
| `FEDMAQ_MARKER_DEVICE`        | literature            | Override Marker device (`cuda` / `cpu`)       |
| `HF_HUB_DISABLE_SYMLINKS`     | literature            | Set automatically on Windows in Docling path  |
| `WANDB_API_KEY`               | experiments, analyses | Experiment tracking                           |

The literature RAG variables (`OPENROUTER_API_KEY`, `FEDMAQ_EMBED_*`) were retired
with the vector-RAG stack in the OKF restructure; only the conversion-QA and
Marker vars above remain.

Create `.env` locally (gitignored); document new vars here when added.

**Setup:** `uv sync` in each Python repo; `uv sync --extra dev` for pytest in experiments.

---

## 8. Agent conventions

- **Registries** (`.claude/project/*.md` in experiments/manuscript; `.cursor/project/*.md` in unmigrated siblings): update when adding baselines, papers, figures, or runs.
- **Rules** (`.claude/rules/*.md` in experiments/manuscript; `.cursor/rules/*.mdc` in unmigrated siblings): concise; one concern per file; experiments owns domain rules.
- **Skills** (`.claude/skills/*/SKILL.md` in experiments; `.cursor/skills/*/SKILL.md` in unmigrated siblings): procedural workflows; prefer skills over ad-hoc commands.
- **Commands** (`.claude/commands/*.md` in experiments): slash-command workflows, e.g. `/add-baseline`, `/align-manuscript`, `/run-benchmark`.
- **No emojis** in repo files (`repo-preferences.md`).
- **MCP:** context7 (library docs), GitHub (issues/PRs) — user-level, if configured.

---

## 9. What not to do

- Parse `papers/*.pdf` directly in chat; reason over `kg/`, cite `markdown/`.
- Hand-edit `markdown/{slug}/paper.md` (pipeline output) or reintroduce a vector store.
- Let literature agents edit experiments/analyses/manuscript code directly.
- Duplicate thesis domain content outside `fedmaq-experiments/.claude/rules/`.
- Add top-level `reproductions/` packages; use `src/fedmaq/baselines/`.

---

## 10. Changelog

- Full session-to-session history lives in [changelog.md](.claude/project/changelog.md) (includes the 2026-07-09 Cursor→Claude Code migration and CLAUDE.md relocation, plus the 2026-07-03 hardening/refactor sessions).

---

## 11. Handoff recommendation

**Recommend handoff.** The grilling-pass de-overclaim (P16) is complete and
self-contained across two branches, each with a PR pending: `kg-deoverclaim-multisignal`
in `fedmaq-literature` (6 gap/finding nodes + `log.md` retitled to the multi-signal
combination framing) and `manuscript-deoverclaim-communication-primary` in
`fedmaq-manuscript` (Ch1–6 communication-primary reframing; compiles clean, 65 pp).
Both are pushed and ready to merge at the user's discretion — open the PRs via the
GitHub compare links (`gh` is not installed locally). This is a clean stopping point;
initiate a new session to clear context.

**Next task — P17: final polishing pass over the manuscript** (`fedmaq-manuscript`).
The user will detail the scope for the next agent, so a manuscript-scoped session is
the right entrypoint (read `fedmaq-manuscript/README.md` + `.claude/rules/`). Context
the next agent needs: the de-overclaim branch above should be merged first (or the
polishing rebased onto it) so the two passes don't conflict; Ch5/Ch6 still carry
`[PLACEHOLDER]` slots that fill once the 195-run grid produces results. Also open when
convenient: **P18** (Section B — stochastic-quantizer switch + both-α ablation in
`fedmaq-experiments`; its own experiments-scoped session), **P7** (WandB + Hydra
ingest, analyses), and **P11** (CFD, ~Oct 2026 stub).
