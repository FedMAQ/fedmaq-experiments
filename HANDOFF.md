# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                         |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-09                                                                    |
| **Last session focus** | literature: OKF restructure — removed the Chroma/LlamaIndex vector-RAG stack, converted the 10 missing canon papers, and migrated all 39 papers into an Open Knowledge Format bundle (`kg/`, two-layer: raw `markdown/` + curated OKF nodes). Retargeted docs/rules/skills and slimmed the registry. Branch `okf-restructure`. |
| **Active repo**        | fedmaq-literature                                                             |
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
| [fedmaq-literature](../fedmaq-literature/)       | PDFs, markdown conversions, OKF knowledge graph (`kg/`)              | [AGENTS.md](../fedmaq-literature/AGENTS.md)    | `thesis-context.mdc` → experiments |
| [fedmaq-analyses](../fedmaq-analyses/)           | Notebooks, thesis figures                                            | [AGENTS.md](../fedmaq-analyses/AGENTS.md)      | `thesis-context.mdc` → experiments |
| [fedmaq-manuscript](../fedmaq-manuscript/)       | LaTeX thesis (Active; Ch 1-4 integrated, Ch 5 drafted, Ch 6 pending) | [README.md](../fedmaq-manuscript/README.md)    | **Owner:** `.claude/rules/`        |
| [fedmaq-presentations](../fedmaq-presentations/) | Beamer slides                                                        | [AGENTS.md](../fedmaq-presentations/AGENTS.md) | `thesis-context.mdc` → experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content. Index via `../fedmaq-experiments/.claude/rules/`.

**Agent tooling layout (mixed state):** `fedmaq-experiments` and `fedmaq-manuscript` have migrated to Claude Code — each owns `.claude/rules/`, `.claude/skills/`, `.claude/project/` (experiments also has `.claude/commands/`). `fedmaq-literature`, `fedmaq-analyses`, and `fedmaq-presentations` have not migrated yet and still use `.cursor/rules/`, `.cursor/skills/`, `.cursor/project/`. Their `thesis-context.mdc` files still point at `fedmaq-experiments/.cursor/rules/`, which no longer exists — those pointers are stale until each repo's own migration pass updates them to `.claude/rules/`. No shared parent config directory across repos (may add later).

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
- **Pending:** `refactor/cleanup-and-feddistill` **merged to `main`** (PR #1, `7b05f17`). Port CFD baseline (P11, ~Oct 2026 — still a config-time-guarded stub). Docker integration. Minor debt: mypy non-blocking (27 errors, mostly Hydra OmegaConf->dict variance + torch `Dataset.__len__` stubs); DAdaQuant unit-test params (phi=3,q_min=4) differ from `dadaquant.yaml` (phi=5,q_min=1) — reconcile by composing real configs in tests.

### [fedmaq-manuscript](../fedmaq-manuscript/) — [Active]

- **Completed:** Chapter 1–4 LaTeX template integrated, Claude audit revisions applied. Chapter 5 drafted. `chapter_4.tex` fixes (SVD schedule, KD temperature split, candidate-formulation table, `|D_pub|` unification, dataset-overview caption, flagged grid-size note) **committed** (`b6c08bc`).
- **Pending:** Reconcile the flagged experimental-grid-size note in `chapter_4.tex` (Software/MLOps Stack subsection) against the intended final run count; finalize Chapter 5; draft Chapter 6; incorporate proposal panel feedback post-defense.

### [fedmaq-literature](../fedmaq-literature/) — [Restructured to OKF]

- **Status details:** Migrated to an **Open Knowledge Format** bundle (branch `okf-restructure`). Two layers: raw `markdown/{slug}/paper.md` (verbatim, citable) + curated OKF nodes under `kg/`. Removed the Chroma/LlamaIndex vector-RAG stack (grep + read replaces retrieval); converted the 10 previously-missing canon papers; all **39** papers are now `type: Paper` nodes in `kg/papers/{slug}.md` (start at `kg/index.md`). Retargeted `AGENTS.md`/`README.md`/`.cursor/rules/`/`.cursor/skills/` to the bundle and slimmed [paper_registry.md](../fedmaq-literature/.cursor/project/paper_registry.md) to conversion status (39 rows). Conversion pipeline unchanged (Docling primary, Marker fallback). 14 tests green.
- **Pending:** Phase 2+ knowledge-layer population (`kg/{methods,concepts,findings,gaps}/` are scaffolded but empty). Deferred polish: conversion-pipeline QA tuning; math rendering in converted bodies.

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

**Current focus:** Literature OKF restructure (Task 14) is complete on branch
`okf-restructure` — merge it to `main` when ready. The two prior open threads are
resolved: the experiments `refactor/cleanup-and-feddistill` branch merged (PR #1),
and the manuscript `chapter_4.tex` edits committed (`b6c08bc`). Next substantive
work is **P7 — WandB + Hydra ingest utilities** (`fedmaq-analyses`). Still open:
P9 (OKF Phase 2 findings in literature), P11 (CFD, ~Oct 2026 stub), and reconciling
the flagged experimental-grid-size note in `chapter_4.tex`.

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

**Recommend handoff.** The literature OKF restructure (P14) is complete and
self-contained on branch `okf-restructure` in `fedmaq-literature`: the vector-RAG
stack is removed, all 39 papers are OKF `type: Paper` nodes under `kg/`, docs/rules/
skills are retargeted, the registry is slimmed, and 14 tests pass. The branch is
ready to merge to `main` at the user's discretion. Both prior open threads have since
closed — the experiments refactor merged (PR #1) and the manuscript `chapter_4.tex`
edits are committed (`b6c08bc`) — so this is a clean stopping point; initiate a new
session to clear context. Next substantive work is **P7: WandB + Hydra ingest
utilities** in `fedmaq-analyses`. Also open when convenient: **P9** (OKF Phase 2 —
populate `kg/findings/` and the other knowledge-layer node types in literature) and
reconciling the flagged experimental-grid-size note in `chapter_4.tex`.
