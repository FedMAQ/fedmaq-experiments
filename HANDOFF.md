# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------ |
| **Last updated**       | 2026-07-09                                                                           |
| **Last session focus** | experiments + manuscript: Cursor to Claude Code migration and doc restructure        |
| **Active repo**        | fedmaq-experiments                                                                   |
| **Blockers**           | None                                                                                 |

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
| [fedmaq-literature](../fedmaq-literature/)       | PDFs, RAG, summaries                                                 | [AGENTS.md](../fedmaq-literature/AGENTS.md)    | `thesis-context.mdc` → experiments |
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
| RAG boundaries     | Drafts → `*/drafts/`; human `approve` before promotion; no cross-repo auto-edits                        |
| Embeddings         | **`Qwen/Qwen3-Embedding-4B`** local GPU; serialize GPU jobs (convert then embed)                        |
| LLM                | OpenRouter: `deepseek/deepseek-v4-flash` (default), `deepseek/deepseek-v4-pro` (synthesis)              |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                          |

---

## 4. Per-repo status

### [fedmaq-experiments](../fedmaq-experiments/) — [Phase 1 Env Complete; Hardened & Hook-Refactored]

- **Status details:** See completed baselines and status in [baseline_registry.md](.claude/project/baseline_registry.md). Fully refactored `TelemetryFedAvg` into modular strategy hooks (`core/strategy_hooks/`), and hardened the codebase with performance, correctness, and robustness optimizations (partition resolution, model reuse, independent client seeds).
- **Pending:** Port remaining SOTA baselines (FedDistill, CFD — Sep 2026), Docker integration.

### [fedmaq-literature](../fedmaq-literature/) — [Complete]

- **Status details:** See 29 parsed and ingested papers in [paper_registry.md](../fedmaq-literature/.cursor/project/paper_registry.md).
- **Pending:** None.

### [fedmaq-analyses](../fedmaq-analyses/) — [Scaffold complete]

- **Status details:** See active figures and notebook status in [figure_registry.md](../fedmaq-analyses/.cursor/project/figure_registry.md).
- **Pending:** WandB/Hydra ingest implementations, Real ablation + thesis figure notebooks.

### [fedmaq-presentations](../fedmaq-presentations/) — [Complete]

- **Status details:** Slide mapping and metadata updated in [slide_registry.md](../fedmaq-presentations/.cursor/project/slide_registry.md).
- **Pending:** None.

### [fedmaq-manuscript](../fedmaq-manuscript/) — [Active]

- **Completed:** Chapter 1–4 LaTeX template integrated, Claude audit revisions applied (benchmark scope, $\alpha$ grid, and Pilot study adjustments). Chapter 5 drafted.
- **Pending:** Finalize Chapter 5, draft Chapter 6, incorporate proposal panel feedback post-defense.

---

## 5. Literature RAG reference

For the architecture stack, ingestion workflows, runtime expectations, and RAG configuration details of the literature indexing pipeline, refer directly to [fedmaq-literature/README.md](../fedmaq-literature/README.md).

---

## 6. Implementation queue

Priority order for upcoming work. Mark items `[x]` when done; add new items at the bottom with date.

| P   | Task                                                       | Repo        | Status |
| --- | ---------------------------------------------------------- | ----------- | ------ |
| 1   | Implement PDF convert (Docling + Marker QA)                | literature  | [x]    |
| 2   | LlamaIndex + Chroma ingest with Qwen3-4B                   | literature  | [x]    |
| 3   | `fedmaq-lit` summarize + approve + OpenRouter              | literature  | [x]    |
| 4   | Phase 1 FL environment (data partition, bandwidth, Flower) | experiments | [x]    |
| 5   | FedAvg / FedProx / FedPAQ / DAdaQuant baselines            | experiments | [x]    |
| 6   | Full manuscript audit + codebase hardening (Ch. 1--4)      | experiments | [x]    |
| 7   | WandB + Hydra ingest utilities                             | analyses    | [ ]    |
| 8   | Review & approve remaining 10 draft summaries (remediate)  | literature  | [x]    |
| 9   | Compile/synthesize summaries into thematic syntheses       | literature  | [ ]    |
| 10  | Port FedDistill baseline                                   | experiments | [ ]    |
| 11  | Port CFD baseline                                          | experiments | [ ]    |

> [!TIP]
> For **Task 8**, the agent should perform the corrections locally by reading the critique files (`summaries/drafts/*_critique.md`) and modifying the draft summaries directly, rather than calling OpenRouter APIs. This keeps the workflow fast and cost-free for the user's OpenRouter account.

**Current focus:** P7 — WandB + Hydra ingest utilities (`fedmaq-analyses`). Tasks 10--11 (FedDistill, CFD) are Sep--Oct 2026 per Gantt.

---

## 7. Environment and secrets

| Variable                      | Used by               | Notes                                         |
| ----------------------------- | --------------------- | --------------------------------------------- |
| `OPENROUTER_API_KEY`          | literature            | LLM via OpenAI-compatible API                 |
| `FEDMAQ_EMBED_MODEL`          | literature            | Default `Qwen/Qwen3-Embedding-4B`             |
| `FEDMAQ_EMBED_FALLBACK_MODEL` | literature            | Default `Qwen/Qwen3-Embedding-0.6B`           |
| `FEDMAQ_EMBED_BATCH_SIZE`     | literature            | Default `4`                                   |
| `FEDMAQ_QA_MIN_MEAN_GRADE`    | literature            | Default `good` (Docling mean_grade threshold) |
| `FEDMAQ_QA_MIN_LOW_GRADE`     | literature            | Default `fair`                                |
| `FEDMAQ_MARKER_DEVICE`        | literature            | Override Marker device (`cuda` / `cpu`)       |
| `HF_HUB_DISABLE_SYMLINKS`     | literature            | Set automatically on Windows in Docling path  |
| `WANDB_API_KEY`               | experiments, analyses | Experiment tracking                           |

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

- Parse `papers/*.pdf` directly in Cursor chat.
- Auto-promote LLM drafts to `summaries/` or `syntheses/` without human approve.
- Let RAG agents edit experiments/analyses/manuscript code directly.
- Duplicate thesis domain content outside `fedmaq-experiments/.claude/rules/`.
- Add top-level `reproductions/` packages; use `src/fedmaq/baselines/`.

---

## 10. Changelog

- See the complete historical archive of session-to-session changes in [changelog.md](.claude/project/changelog.md).

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

---

## 11. Handoff recommendation

**Recommend handoff.** The Cursor-to-Claude-Code migration and CLAUDE.md restructure for `fedmaq-experiments` and `fedmaq-manuscript` are complete: `.cursor/`/`.agents/` removed, `CLAUDE.md` at repo root in both repos, `AGENTS.md` retired, tests passing (20/20). This is a clean, self-contained stopping point — initiate a new agent session to clear context. The next session should focus on **P7: WandB + Hydra Ingest Utilities** in the `fedmaq-analyses` repository (unchanged from before this session).
