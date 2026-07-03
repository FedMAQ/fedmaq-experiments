# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------ |
| **Last updated**       | 2026-07-03                                                                           |
| **Last session focus** | experiments: Codebase Hardening, Performance Optimization & Mathematical Correctness |
| **Active repo**        | fedmaq-experiments                                                                   |
| **Blockers**           | None                                                                                 |

---

## 1. Quick start (new agent)

1. Open the **multi-root workspace** with all five `fedmaq-*` repos.
2. Read this file end-to-end, then the **active task** in [Section 6](#6-implementation-queue).
3. Load domain rules from [`fedmaq-experiments/.cursor/rules/`](.cursor/rules/) (canonical thesis context).
4. Work in **one primary repo** per task; use sibling `AGENTS.md` for entrypoints.
5. Before ending a session, update this handoff file with changelog entries and recommendations for clean context.

**Candidate:** Christian Joseph Bunyi | **Institution:** De La Salle University | **Advisor:** Fritz Flores

---

## 2. Workspace map

| Repo                                             | Role                              | Agent entry                                    | Domain rules                       |
| ------------------------------------------------ | --------------------------------- | ---------------------------------------------- | ---------------------------------- |
| [fedmaq-experiments](../fedmaq-experiments/)     | FedMAQ code, Hydra, Flower, WandB | [AGENTS.md](../fedmaq-experiments/AGENTS.md)   | **Owner:** `.cursor/rules/`        |
| [fedmaq-literature](../fedmaq-literature/)       | PDFs, RAG, summaries              | [AGENTS.md](../fedmaq-literature/AGENTS.md)    | `thesis-context.mdc` → experiments |
| [fedmaq-analyses](../fedmaq-analyses/)           | Notebooks, thesis figures         | [AGENTS.md](../fedmaq-analyses/AGENTS.md)      | `thesis-context.mdc` → experiments |
| [fedmaq-manuscript](../fedmaq-manuscript/)       | LaTeX thesis (template pending)   | [README.md](../fedmaq-manuscript/README.md)    | Defer until template               |
| [fedmaq-presentations](../fedmaq-presentations/) | Beamer slides                     | [AGENTS.md](../fedmaq-presentations/AGENTS.md) | `thesis-context.mdc` → experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content. Index via `../fedmaq-experiments/.cursor/rules/`.

**Cursor layout:** Each repo owns `.cursor/rules/`, `.cursor/skills/`, `.cursor/project/`. No shared parent `.cursor/` (may add later).

---

## 3. Locked architectural decisions

| Topic              | Decision                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| Thesis context     | `fedmaq-experiments/.cursor/rules/` (decomposed from `context.md`; `context.md` is human snapshot only) |
| Experiments layout | uv monorepo, code under `src/fedmaq/core/` and `src/fedmaq/baselines/`                                  |
| Tooling            | Preferred stack in `tech-stack.mdc`; adopt extra libs (pandas, sklearn, etc.) when justified            |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                         |
| RAG boundaries     | Drafts → `*/drafts/`; human `approve` before promotion; no cross-repo auto-edits                        |
| Embeddings         | **`Qwen/Qwen3-Embedding-4B`** local GPU; serialize GPU jobs (convert then embed)                        |
| LLM                | OpenRouter: `deepseek/deepseek-v4-flash` (default), `deepseek/deepseek-v4-pro` (synthesis)              |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                          |

---

## 4. Per-repo status

### [fedmaq-experiments](../fedmaq-experiments/) — [Phase 1 Env Complete; Hardened & Hook-Refactored]

- **Status details:** See completed baselines and status in [baseline_registry.md](.cursor/project/baseline_registry.md). Fully refactored `TelemetryFedAvg` into modular strategy hooks (`core/strategy_hooks/`), and hardened the codebase with performance, correctness, and robustness optimizations (partition resolution, model reuse, independent client seeds).
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

- **Completed:** Chapter 1–4 LaTeX template integrated, Claude audit revisions applied (benchmark scope, $\alpha$ grid, and Pilot study adjustments).
- **Pending:** Draft final Chapters 5 and 6, incorporate proposal panel feedback post-defense.

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

- **Registries** (`.cursor/project/*.md`): update when adding baselines, papers, figures, or runs.
- **Rules** (`.cursor/rules/*.mdc`): concise; one concern per file; experiments owns domain rules.
- **Skills** (`.cursor/skills/*/SKILL.md`): procedural workflows; prefer skills over ad-hoc commands.
- **No emojis** in repo files (`repo-preferences.mdc`).
- **MCP:** context7 (library docs), GitHub (issues/PRs) — user-level.

---

## 9. What not to do

- Parse `papers/*.pdf` directly in Cursor chat.
- Auto-promote LLM drafts to `summaries/` or `syntheses/` without human approve.
- Let RAG agents edit experiments/analyses/manuscript code directly.
- Duplicate thesis domain content outside `fedmaq-experiments/.cursor/rules/`.
- Add top-level `reproductions/` packages; use `src/fedmaq/baselines/`.

---

## 10. Changelog

- See the complete historical archive of session-to-session changes in [changelog.md](.cursor/project/changelog.md).

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

Recommend initiating a new clean agent session to clear the current conversation history. The next session should focus on **P7: WandB + Hydra Ingest Utilities** in the `fedmaq-analyses` repository.
