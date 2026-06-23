# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                            |
| ---------------------- | -------------------------------------------------------------------------------- |
| **Last updated**       | 2026-06-23                                                                       |
| **Last session focus** | Configure sequential manuscript Gantt Chart and initialize `.cursor` rules stubs |
| **Active repo**        | fedmaq-manuscript                                                                |
| **Blockers**           | None                                                                             |

---

## 1. Quick start (new agent)

1. Open the **multi-root workspace** with all five `fedmaq-*` repos.
2. Read this file end-to-end, then the **active task** in [Section 6](#6-implementation-queue).
3. Load domain rules from [`fedmaq-experiments/.cursor/rules/`](.cursor/rules/) (canonical thesis context).
4. Work in **one primary repo** per task; use sibling `AGENTS.md` for entrypoints.
5. Before ending a session, run the **`agent-handoff` skill** (`.cursor/skills/agent-handoff/`) to update this file and emit a handoff message.

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
| Experiments layout | uv monorepo, 4 phases under `src/fedmaq/`, baselines in `src/fedmaq/baselines/`                         |
| Tooling            | Preferred stack in `tech-stack.mdc`; adopt extra libs (pandas, sklearn, etc.) when justified            |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                         |
| RAG boundaries     | Drafts → `*/drafts/`; human `approve` before promotion; no cross-repo auto-edits                        |
| Embeddings         | **`Qwen/Qwen3-Embedding-4B`** local GPU; serialize GPU jobs (convert then embed)                        |
| LLM                | OpenRouter: `deepseek/deepseek-v4-flash` (default), `deepseek/deepseek-v4-pro` (synthesis)              |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                          |

---

## 4. Per-repo status

### fedmaq-experiments — [Scaffold complete]

| Done                                                           | Pending                                         |
| -------------------------------------------------------------- | ----------------------------------------------- |
| `pyproject.toml`, `src/fedmaq/` phase packages, `conf/`, tests | Phase 1 environment implementation              |
| 11 `.cursor/rules/`, registries, 2 skills                      | Port baseline code into `src/fedmaq/baselines/` |
| `context.md` deprecation notice                                | Docker, `scripts/run.py`, WandB integration     |

Key paths: `src/fedmaq/phase1_env/` … `phase4_benchmark/`, `.cursor/project/baseline_registry.md`

### fedmaq-literature — [Complete]

| Done                                                                   | Pending |
| ---------------------------------------------------------------------- | ------- |
| Folder layout, `.cursor/` rules/skills, `paper_registry.md` (complete) | None    |
| Docling + Marker convert pipeline, QA, `meta.yaml`, CLI                |         |
| `fedmaq-lit convert` / `ingest --convert-only`, unit tests             |         |
| Smoke-tested on `hinton-2015-distillation`, `li-2020-fedprox`          |         |
| Batch conversion CLI (`--all` flag) and registration of all 29 papers  |         |
| LlamaIndex + Chroma ingest with Qwen3-4B                               |         |
| OpenRouter summarize & approve workflow CLI                            |         |
| Chroma RAG local query & LLM synthesis CLI                             |         |

Stack: Docling primary, Marker GPU fallback → `markdown/{slug}/` → Qwen3-4B → Chroma → query/summarize.

### fedmaq-analyses — [Scaffold complete]

| Done                                                      | Pending                                 |
| --------------------------------------------------------- | --------------------------------------- |
| `data/` layout + README, plot style stub, sample notebook | WandB/Hydra ingest implementations      |
| `.cursor/` rules, skills, `figure_registry.md`            | Real ablation + thesis figure notebooks |

### fedmaq-presentations — [Migration complete]

| Done                                                | Pending                                     |
| --------------------------------------------------- | ------------------------------------------- |
| `.agents/` → `.cursor/`, metadata aligned to FedMAQ | Slide content updates for vision-FL framing |
| `slide_registry.md` paths fixed                     |                                             |

### fedmaq-manuscript — [Active]

| Done                                                           | Pending                                          |
| -------------------------------------------------------------- | ------------------------------------------------ |
| LaTeX template integrated with Chapters 1--4                   | Draft final Chapters 5 and 6                     |
| Granular, non-overlapping Gantt Chart of Activities configured | Incorporate proposal panel feedback post-defense |
| `.cursor/` rules configured (`thesis-context`, `latex_rules`)  |                                                  |

---

## 5. Literature RAG reference (implementation spec)

```txt
papers/*.pdf
  → Docling convert → QA → Marker fallback if low confidence
  → markdown/{slug}/paper.md + meta.yaml
  → LlamaIndex IngestionPipeline → storage/chroma/ (gitignored)
  → fedmaq-lit summarize → summaries/drafts/{slug}.md
  → human: fedmaq-lit approve → summaries/{slug}.md
  → syntheses/drafts/ → approve-synthesis → syntheses/{topic}.md
```

**Embedding:** `FEDMAQ_EMBED_MODEL=Qwen/Qwen3-Embedding-4B`, fallback `0.6B`, batch size 4–8. Query instruct in `src/fedmaq_literature/ingest/__init__.py`.

**LLM Models (OpenRouter):** Use `deepseek/deepseek-v4-flash` for drafting summaries. Always use `deepseek/deepseek-v4-pro` for automated reviews, verification runs, and global thematic syntheses to ensure mathematical correctness.

**GPU (RTX 5060 8GB):** Do not run Docling/Marker and 4B embedder concurrently.

> [!IMPORTANT]
> **Expected Execution Runtimes:**
>
> - **Full PDF to Markdown Conversion (Docling + Marker QA):** ~6.5 to 7 hours total. Avoid re-running conversions from scratch unless necessary.
> - **Full RAG Ingestion & Embedding (Qwen3-Embedding-4B):** ~14 minutes per paper on CUDA GPU (RTX 5060) (total ~6.8 hours for 29 papers). Ingestion is run sequentially in a loop to provide real-time visibility, log updates, and incremental SQLite commits.
> - **Paper Summarization (DeepSeek-v4-Flash via OpenRouter):** ~20 to 25 seconds per paper.

**Skills:** `.cursor/skills/ingest-paper`, `summarize-paper`, `approve-summary`, `query-literature` (synthesize skills TBD).

**Component Roles & Purpose:**

- **Chroma Vector DB (RAG):** Manages split text chunks for granular, passage-level retrieval (e.g., retrieving exact formulas or parameters).
- **Paper Summaries (`summaries/`):** Lightweight markdown files designed to fit easily inside the LLM's context window. They provide high-level summaries of methodology, limitations, and relevance for global reasoning.
- **Thematic Syntheses (`syntheses/`):** Aggregates summaries by topic to trace cross-paper evidence, support core claims, and identify literature gaps.

---

## 6. Implementation queue

Priority order for upcoming work. Mark items `[x]` when done; add new items at the bottom with date.

| P   | Task                                                       | Repo        | Status                  |
| --- | ---------------------------------------------------------- | ----------- | ----------------------- |
| 1   | Implement PDF convert (Docling + Marker QA)                | literature  | [x]                     |
| 2   | LlamaIndex + Chroma ingest with Qwen3-4B                   | literature  | [x]                     |
| 3   | `fedmaq-lit` summarize + approve + OpenRouter              | literature  | [x]                     |
| 4   | Phase 1 FL environment (data partition, bandwidth, Flower) | experiments | [ ]                     |
| 5   | FedAvg baseline in `src/fedmaq/baselines/`                 | experiments | [ ]                     |
| 6   | WandB + Hydra ingest utilities                             | analyses    | [ ]                     |
| 7   | Manuscript `.cursor/` stub                                 | manuscript  | [ ] (blocked: template) |
| 8   | Review & approve remaining 10 draft summaries (remediate)  | literature  | [x]                     |
| 9   | Compile/synthesize summaries into thematic syntheses       | literature  | [ ]                     |

> [!TIP]
> For **Task 8**, the agent should perform the corrections locally by reading the critique files (`summaries/drafts/*_critique.md`) and modifying the draft summaries directly, rather than calling OpenRouter APIs. This keeps the workflow fast and cost-free for the user's OpenRouter account.

**Current focus:** P4 — Phase 1 FL environment (data partition, bandwidth, Flower) (`fedmaq-experiments`).

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

### 2026-06-23 — Manuscript Gantt Chart Refinement and Cursor rules initialization

- Re-designed the calendar of activities in `fedmaq-manuscript`'s `chapter_4.tex` to be strictly sequential and non-overlapping.
- Extended the schedule to April 2027 to de-risk baseline implementation and benchmarking under a 15-unit coursework load.
- Completely excluded December 2026 from active research, indicating coursework finals and holidays, and added a justifying text paragraph.
- Configured `.cursor/rules/` stubs in the `fedmaq-manuscript` repository (`thesis-context.mdc`, `latex_rules.mdc`, and `repo-preferences.mdc`).
- Swapped slides preparation and proposal defense/revision rows to match an April 2027 defense timeline.
- Verified successful LaTeX compilation using `pdflatex main.tex`.

### 2026-06-23 — Sequential Ingestion Refactoring, Math formatting fixes, and Automated Summary Review

- Refactored `fedmaq-lit ingest --all` pipeline to run sequentially in a loop over papers. This preserves memory state but commits after each paper, enabling real-time logging and checkpointing in Chroma DB.
- Fixed mathematical subscript notation in the LLM system prompt for `fedmaq-lit summarize` to prevent the model from replacing subscripts (`_`) with asterisks (`*`) inside LaTeX math blocks.
- Created `auto_review.py` script that uses `deepseek-v4-pro` to cross-reference draft summaries against full paper texts, automatically approving 18 drafts and flagging 10 drafts with detailed critiques for correction.
- Completed comparative study between `deepseek-v4-flash` and `deepseek-v4-pro` as reviewers, demonstrating that Pro has significantly higher symbolic/mathematical accuracy and avoids false approvals.

### 2026-06-22 — Math loop collapse deduplication and RAG database re-embedding

- Identified layout/OCR model loop collapse (repetitive `\quad \text {to}`, `\quad \text {to be}`, `\ ` spaces, etc.) in 13 converted papers that corrupted RAG retrieval.
- Implemented `deduplicate_math` inside `_clean_math_block` to clean loop collapses, strip empty math blocks, and normalize spacing specifically within math blocks without affecting markdown tables.
- Ran cleanup across the entire converted markdown corpus and validated that 13 papers were cleaned successfully.
- Reverted/cleaned existing Chroma vector storage and launched a full CUDA re-embedding task (`uv run fedmaq-lit ingest --all`) on Qwen3-Embedding-4B in FP16 precision.

### 2026-06-22 — CLI summarize, approve, query commands implemented

- Created `.env.copy` templates for environment configuration across repos.
- Added `update_registry_summary` function in `registry.py`.
- Configured OpenRouter client with DeepSeek models (`deepseek/deepseek-v4-flash` for summaries, `deepseek/deepseek-v4-pro` for synthesis queries).
- Implemented `fedmaq-lit summarize` command to generate draft markdown summaries under `summaries/drafts/`.
- Implemented `fedmaq-lit approve` command to promote draft summaries to `summaries/`.
- Implemented `fedmaq-lit query` command to retrieve local Chroma context and synthesize answers.
- Added 21 total tests passing in pytest (including `test_workflows.py`).

### 2026-06-22 — Literature Batch Ingestion Execution

- Executed literature RAG ingestion for all 29 papers using `uv run fedmaq-lit ingest --all`.
- Verified GPU utilization at 100% and memory at ~7.6GB / 8.1GB (GeForce RTX 5060) during the 6h 42m run.
- Confirmed all 29 papers are successfully chunked, embedded, and stored in Chroma DB.
- Updated `paper_registry.md` to mark all papers as `ready` in the `Indexing` column.
- Ensured all tests pass (`pytest` 17 passed).

### 2026-06-22 — Literature RAG Ingestion Pipeline with ChromaDB

- Implemented LlamaIndex IngestionPipeline with ChromaVectorStore persisting to `storage/chroma`.
- Configured Qwen3-4B embedding model support with automated CUDA detection and `torch.float16` data type initialization.
- Added explicit CUDA 13.2 wheels for `torch` and `torchvision` to `pyproject.toml` tool configuration to ensure `uv` builds a GPU-capable environment.
- Implemented smart CLI skipping: PDF-to-Markdown conversion is now skipped automatically if the paper is already marked ready and the converted markdown file exists.
- Added `--device` flag to explicitly target CPU or GPU and added a fail-safe that warns and aborts if CUDA is missing when trying to run heavy GPU models.
- Added unit tests in `tests/test_ingest.py` verifying document formatting, metadata parsing, and Chroma database deduplication.

### 2026-06-21 — Literature paper registration and batch conversion

- Registered all 16 unmatched PDFs under `papers/` in `paper_registry.md` using derived slugs, labels, and tags.
- Fixed existing mismatched PDF labels (e.g. `liu-2023-adagq`, `jimenez-2024-non-iid-survey`, `qin-2025-kd-survey`) to enable correct matching and resolution.
- Updated `cli.py` to support batch conversion of all pending papers via a new `--all` command flag.
- Created `tests/test_cli.py` to test the new batch-convert CLI argument parser features.
- Completed the batch-convert job and successfully reattempted and converted the failed `richter-2024-electric-load` paper, marking all 29 papers as `ready` in the registry.
- Analyzed KaTeX parsing/rendering errors in Docling outputs (layout boundary overlaps, transformer generation loop collapses, multiline bracket mismatches) and decided to ignore them because they do not impact downstream RAG or LLM comprehension.

### 2026-06-21 — Literature math cleanup and lint exclusion

- Addressed KaTeX parse errors due to alignment characters (`&` or `\\`) not being wrapped.
- Improved `_clean_math_block` to strip `equation`/`equation*` wrappers (with optional labels) to avoid nested environment errors.
- Enhanced `_post_process_markdown` to detect and convert aligned/multiline inline math (`$...$`) into block math wrapped in `aligned`.
- Added unit tests in `test_pipeline.py` to verify the stripping, wrapping, and inline conversion logic.
- Created `.markdownlintignore` to exclude the generated `markdown/` directory from lints and deleted redundant `.markdownlint.json`.
- Regenerated `hinton-2015-distillation` markdown successfully with all equations rendered clean.

### 2026-06-19 — Literature PDF convert pipeline

- Implemented Docling primary + Marker GPU fallback in `src/fedmaq_literature/convert/`.
- Added QA using Docling confidence grades and content heuristics; writes `markdown/{slug}/paper.md` + `meta.yaml`.
- Wired `fedmaq-lit convert`, `ingest --convert-only`, `list-slugs`; registry auto-updates conversion status.
- Added unit tests (registry, QA, pipeline write); Windows HF symlink workaround in Docling adapter.
- Smoke-tested convert on `hinton-2015-distillation` (Docling, QA passed).

### 2026-06-18 — Workspace scaffold

- Created `.cursor/` structure across experiments, literature, analyses, presentations.
- Decomposed `context.md` into experiments `.mdc` rules.
- Scaffolded uv monorepos, literature CLI stub, analyses data layout.
- Migrated presentations `.agents/` → `.cursor/`; aligned metadata to FedMAQ.
- Added README + AGENTS.md per repo.
- Created this `HANDOFF.md` and `agent-handoff` skill.

---

## 11. Handoff message template

When ending a session, the `agent-handoff` skill produces a message like this for the next chat:

```txt
FedMAQ workspace handoff — read fedmaq-experiments/HANDOFF.md first.

Context: [1 sentence on thesis goal]
Last session: [what was done]
Active repo: [repo name]
Next task: [specific item from Implementation queue]
Read: fedmaq-experiments/.cursor/rules/ + [repo]/AGENTS.md
Constraints: [any blockers or do-nots for next task]
```

The skill fills this template automatically; do not hand off without updating Section 6 and the changelog.
