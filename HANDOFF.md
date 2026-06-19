# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                 |
| ---------------------- | ----------------------------------------------------- |
| **Last updated**       | 2026-06-19                                            |
| **Last session focus** | Literature PDF convert pipeline (Docling + Marker QA) |
| **Active repo**        | fedmaq-literature                                     |
| **Blockers**           | None                                                  |

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

### fedmaq-experiments — [SCaffold complete]

| Done                                                           | Pending                                         |
| -------------------------------------------------------------- | ----------------------------------------------- |
| `pyproject.toml`, `src/fedmaq/` phase packages, `conf/`, tests | Phase 1 environment implementation              |
| 11 `.cursor/rules/`, registries, 2 skills                      | Port baseline code into `src/fedmaq/baselines/` |
| `context.md` deprecation notice                                | Docker, `scripts/run.py`, WandB integration     |

Key paths: `src/fedmaq/phase1_env/` … `phase4_benchmark/`, `.cursor/project/baseline_registry.md`

### fedmaq-literature — [Scaffold complete]

| Done                                                                  | Pending                                             |
| --------------------------------------------------------------------- | --------------------------------------------------- |
| Folder layout, `.cursor/` rules/skills, `paper_registry.md` (partial) | Complete `paper_registry` for all PDFs in `papers/` |
| Docling + Marker convert pipeline, QA, `meta.yaml`, CLI               | LlamaIndex + Chroma ingest                          |
| `fedmaq-lit convert` / `ingest --convert-only`, unit tests            | OpenRouter summarize workflow, approve commands     |
| Smoke-tested on `hinton-2015-distillation`                            | Batch-convert remaining registry slugs              |

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

### fedmaq-manuscript — [Deferred]

Awaiting LaTeX template. Stub README only.

---

## 5. Literature RAG reference (implementation spec)

```
papers/*.pdf
  → Docling convert → QA → Marker fallback if low confidence
  → markdown/{slug}/paper.md + meta.yaml
  → LlamaIndex IngestionPipeline → storage/chroma/ (gitignored)
  → fedmaq-lit summarize → summaries/drafts/{slug}.md
  → human: fedmaq-lit approve → summaries/{slug}.md
  → syntheses/drafts/ → approve-synthesis → syntheses/{topic}.md
```

**Embedding:** `FEDMAQ_EMBED_MODEL=Qwen/Qwen3-Embedding-4B`, fallback `0.6B`, batch size 4–8. Query instruct in `src/fedmaq_literature/ingest/__init__.py`.

**GPU (RTX 5060 8GB):** Do not run Docling/Marker and 4B embedder concurrently.

**Skills:** `.cursor/skills/ingest-paper`, `summarize-paper`, `approve-summary`, `query-literature` (synthesize skills TBD).

---

## 6. Implementation queue

Priority order for upcoming work. Mark items `[x]` when done; add new items at the bottom with date.

| P   | Task                                                       | Repo        | Status                  |
| --- | ---------------------------------------------------------- | ----------- | ----------------------- |
| 1   | Implement PDF convert (Docling + Marker QA)                | literature  | [x]                     |
| 2   | LlamaIndex + Chroma ingest with Qwen3-4B                   | literature  | [ ]                     |
| 3   | `fedmaq-lit` summarize + approve + OpenRouter              | literature  | [ ]                     |
| 4   | Phase 1 FL environment (data partition, bandwidth, Flower) | experiments | [ ]                     |
| 5   | FedAvg baseline in `src/fedmaq/baselines/`                 | experiments | [ ]                     |
| 6   | WandB + Hydra ingest utilities                             | analyses    | [ ]                     |
| 7   | Manuscript `.cursor/` stub                                 | manuscript  | [ ] (blocked: template) |

**Current focus:** P2 — LlamaIndex + Chroma ingest with Qwen3-4B (`fedmaq-literature`).

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

Reverse chronological. Agents append one entry per session when using `agent-handoff` skill.

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

```
FedMAQ workspace handoff — read fedmaq-experiments/HANDOFF.md first.

Context: [1 sentence on thesis goal]
Last session: [what was done]
Active repo: [repo name]
Next task: [specific item from Implementation queue]
Read: fedmaq-experiments/.cursor/rules/ + [repo]/AGENTS.md
Constraints: [any blockers or do-nots for next task]
```

The skill fills this template automatically; do not hand off without updating Section 6 and the changelog.
