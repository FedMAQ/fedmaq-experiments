# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                         |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-09                                                                    |
| **Last session focus** | experiments: deep cleanup/refactor — hook-based client/server dispatch (no `alg_name` branching), dedup into `core/`, correctness fixes (FedMAQ grad-norm arch, q=1 NaN, DAdaQuant clamp), and a full FedDistill+ port. Branch `refactor/cleanup-and-feddistill`. |
| **Active repo**        | fedmaq-experiments                                                            |
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

### [fedmaq-experiments](../fedmaq-experiments/) — [Deep-refactored, hook-based; FedDistill+ ported]

- **Status details:** See completed baselines in [baseline_registry.md](.claude/project/baseline_registry.md). This session (branch `refactor/cleanup-and-feddistill`, not yet merged to `main`) removed all `alg_name` string-dispatch: client-side local training now goes through `ClientFitStrategy` hooks (`core/client_hooks/`), and server-side time/comms modelling through new `StrategyHook` methods (`download_size_bytes`, `compute_speed_scale`, `local_train_sample_count`, `server_sim_time`) — `strategy.py`/`NetworkSimulator` carry no per-algorithm branches. Magic numbers (fedkd `2.5`, server KD `2000.0`, fedmd pretrain `10`) moved to config. Deduped KD/partition/SVD/quantization helpers into `core/`. Correctness fixes: FedMAQ grad-norm now on the KD-student architecture (was silently zeroed on CIFAR), 1-bit FedPAQ sign-quantization (was NaN), DAdaQuant per-client `q` clamp, `set_model_parameters` shape guard. **FedDistill+ ported** (FedAvg weights + label-wise logit KD; `core/{client_hooks,strategy_hooks}/feddistill.py`). Safety net added: `simulation.run(cfg)`, composition + golden time/bytes tests, 54 tests green, ruff clean.
- **Pending:** Merge `refactor/cleanup-and-feddistill` to `main` (open PR). Port CFD baseline (P11, ~Oct 2026 — still a config-time-guarded stub). Docker integration. Minor debt: mypy non-blocking (27 errors, mostly Hydra OmegaConf->dict variance + torch `Dataset.__len__` stubs); DAdaQuant unit-test params (phi=3,q_min=4) differ from `dadaquant.yaml` (phi=5,q_min=1) — reconcile by composing real configs in tests.

### [fedmaq-manuscript](../fedmaq-manuscript/) — [Active]

- **Completed:** Chapter 1–4 LaTeX template integrated, Claude audit revisions applied. Chapter 5 drafted. `chapter_4.tex` updated this session with reported fixes (SVD schedule, KD temperature split, candidate-formulation table, `|D_pub|` unification, dataset-overview caption, flagged 516-run grid discrepancy) — **uncommitted**, left for user to review/commit.
- **Pending:** User to reconcile the flagged experimental-grid-size note in `chapter_4.tex` (Software/MLOps Stack subsection) against the intended final run count; commit `chapter_4.tex`; finalize Chapter 5; draft Chapter 6; incorporate proposal panel feedback post-defense.

### [fedmaq-literature](../fedmaq-literature/) — [Complete]

- **Status details:** See 29 parsed and ingested papers in [paper_registry.md](../fedmaq-literature/.cursor/project/paper_registry.md).
- **Pending:** None.

### [fedmaq-analyses](../fedmaq-analyses/) — [Scaffold complete]

- **Status details:** See active figures and notebook status in [figure_registry.md](../fedmaq-analyses/.cursor/project/figure_registry.md).
- **Pending:** WandB/Hydra ingest implementations, Real ablation + thesis figure notebooks.

### [fedmaq-presentations](../fedmaq-presentations/) — [Complete]

- **Status details:** Slide mapping and metadata updated in [slide_registry.md](../fedmaq-presentations/.cursor/project/slide_registry.md).
- **Pending:** None.

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
| 10  | Port FedDistill baseline                                   | experiments | [x]    |
| 11  | Port CFD baseline                                          | experiments | [ ]    |
| 13  | Deep cleanup/refactor: hook-based client/server dispatch, dedup, correctness fixes | experiments | [x] |
| 12  | Align experiments code + Ch. 1/4 manuscript (proxy pool, discrete bit-widths, FedDistill spec); apply manuscript-side fixes to chapter_4.tex | experiments + manuscript | [x] |

> [!TIP]
> For **Task 8**, the agent should perform the corrections locally by reading the critique files (`summaries/drafts/*_critique.md`) and modifying the draft summaries directly, rather than calling OpenRouter APIs. This keeps the workflow fast and cost-free for the user's OpenRouter account.

**Current focus:** Merge the `refactor/cleanup-and-feddistill` branch to `main` (open a PR; the branch has 10 commits, 54 tests green, ruff clean). Then P7 — WandB + Hydra ingest utilities (`fedmaq-analyses`). P11 (CFD) remains ~Oct 2026 per Gantt and is a config-time-guarded stub. Still open from prior session: review/commit the `chapter_4.tex` edits (Task 12) and resolve the flagged experimental-grid-size note.

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

- Full session-to-session history lives in [changelog.md](.claude/project/changelog.md) (includes the 2026-07-09 Cursor→Claude Code migration and CLAUDE.md relocation, plus the 2026-07-03 hardening/refactor sessions).

---

## 11. Handoff recommendation

**Recommend handoff.** The manuscript-alignment task (P12) is complete and self-contained: experiments code changes are committed (`8590d1f`), tests pass (22/22), and the reported manuscript-side fixes have been applied directly to `chapter_4.tex` (uncommitted, in `fedmaq-manuscript`, awaiting user review/commit). This is a clean stopping point — initiate a new agent session to clear context. The next session should focus on **P7: WandB + Hydra Ingest Utilities** in the `fedmaq-analyses` repository (unchanged from before this session), or on reviewing/committing the `chapter_4.tex` edits if the user wants that done first.
