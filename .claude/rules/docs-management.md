# Context Docs Management

Conventions for `docs/`, `.claude/project/`. Full sweep + auto-fix: `docs-audit` skill.

- **Single canonical registry**: `docs/experiments/README.md` tracks every experiment. Do not create a second registry (e.g. under `.claude/project/`) — it will drift and duplicate.
- **`docs/plans/<name>.md` is active-only**: exists only while it has open questions. When fully resolved, merge the resolution into `docs/DECISIONS.md` and delete the plan file. Git history is the record — don't keep resolved plans around as archives.
- **Archive pattern = per-directory `archive/` subfolder** (e.g. `docs/experiments/archive/`), not a top-level `docs/archive/`. Archived content stays next to its live counterpart.
- **`docs/DECISIONS.md` is append-only**: single source of truth for resolved decisions, dated entries. `STATUS.md`/plans link to it instead of repeating content.
- **When editing any doc with a "Last updated" header, update the date to the current session date.**
- **No committed handoff file.** Session-to-session orientation is a *temporary* artifact produced by the `handoff` skill, never a tracked file in this repo. A committed `HANDOFF.md` existed until 2026-07-31 and failed predictably: it accreted durable operational content (the execution model, the dispatch order) that then went stale and was cited from three other docs, while the transient action items it was actually for were rewritten every session. Durable operational reference now lives in `docs/RUNBOOK.md`, current state in `docs/STATUS.md`, resolved decisions in `docs/DECISIONS.md`. If you are about to write next-session context into a tracked file, that is the mistake this rule names.
- **No section numbers in heading titles**: Do not prefix Markdown section headers with numbers (e.g. use `## Important Context`, not `## 1. Important Context`). This avoids renumbering churn and broken anchor links when reordering sections. Always retain explicit IDs for decisions (e.g. `Decision 36`), audit findings (e.g. `F10`), and manuscript cross-references (e.g. `§4.1`).
