# Context Docs Management

Conventions for `docs/`, `HANDOFF.md`, `.claude/project/`. Full sweep + auto-fix: `docs-audit` skill.

- **Single canonical registry**: `docs/experiments/README.md` tracks every experiment. Do not create a second registry (e.g. under `.claude/project/`) — it will drift and duplicate.
- **`docs/plans/<name>.md` is active-only**: exists only while it has open questions. When fully resolved, merge the resolution into `docs/DECISIONS.md` and delete the plan file. Git history is the record — don't keep resolved plans around as archives.
- **Archive pattern = per-directory `archive/` subfolder** (e.g. `docs/experiments/archive/`), not a top-level `docs/archive/`. Archived content stays next to its live counterpart.
- **`docs/DECISIONS.md` is append-only**: single source of truth for resolved decisions, dated entries. `STATUS.md`/`HANDOFF.md`/plans link to it instead of repeating content.
- **When editing any doc with a "Last updated" header, update the date to the current session date.**
- **`HANDOFF.md` is next-agent orientation only** — action items, not history. Detailed findings/audits go in `docs/`.
- **No section numbers in heading titles**: Do not prefix Markdown section headers with numbers (e.g. use `## Important Context`, not `## 1. Important Context`). This avoids renumbering churn and broken anchor links when reordering sections. Always retain explicit IDs for decisions (e.g. `Decision 36`), audit findings (e.g. `F10`), and manuscript cross-references (e.g. `§4.1`).
