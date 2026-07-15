---
name: docs-audit
description: >-
  Full sweep of context docs (docs/, HANDOFF.md, .claude/project/) for staleness,
  broken numbering, and duplication. Auto-fixes mechanical issues, flags
  judgment-call issues. Use when asked to audit/review the docs system, or
  periodically after a batch of doc edits to catch drift.
---

# Docs Audit

Conventions enforced: [.claude/rules/docs-management.md](../../rules/docs-management.md).

1. **Inventory**: list `docs/**/*.md`, `HANDOFF.md`, `.claude/project/*.md`.
2. **Staleness**: for each doc with a "Last updated" header, compare against its own body content and cross-linked docs (e.g. does it reference a decision dated later than its own header?). Auto-fix: bump the date.
3. **Section numbering**: for each doc with `## N.` headers, check the sequence is contiguous. Auto-fix: renumber.
4. **Duplicate/overlapping registries or content**: grep for docs covering the same ground (e.g. two experiment trackers, two decision logs). Flag — do not auto-merge; report the overlap and recommend which should be canonical per `docs-management.md`.
5. **`docs/plans/` lifecycle**: any plan whose "open questions" section is fully resolved (check against `docs/DECISIONS.md` for a matching dated entry) should be deleted after confirming its content made it into `DECISIONS.md`. Flag if unclear.
6. **Archive pattern conformance**: confirm archived material lives under a per-directory `archive/` subfolder, not a top-level `docs/archive/`. Flag deviations.
7. **Dead links**: check relative links/paths in edited docs resolve to real files. Auto-fix obvious renames; flag ambiguous cases.
8. **Report**: summarize what was auto-fixed vs. what needs a human decision, in that order.
