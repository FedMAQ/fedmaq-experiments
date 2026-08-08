---
name: docs-audit
description: >-
  Full sweep of the tracked context surface (AGENTS.md, CONTEXT.md,
  .Codex/rules/, docs/adr/, docs/agents/, docs/experiments/, docs/audits/)
  for staleness, duplicated facts, dangling references, and drift from the
  layout in ADR-0014. Auto-fixes mechanical issues, flags judgment calls.
  Use when asked to audit the docs system, or after a batch of doc edits.
---

# Docs Audit

Conventions enforced: `CONTEXT.md` § Working conventions, and
[ADR-0014](../../../docs/adr/0014-agentic-context-layout.md) for the layout.

1. **Inventory.** `AGENTS.md`, `CONTEXT.md`, `.Codex/rules/*.md`, `docs/**/*.md`.
   A tracked `HANDOFF.md` — or any committed next-session file — is itself a finding.
   Flag it; do not audit its contents.

2. **One canonical home per fact.** The core check. Grep for the same number, status
   or decision stated in two tracked files. Flag; do not auto-merge — report the
   overlap and say which should be canonical.
   - **Run counts are the known-recurring case.** No tracked file may carry one.
     Totals belong in the pinned dispatch Issue; per-stage arithmetic is pinned by
     `tests/test_simulation.py`, not by prose. Any run count in `docs/` or
     `.Codex/` is a finding, even if currently correct.

3. **Dangling references.** Relative links resolving to real files; `ADR-NNNN`
   citations resolving to a file in `docs/adr/`; skill names in prose resolving to a
   real skill. Auto-fix obvious renames, flag ambiguous ones.
   - `Decision N` citations in `conf/`, `src/`, `tests/` and `scripts/` are **not**
     findings. They are deliberate — see ADR-0014's crosswalk. Do not "fix" them.

4. **No archives.** An `archive/` subdirectory anywhere under `docs/` is a finding.
   Superseded material is deleted; git history is the record.

5. **Layout drift.** Compare against ADR-0014's structure and the reference layout it
   adopts. Flag: always-loaded rules growing past a screen or two; reference material
   in `.Codex/rules/` that belongs in `docs/agents/`; a second registry appearing
   anywhere; live state accumulating in a tracked file.

6. **Superseded-but-live content.** A doc describing a mechanism, pick or count that a
   later ADR overturned, without saying so. The soft-voting and MobileNetV2GN-smoke
   experiment pages carry supersession banners for this reason — check that pattern
   still holds wherever an experiment record predates a freeze.

7. **Staleness.** For any doc with a "Last updated" header, check its body against
   cross-linked docs. Auto-fix: bump the date. Prefer deleting a stale date header
   over maintaining one.

8. **Report** what was auto-fixed, then what needs a human decision, in that order.

**Done when** every finding is either fixed or listed with a recommendation, and the
inventory in step 1 has no file that step 5 would say does not belong.
