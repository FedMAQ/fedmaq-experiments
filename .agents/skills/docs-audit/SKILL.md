---
name: docs-audit
description: >-
  Reproduce the historical August 2026 six-repository context migration audit
  against its recorded workspace state; do not use it to assess current policy.
---

# Docs Audit

Use this model-invoked skill only to reproduce the August 2026 migration audit
against its recorded workspace state. It routes to the read-only validator
established by #77; semantic candidates remain for human disposition. It is not
a current-state check.

**Historical scope:** This validator and inventory record the August 2026
migration. On 2026-09-24, the thesis author retired repository-local
`CONTEXT.md` glossaries and `docs/agents/domain.md` routing. The inventory is
preserved as migration evidence and no longer describes desired state. Do not
run the validator against current checkouts.

The authoritative inventory and boundary are
[the inventory](../../../docs/agents/context-modernization-inventory.json). The
[validator](scripts/validate_workspace.py) is documentation governance, outside the
experimental pipeline and source_manifest.json scope.

## Run

From the workspace parent, run this deterministic command:

    uv run --project fedmaq-experiments python fedmaq-experiments/.agents/skills/docs-audit/scripts/validate_workspace.py --workspace . --inventory fedmaq-experiments/docs/agents/context-modernization-inventory.json

The command reads the six repository checkouts and the declared JSON inventory,
then prints PASS or FAIL. --json emits machine-readable counts and lists.
--self-test exercises the checked-in broken-link and nested-import negative
fixtures in a temporary workspace and must print PASS; the temporary workspace
is removed.

## Checks

- Inventory coverage for all tracked entrypoints, rules, project skills,
  contexts, ADRs, and docs/agents references in exactly six repositories.
- Required owner, disposition, successor field, and unresolved: false for
  every record; missing or empty values fail. Out-of-scope exclusions and
  scoped exceptions remain explicit.
- Relative Markdown links and inline agent-rule/skill paths resolve to current
  files; exact duplicate document bytes fail; AGENTS imports fail after the
  baseline-only exception expires.
- Each semantic authority candidate is mapped to its human-approved inventory
  owner and disposition; grep never decides ownership, meaning, or a successor.
- Every declared conditional branch is expanded, must match a current file, and
  every match must have an inventory disposition.
- Baseline Git blobs, current loader bytes/lines, aggregate totals, and recorded
  reduction percentages must reproduce exactly.

The validator has no write path or auto-fix mode. It does not authorize edits to
method, behavior, configuration, protocol, evidence, scope, or claims.

Done when the deterministic command and self-test return PASS, the output is
available for issue evidence, and every semantic candidate has an explicit human
disposition in the inventory before downstream migration proceeds.
