---
name: docs-audit
description: >-
  Run the read-only structural validator for the six FedMAQ repositories and
  report broken references, missing dispositions, topology drift, exact
  duplicate documents, explicit exclusions, and semantic authority candidates.
  Use when auditing agent context or after a context migration; it never edits.
---

# Docs Audit

Invocation: model-invoked for an agent-context audit or post-migration check;
the maintainer runs the deterministic command and reviews semantic candidates.

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
  every record; explicit out-of-scope exclusions and scoped exceptions.
- Relative Markdown links resolve to current files; exact duplicate document
  bytes fail; AGENTS imports fail after the baseline-only exception expires.
- Semantic authority candidates are reported separately for human disposition;
  grep never decides ownership, meaning, or a successor.

The validator has no write path. It has no auto-fix mode and does not authorize
edits to method, behavior, configuration, protocol, evidence, scope, or claims.

Done when the deterministic command returns PASS, its output is retained as
issue evidence, and every semantic candidate has an explicit human disposition
in the inventory before downstream migration proceeds.
