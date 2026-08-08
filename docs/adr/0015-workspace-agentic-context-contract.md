# ADR-0015: Workspace agentic-context contract

**Status**: Accepted
**Date**: 2026-08-09

## Context

The six FedMAQ repositories were developed with Claude Code entrypoints and
tool-specific rule directories. Codex needs a native `AGENTS.md`, and several
spokes accumulated duplicated domain pointers, session-memory guidance,
changelogs, and status registries. The resulting layout made authority depend
on which agent happened to open the repository.

The workspace is intentionally a coordinated six-repository checkout. The
experiments repository owns shared vocabulary and cross-repository authority.

## Decision

Every repository uses this layout:

```text
AGENTS.md                 canonical shared instruction index
CLAUDE.md                 imports AGENTS.md
CONTEXT.md                hub glossary or spoke orientation
.agent/rules/             stable, tool-neutral rules
docs/agents/              consult-on-demand references
docs/adr/                 durable decisions
.claude/, .Codex/         tool-native extensions only
```

`AGENTS.md` is lean and load-bearing. It points to `CONTEXT.md`, imports only
always-active rules, and directs task-specific reading. `CLAUDE.md` imports it
instead of mirroring it.

`fedmaq-experiments/CONTEXT.md` remains the canonical shared glossary and
authority map. Spoke `CONTEXT.md` files name local ownership and point to the
hub; they do not duplicate shared terms. Context files carry no live work
state.

GitHub Issues are the sole home for open, blocked, and current work. Git
history is the historical record. Tracked handoffs, changelogs, and manual
status queues are prohibited. Durable artifact inventories may remain only
when they describe committed artifacts rather than a work queue.

Shared rules are relocated directly to `.agent/rules/`; no symlinks, copied
compatibility files, or transitional aliases are retained. Claude Code skills
and settings, and Codex-only configuration, remain native to their tools.

This ADR is the workspace contract. A spoke adds an ADR only for a local,
hard-to-reverse exception.

## Consequences

- This supersedes the journal-paper ADR-0012's claim to be the workspace
  reference; that ADR remains the historical source for its repository-local
  design.
- A repository may assume sibling checkouts exist for cross-repository work.
- Existing context migrations must be reviewed and committed per repository.
