# ADR-0015: Workspace agentic-context contract

**Status**: Accepted
**Date**: 2026-08-09
**Scope**: Six FedMAQ repositories; current loader and ownership contract

## Context

The six FedMAQ repositories need one predictable boundary between always-loaded
instructions, task guidance, durable decisions, and live work state. Codex and
Claude discover repository instructions differently, so a tool-specific import
assumption cannot be the shared contract. Duplicated skills and status material
also make ownership and freshness ambiguous.

The loader observations and their source records are maintained in the
[context-modernization inventory](../agents/context-modernization-inventory.json).

## Decision

### Loader contract

- Codex discovers `AGENTS.override.md` and `AGENTS.md` from the project root
  toward the working directory. `@` lines are not a Codex import mechanism.
- Claude loads ancestor `CLAUDE.md` files and expands their `@` imports. Each
  repository's `CLAUDE.md` therefore remains a thin `@AGENTS.md` wrapper.
- `AGENTS.md` contains only the universal core. Activity-specific rules and
  skills are named by sharp plain-language pointers and read on demand; no
  nested `AGENTS.md` imports are used.

### Ownership and skill lifecycle

- `fedmaq-experiments` owns the shared glossary, workspace contract, and
  cross-repository authority. Spoke `CONTEXT.md` files provide local orientation
  and point to that authority without copying it.
- Each shared project skill has one canonical owner at
  `.agents/skills/<name>/SKILL.md`. `.claude/` is reserved for genuinely
  Claude-specific extensions; shared skills are not copied, symlinked, or kept
  under compatibility aliases.
- Every new or revised skill declares model or user invocation, routes only the
  branches it owns, discloses conditional detail progressively, and ends with a
  checkable completion criterion. Validate changed skills and update active
  references when a skill is renamed or removed.
- Installed or personal skill distributions are outside this repository
  contract. A project skill is deleted only after its active consumers move to
  its successor.

### Assurance and durable decisions

- Agent-context edits are post-assurance governance changes outside the bound
  candidate and its freeze evidence. Each slice classifies its proposed paths
  against the assurance envelope before editing.
- A proposal that changes methodology, protocol, behavior, configuration,
  evidence, scope, or scientific claims—or whose classification is ambiguous—
  stops and requires a separate assurance-impact decision and thesis-author
  disposition.
- ADRs record unique durable decisions. Historical evidence and incident or
  rebuild chronology belong in their authoritative records, not in a competing
  live policy. Cross-repository ADR references name the owning repository.
- GitHub Issues are the sole live-state record for open, blocked, and current
  work. Git history is the historical record; do not create tracked handoffs,
  changelogs, or status queues. Consult `docs/agents/` for on-demand reference
  procedures and `docs/adr/` for durable decisions.
- When the thesis author explicitly authorizes it, a superseded ADR may be
  folded into its current authority and deleted despite a former local
  retention rule. Preserve the ADR identifier as a permanent gap; do not
  renumber for contiguity.
- On 2026-08-30, the thesis author explicitly authorized Journal ADR-0004 and
  Journal ADR-0012 to be folded or deleted after consumer migration. This
  supersedes the former Journal ADR-0012 rule that superseded ADRs are never
  deleted.
