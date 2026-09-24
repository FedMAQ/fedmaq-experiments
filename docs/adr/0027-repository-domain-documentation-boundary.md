# ADR-0027 — Repository domain documentation boundary

**Status**: Accepted · 2026-09-24
**Scope**: Six FedMAQ thesis repositories; domain terminology and routing
**Supersedes**: ADR-0014's local decision 2, ADR-0015's shared-glossary
ownership rule, and ADR-0016's reference to a thesis-root `CONTEXT.md`

## Context

The thesis author has retired repository-local `CONTEXT.md` glossaries and
`docs/agents/domain.md` routing. Those files required agents to enter a
separate domain-model layer before using repository documentation, while the
project already records method and protocol decisions in repository docs and
ADRs, and current work state in GitHub Issues.

The August 2026 migration records remain historical evidence. They explain the
former layout and are not current instructions.

## Decision

- The six thesis repositories maintain no `CONTEXT.md` glossary or
  `docs/agents/domain.md` routing document.
- Agents take terminology and authority from the target repository's docs,
  ADRs, configuration, and current GitHub Issues. Follow the repository's
  `AGENTS.md` for local procedures.
- Keep operational references in `docs/agents/` when a task needs them.
  Record durable decisions in ADRs and live state in GitHub Issues.
- Put a definition in the procedure or document that needs it when an
  operational distinction would otherwise be lost. Do not recreate a
  repository-wide glossary or domain-model bundle.
- Preserve dated migration ADRs, audits, freeze records, and manifests as
  written. Their references to the former layout describe the historical state.

## Consequences

- Each repository's `AGENTS.md` points to its actual docs, ADRs, and issue
  authority instead of a domain-model route.
- Active procedures retain only the local distinctions they use; the old
  migration inventory remains a historical record and is not a current-state
  validator.
- This decision changes documentation organization only; it does not alter
  method, protocol, configuration, evidence, or scientific claims.
