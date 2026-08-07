# Audits

All foundational audits of the FedMAQ codebase and algorithm formulation are
complete. Every finding (F1–F18) and code-craftsmanship item was resolved, dropped,
or folded into a decision record.

**Where the findings went:**

| Findings | Resolved in |
| :-- | :-- |
| F1–F9 (FL engineering & code quality) | [ADR-0001](../adr/0001-client-kd-teacher-deepcopy-is-structural.md), [ADR-0007](../adr/0007-architecture-deepening-seams.md) |
| F10–F18 (distillation baseline health) | [ADR-0005](../adr/0005-baseline-stack-membership.md) |
| Telemetry grounding | [ADR-0002](../adr/0002-hardware-telemetry-grounding.md) |
| Experiment-defensibility passes 3–5 | [ADR-0009](../adr/0009-run-identity-and-analysis-scoping.md) through [ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md) |

The detailed audit logs were deleted with the archive directory in the 2026-08-07
context migration ([ADR-0014](../adr/0014-agentic-context-layout.md)). Nothing live
cites them, because every verdict they reached is stated in the ADR that carries it.
They are recoverable at **`f7a095d^:docs/audits/archive/`** — named explicitly
because `git log --follow` does not traverse a wholesale directory delete.

Current project state is tracked in GitHub Issues, not here.
