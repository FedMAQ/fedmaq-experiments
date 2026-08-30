# ADR-0014 — Agentic context layout: ADRs are the sole decision record

**Status**: Superseded for workspace-wide layout by [ADR-0015](0015-workspace-agentic-context-contract.md); retained as this repository's historical migration record
**Supersedes**: `docs/DECISIONS.md` Decisions 14–17, 63, 69 (file deleted)
**Historical source**: Journal ADR-0012 supplied the precursor layout; it was
folded and deleted under the thesis-author authorization recorded by ADR-0015.
Recover it from `fedmaq-journal-article` with
`git show fc3fc399283850d6d1c55abb9aad8341c7ce327d:docs/adr/0012-agentic-context-layout.md`.

## Historical scope

This record documents the experiments repository's context migration. It retains the
old Decision identifiers needed by frozen comments and historical evidence; Git
history carries the superseded files and chronology. Recovery archaeology for the
removed mutable-status surface remains discoverable at
`75df164^:docs/STATUS.md`; that status was superseded by Issue #10 and the live
Issues, not retained as a second authority here.

## Decision

The migration adopted `CLAUDE.md` as entry point, `CONTEXT.md` as orientation,
`docs/agents/` for consult-on-demand reference, `docs/adr/` as the sole decision
record, and GitHub Issues for live state. The current `.agents/rules/` loader
contract supersedes the historical tool-specific layout in [ADR-0015](0015-workspace-agentic-context-contract.md).

### Local decisions

**1. Superseded ADRs may be deleted or folded.**

ADR-0012's former retention rule does not apply in this repo. Superseded decisions
may be folded into current authorities; Git history is the receipt. Identifiers
remain permanent gaps and are never reused.

**2. `CONTEXT.md` remains the shared glossary.**

The experiments `CONTEXT.md` is the canonical vocabulary for the code/manuscript
boundary and carries the authority map and working conventions. Spoke contexts point
to it rather than copying shared domain content.

**3. Live state moves to Issues, split by update cadence.**

**4. Existing project skills are revised in place where their branches differ.**

**5. Scope is this repository.** Cross-repository pointers name their owning
repository; sibling migrations have their own tickets.

### Governing rule

**One canonical home per fact.** A number, status, or decision lives in one place and
other documents point to it. The concrete corollaries are in `CONTEXT.md`.

## Crosswalk: old `Decision N` → new record

**This table is the bridge in both directions.** Frozen configs and code comments
retain historical `Decision N` references; the identifiers are not renumbered.

Full text of any entry: `git show 47fca68:docs/DECISIONS.md`.

| Decision(s) | Now in |
| :-- | :-- |
| 1–13, 52, 75 | [ADR-0004](0004-confirmatory-grid-design.md) — grid design, iso-architecture, freeze boundary, comparison regime, uniform-memory arm |
| 14–17, 63, 69 | **ADR-0014** (this file) — doc conventions, superseded by this migration |
| 18, 19, 40 | [ADR-0006](0006-determinism-and-the-golden-diff-gate.md) — determinism, bit-exact gate |
| 20, 42–44, 46–48, 50, 51 | [ADR-0007](0007-architecture-deepening-seams.md) — where each concern lives |
| 21–23, 25, 26, 45, 49 | [ADR-0005](0005-baseline-stack-membership.md) — baseline stack, exclusions, FedKD fixes, validation scope |
| 24 | *no ADR* — the FedMD digest-epoch trim, moot before it ran (superseded by Decision 25 / ADR-0005) |
| 27–31, 33–35, 53, 61, 70, 79, 80 | [ADR-0008](0008-exploration-protocol-and-the-empty-refinement-layer.md) — exploration protocol, √2σ rule, empty layer |
| 32 | [ADR-0001](0001-client-kd-teacher-deepcopy-is-structural.md) — client-KD teacher deepcopy |
| 36–38 | [ADR-0002](0002-hardware-telemetry-grounding.md) — hardware telemetry grounding |
| 39, 54–57, 71, 76 | [ADR-0009](0009-run-identity-and-analysis-scoping.md) — run identity, output paths, analysis scoping |
| 41 | [ADR-0003](0003-training-skeleton-seam.md) — the `run_epochs` seam design |
| 58–60, 68, 72, 74, 88 | [ADR-0010](0010-freeze-machinery-and-pre-registration.md) — freeze machinery, pre-registration, the tag |
| 62 | *no ADR* — a `ruff` cleanup pass before the freeze |
| 64–66, 82–85 | [ADR-0012](0012-formulation-selection-and-the-iso-byte-amendment.md) — selection criterion, iso-byte amendment, Formulation 2 |
| 67, 73, 81, 87 | [ADR-0011](0011-baseline-matched-tuning.md) — Stage 1b, DAdaQuant's units, the verdicts |
| 77, 78 | [ADR-0013](0013-execution-infrastructure-failures.md) — `client_gpus`, partition-ID retry, Ray teardown |
| 86 | *no ADR* — a measured **result**, not a decision. Pinned dispatch-state Issue. |

**Results are not decisions.** Measured outcomes and raw per-cell numbers belong to
the pinned Issues and evidence owners, not this ADR directory.

## Consequences

- New decisions get a new numbered ADR, not an appended log entry.
- Run counts and mutable state belong to Issues, not `docs/`.
- The `docs-audit` skill validates this context surface.
- Git history remains the recovery record for deleted decision, audit, experiment,
  runbook, and status files.
