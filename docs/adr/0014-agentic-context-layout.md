# ADR-0014 — Agentic context layout: ADRs are the sole decision record

**Status**: Superseded for workspace-wide layout by [ADR-0015](0015-workspace-agentic-context-contract.md); retained as this repository's historical migration record
**Supersedes**: `docs/DECISIONS.md` Decisions 14–17, 63, 69 (file deleted)
**Adopts**: [`../../../fedmaq-journal-paper/docs/adr/0012-agentic-context-layout.md`](../../../fedmaq-journal-paper/docs/adr/0012-agentic-context-layout.md)

## Context

This repo's agentic context had grown to roughly three times the size of the
workspace's reference layout, and the growth was structural rather than incidental:

- **`docs/DECISIONS.md`** — 1718 lines, 88 entries, two heading conventions, entries
  numbered out of append order. Append-only logs record supersession by pointing
  *forward* from a live entry, so a dead decision sits there looking current. An ADR
  says "superseded" in place.
- **`docs/STATUS.md`** — 254 lines mixing durable reference with state that changed
  every session. It had gone internally inconsistent: one block asserted nothing in
  the grid had been executed while a later block in the same file corrected it.
- **`.claude/rules/`** — six files, several of which existed to describe other files.
- **`.claude/project/`** — a stale changelog, a workspace map, and a baseline registry
  living apart from the rules that govern baselines.
- **Two `archive/` subtrees** — 3343 lines nothing cited for a verdict.

The workspace already had a designed answer. `fedmaq-journal-paper`'s ADR-0012 is the
reference structure and states that the siblings are pending migration to it. **This
is that migration, for this repo.**

## Decision

Adopt ADR-0012's layout — `CLAUDE.md` as entry point, `CONTEXT.md` as orientation,
`.claude/rules/` for always-loaded rules, `docs/agents/` for consult-on-demand
reference, `docs/adr/` as the sole decision record, and GitHub Issues for live state.
Read that ADR for the reasoning; it is not restated here.

### Deltas from ADR-0012, and why

**1. Superseded ADRs may be deleted or folded. This overrides ADR-0012's policy.**

ADR-0012 says "superseding one means editing the old file's `Status` line to point
forward, never deleting it." **That policy does not apply in this repo.** Leanness is
prioritized over the receipt trail: 88 entries collapsed into ten thematic ADRs
rather than 88 files, and two entries got no ADR at all. Git history is the receipt.
A future reader should not assume ADR-0012's default governs here.

**2. `CONTEXT.md` remains a glossary. This is deliberate, and ADR-0012 agrees.**

ADR-0012 says a `CONTEXT.md` is "NOT a glossary" — but its own reasoning names the
exception: the journal-paper's copy avoids being one *because* "All shared vocabulary
defers to `fedmaq-experiments/CONTEXT.md`." **This is the file that rule was written
to protect.** It is the canonical vocabulary for the code/manuscript boundary and
resolves real naming drift between them. Do not "fix" it toward pointer-only. It
gains the authority map and working conventions on top of the glossary.

**3. Live state moves to Issues, split by update cadence.** Dispatch state and
manuscript sync are separate pinned Issues — different cadences, different audiences.
Both are edited in place, not appended as comments.

**4. No new skill.** `docs-audit` is rescoped in place; `run-benchmark` and
`run-minitest` are untouched.

**5. Scope is this repo only.** `fedmaq-literature` and `fedmaq-analyses` received
pointer fixes where they named files this migration renamed. `fedmaq-manuscript` and
`fedmaq-presentations` were untouched. The siblings remain pending migration, as
ADR-0012 already states for the workspace generally.

### The rule the whole layout serves

**One canonical home per fact.** A number, a status or a decision lives in exactly one
place and everything else points at it. The concrete corollaries — no archives,
reference behind pointers, no committed handoff file, no second registry — are in
`CONTEXT.md` § Working conventions, which is where an agent will actually be reading.

## Crosswalk: old `Decision N` → new record

**This table is required, not decorative.** Thirteen files under `conf/` are frozen
downstream of the `pre-registration` tag and cite `Decision N` in comments; they
cannot be edited. Citations in `src/`, `tests/` and `scripts/` are also left alone
deliberately — they are comments and docstrings, and a reviewer who checks out the tag
gets a tree where `docs/DECISIONS.md` still exists and those numbers still resolve.
Renumbering at HEAD would leave the tagged artifact speaking a vocabulary nothing at
HEAD defines. **This table is the bridge in both directions.**

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

**Results are not decisions.** Decision 86 (FedMAQ vs. the uncompressed control at
equal bytes) and the raw per-cell numbers behind the exploration and tuning verdicts
went to Issues. An ADR directory carrying live findings reads as settled policy to a
future agent, and goes stale the moment the confirmatory grid lands.

## Consequences

- **New decisions get a new numbered ADR**, not an entry appended to a log.
- **Nothing in `docs/` carries a run count.** If you find one, it is a bug.
- The `docs-audit` skill audits the full context surface against this layout.
- **Recovery SHAs, named because `git log --follow` traverses none of these**
  (a wholesale directory delete, and a move-plus-rewrite in one commit):
  - `47fca68:docs/DECISIONS.md` — all 88 entries in full
  - `f7a095d^:docs/audits/archive/`, `f7a095d^:docs/experiments/archive/`
  - `75df164^:docs/RUNBOOK.md`, `75df164^:docs/STATUS.md`
