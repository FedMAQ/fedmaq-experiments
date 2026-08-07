# ADR-0010 — Freeze machinery: what pre-registration has to survive

**Status**: Accepted · 2026-07-31 through 2026-08-06
**Supersedes**: `docs/DECISIONS.md` Decisions 58–60, 68, 72, 74, 88 (file deleted; see ADR-0014)

## Context

ADR-0004 draws a hard line between exploration and confirmation and stakes the
thesis's defensibility on it. A line like that is only worth what its machinery
enforces. The recurring failure mode this ADR exists to close is a **branch that
lives only where no reader can hold the thesis to it** — a YAML comment, a runbook
step, a docstring — while the manuscript states the unconditional version.

## Decision

### Ablation arms inherit the freeze; they do not restate it

Each ablation arm is `defaults: [fedmaq, _self_]` plus its own removal and nothing
else. Previously all five restated the refinement flags as literals, which made the
freeze a five-file hand edit at the single highest-stakes moment in the runbook,
and a partial edit desynced three arms. Parity stops being a property tests check
after the fact and becomes one the configs **cannot violate**: a second difference
is no longer expressible.

`fedmaq_no_refinements` still names all three flags, because "every refinement off"
is its *definition* rather than a copy of the freeze — whatever subset
`fedmaq.yaml` freezes on, that arm removes exactly that subset.

### Inheritance destroys self-description, so the snapshot is generated

Reading an inherited arm no longer tells you what it runs, and the manuscript's
"recoverable frozen configuration" promise leans on exactly that.
`scripts/dump_frozen_configs.py` composes each `fedmaq*` config exactly as
`scripts/run.py` would and commits the resolved result to
`docs/freeze/resolved_configs.yaml` with a per-arm `config_sha256` — taken over the
algorithm block alone, so it is stable across the dataset/skew/seed a matrix sweeps
and does *not* equal any run manifest's whole-config digest.

Being generated, it cannot drift the way five hand-maintained copies could.
`--check` exits non-zero on a stale snapshot and
`test_frozen_config_snapshot_is_current` runs that check in the suite, so a freeze
commit that edits `fedmaq.yaml` without regenerating **fails rather than ships**.

> Regenerating it is a runbook step, **and an instruction in a runbook is not a
> guard — the test is.** A stale snapshot is worse than none, since it describes a
> configuration the tag does not contain.

### Failure branches are pre-registered in the manuscript body, not in YAML

**The empty-freeze branch.** A matrix header and a test already pre-registered that
an empty surviving set drops Ablation Configuration 8 and the contribution bullet
resting on its contrast. But the manuscript said flatly that the study "consists of
eight test configurations", so the pre-registration was one no reader could hold
the thesis to — and writing it in *after* seeing an empty result is the exact thing
pre-registration prevents. The failure branch (empty set, FedMAQ ships unrefined,
no subset retries, no tuning rescue) and Configuration 8's existence condition now
live in the manuscript body. It executed; see ADR-0008.

**The reserved recheck.** A YAML comment pre-registered that a formulation winner
other than the incumbent triggers a 6-run recheck of the frozen refinement layer,
and that a failed recheck rewrites `fedmaq.yaml` and every ablation arm. A
procedure that spends uncounted runs and mutates the frozen configuration was
visible nowhere in the manuscript. Kept rather than deleted — this is one named
mechanistic coupling, not the generic sequential-selection caveat — but restated as
a **veto, not a search**: it can only remove from the frozen set, never add; the
factorial is never re-opened; no subset is retried.

**Chapter 6's contributions carry their withdrawal conditions.** Two contributions
were stated with no condition attached while the methods chapter had already
pre-registered their withdrawal. Nothing false was asserted — the bullets were
placeholders — but *a branch that exists only in the chapter that will not be
written under it is not pre-registered, it is remembered.* Both bullets now carry
their branches.

### Every confirmatory run postdates the tag

The calendar placed all reported baseline runs *before* the tag that locks the
baseline hyperparameter table those very runs are configured by — and the
matched-tuning stage (ADR-0011) is entitled to rewrite five of those constants
right up until the tag fires. Five of six baselines would have reported accuracies
measured under values the freeze had not yet fixed: the exact tuning asymmetry
ADR-0011 exists to remove, reintroduced through the schedule.

**Resolved: the confirmatory block runs whole, after the freeze.** The earlier
window is stated as baseline *reproduction* — implementation, correctness checks
against published behaviour, short validation runs — with no confirmatory
measurement in it.

*Rejected: a pre-registered conditional re-run* (baselines keep the early window;
any baseline whose constant changes has its grid rows re-run after the tag).
Expected cost is near zero and it preserves the calendar untouched, but the common
case still reports rows measured before the table locking them was tagged, and one
only knows no re-run was needed *after* seeing the tuning result — a
pre-registration that is partly retrospective. The compute is allocation-bound
rather than month-bound, so there was no reason to buy schedule with defensibility.

*Rejected: dropping the baseline table from the tagged artifacts* — cheapest, and
it unwinds ADR-0011 one day after it was added.

### The tag records a freeze; it does not constitute one

The `pre-registration` tag moved, and the move is recorded rather than done
quietly. The old tag's message named a formulation that was later superseded, and
it predated both the baseline table and the freeze it was supposed to carry — it
locked none of the three artifacts correctly.

- **The configs froze at the "Freeze Formulation 2" commit**, the earliest at which
  all three artifacts are right: the fixed mechanism set, the baseline table, and
  the selected formulation. The ablation sweep dispatched the same day, so the arms
  inherited exactly that state.
- **The tag sits at a later commit, deliberately.** Tagging the freeze commit was
  tried first and rejected on inspection: that tree still names an analyser that
  structurally cannot read its own runs (ADR-0011), so anyone checking out the tag
  cannot recompute the baseline table. A tag whose message cites a decision while
  its tree lacks it is self-undermining. The `conf/algorithm/` and `docs/freeze/`
  diff between the two commits is **empty** and every `conf/` change between them
  is comment-only, so the later placement costs nothing evidentially.
- **On ordering.** The runbook places the tag before the ablation, and the ablation
  ran first. That is admissible only because the tag *records* a freeze rather than
  constituting one — and once that is granted, placement is a documentation choice,
  so the tree that documents the freeze best wins. **What carries the "freeze
  preceded dispatch" claim is this record, the commit SHAs, and the empty
  `conf/algorithm/` diff — not the tag's position.** Stated so the retrospective
  tag is not read as a freeze reconstructed after results.

## Consequences

- **Nothing downstream of the `pre-registration` tag may edit a frozen config.**
  Thirteen files under `conf/` are frozen; they cite the old `Decision N` numbering
  and cannot be updated, which is why ADR-0014 carries a crosswalk table.
- A pre-registered branch is never deleted, even after the branch it describes has
  fired or been ruled out. Consolidating decisions must not silently drop an
  ablation-branch outcome.
- New failure branches go in the manuscript body first, and in the matrix header
  second. If it exists only in a comment, it is not pre-registered.
