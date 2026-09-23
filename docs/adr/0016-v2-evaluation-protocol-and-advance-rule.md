# ADR-0016: FedMAQ-v2 evaluation protocol, advance rule, and analysis isolation

**Status**: Amended 2026-09-23; original dispatch design suspended
**Date**: 2026-08-17

## 2026-09-23 post-campaign amendment

The completed 499-cell replacement campaign is the first reported study. Its
artifacts and registered selection are preserved; no campaign-wide rerun is
planned. The observed FedKD nonfinite test losses make that arm invalid for
the first-study primary comparison. Report its attempted runs and the
post-run exclusion transparently, without treating a suspected implementation
cause as established. Full provenance closure and paired analysis are still
required before numerical conclusions are promoted to thesis claims.

FedMAQ-v2 now has two separately controlled branches. One investigates
server-side KD repair against a no-KD anchor while holding the replacement
campaign's selected power-mean configuration (p=0.5, omega=0.5) and other
non-KD factors fixed. The other investigates the no-KD-centered method and
non-KD changes justified by the first-study analysis; those changes must not
be combined with KD repair in a comparison intended to isolate KD's effect.
The exact candidate families, matrices, advance gates, and fresh-seed design
will be registered before any v2 dispatch. The original pilot, screen,
confirmatory counts, thresholds, and six-baseline roster below are historical
design, **not authorization to dispatch**.

FedKD may re-enter a future comparison only after bounded code/trace/paper
forensics and an exact-candidate stability gate. The local smoke gate must
cover all three previously nonfinite CIFAR-10 alpha=0.1 seeds beyond round 9;
the author-run preflight must then finish 100 rounds on those same seeds using
the exact frozen v2 candidate. A failure keeps FedKD excluded without
blocking either FedMAQ-v2 branch. Neither gate retroactively rescues the
first-study FedKD arm.

The first-study communication scalar remains the one common cumulative-byte
budget across the compared runs in a condition, as registered in ADR-0012;
accuracy-versus-cumulative-bytes curves are primary. The per-seed, pairwise
budget below is a proposed v2-specific instrument and must never silently
replace the first-study scalar. Any sensitivity analysis using it must be
labelled separately. Preserve first-study and v2 artifacts and analysis
namespaces independently.

## Context

The FedMAQ-v2 server-KD repair study was finalized as a plan document
(`fedmaq-experiments#15`, 2026-08-14) that lived only in that Issue and a
Windows temp copy. `docs/adr/` is the sole durable decision record
(ADR-0014); a plan a reader cannot find there is not one this project can be
held to, which is the exact failure ADR-0010 exists to close for the v1
grid. `fedmaq-manuscript#9` additionally found the approved plan's freeze
paragraph left two things genuinely undefined even though most of the advance
rule was already specified: how the interpolator handles a curve with
duplicate or backwards-moving cumulative-byte points, and how a tie is broken
when more qualifying repairs remain than advancing slots. `fedmaq-manuscript#16`
tracks promoting the plan itself, closing those two gaps, and adding the
v1/v2 analysis-isolation requirements the plan's harness change did not yet
state.

This ADR is the promoted, durable version of that plan. It has been verified
against the recovered plan text (`fedmaq-experiments#15`, unedited since
creation) for consistency; nothing here contradicts it, and the two closed
gaps are additions, not revisions.

## Original decision (historical; superseded for v2 dispatch)

### Scope and preservation

FedMAQ-v2 is a separately versioned, exploratory study that changes only
server-side KD. v1 artifacts, configs, and protocol are preserved untouched.
Every non-KD factor is locked to v1: Formulation 2, 100 rounds, models,
partitions, heterogeneous-memory distribution, and the post-processing
pipeline. New v2 configurations inherit frozen `fedmaq` without modifying it.

Five repair families, never combined: `no_kd` (mandatory anchor, `kd_epochs=0`
with an early bypass before teacher reconstruction or proxy-loader
construction), `delayed_ramp` (KD begins at round 10/25/50, linear ramp to
full weight over the following 10 rounds), `confidence_filter` (retain
teachers at normalized entropy `<=0.60/0.75/0.90`), `disagreement_weight`
(teacher weight `exp(-beta * JS(...))`, beta 0.5/1/2), `temperature_calibration`
(fixed T=0.5/2/4). Every other KD factor — optimizer, learning rate,
momentum, proxy split/batch size/ordering, loss form, `soft_voting=false` —
stays frozen to v1.

### Evaluation protocol

**Iso-byte scoring.** For every paired comparison and seed `s`, `B*_s` is the
minimum of the two same-seed terminal cumulative-byte budgets, defined
independently per pair. Both curves are scored at `B*_s`; a curve unable to
be scored there blocks the cell rather than being pooled with an unrelated
family, baseline, or seed.

**Interpolation.** Linear, only within each curve's own observed inclusive
byte range — never extrapolated; a non-bracketing curve blocks the cell for
diagnosis or re-dispatch. Seed aggregation computes each seed's paired delta
at its own `B*_s` first, then averages across seeds. Two behaviors closed by
`fedmaq-manuscript#9`, absent from the original plan text:

- **Duplicate cumulative-byte points.** When two consecutive logged rounds in
  one curve report the identical cumulative-byte total, the interpolator
  keeps the later (higher-round) logged accuracy and drops the earlier
  duplicate before interpolating.
- **Non-monotone cumulative-byte points.** If cumulative bytes decrease
  between two consecutive logged rounds in one curve, that is a data-integrity
  failure — it blocks the cell for diagnosis, identically to a non-bracketing
  curve. It is never silently sorted or repaired.

This interpolated metric is v2-only and must be reported as such; it is never
directly compared to v1's discrete historical scorer.

**Pilot** (13 runs: one no-KD reference plus 12 KD settings, existing seed 42,
CIFAR-10 α=1.0). Per family, select the setting with the highest
FedMAQ-v2-minus-no-KD accuracy at `B*_42`. Exact ties resolve by predeclared
setting order: delay `[10, 25, 50]`, entropy threshold `[0.90, 0.75, 0.60]`,
beta `[0.5, 1, 2]`, temperature `[2, 0.5, 4]`. A family is never eliminated at
this stage even if every scored setting loses. Pilot selection is final
before the screen begins.

**Screen** (up to 75 condition-runs: no-KD plus the four pilot-selected
variants, existing seeds 0/42/123, CIFAR-10 α∈{0.1,1.0}, CIFAR-100
α∈{0.1,1.0}, FEMNIST). The matching pilot observations may be reused for the
CIFAR-10 α=1.0/seed-42 cell only when resolved configs, output provenance,
and analysis version match.

**Freeze** — the advance/no-advance decision. `no_kd` always advances. A
non-anchor repair advances only if, on screen data, it improves on no-KD by
`>=1.0pp` on at least one priority cell (CIFAR-10 α=1.0 or FEMNIST), and is no
worse than `-1.0pp` on the other priority cell, both CIFAR-100 cells, and
CIFAR-10 α=0.1. All margins are paired seed-level mean differences at each
seed's `B*_s`. At most two non-anchor repairs advance; if more than two
qualify, rank by worst-priority-cell margin versus no-KD (i.e.
`min(margin_cifar10_a1, margin_femnist)`) and take the top two. **Tie-break,
closed by `fedmaq-manuscript#9`, absent from the original plan text**: an
exact tie in that ranking value, with only one advancing slot remaining, is
broken by the predeclared family order `delayed_ramp > confidence_filter >
disagreement_weight > temperature_calibration` — the same predeclared-order
philosophy the pilot stage already uses, not a secondary metric. Incomplete
expected seed coverage blocks scoring. The freeze artifact is versioned and
records the anchor, zero/one/two qualified repairs, exact settings,
resolved-config hashes, matrices, output groups, analyzer commit/version,
comparator rule, fresh seeds, and every gate above.

**Confirm** (210/240/270 condition-runs, depending on qualifier count: fresh
seeds 7/19/37/73/101, the frozen v2 set against FedAvg, FedProx, FedPAQ,
DAdaQuant, FedDistill, FedKD, on CIFAR-10 α∈{0.1,0.5,1.0}, CIFAR-100
α∈{0.1,1.0}, FEMNIST). Matched iso-byte accuracy is primary; KD-only
simulated server time and KD proxy passes are mandatory secondary costs.
CIFAR-10 α=0.5 is a predeclared corroborative check, not a binding gate. The
primary claim, competitive recovery: each frozen repair must improve on no-KD
by `>=1.0pp` on at least one priority cell and be no worse than `-1.0pp` on
the other, and be within `-1.0pp` of FedAvg and of the higher of the
FedPAQ/DAdaQuant paired arm means, separately in each priority cell — the
plan never takes a per-seed oracle maximum. This is a practical predeclared
recovery/non-regression criterion, not a formal non-inferiority test.
Seed-level paired deltas and two-sided 95% paired Student-t intervals are
reported for every stated comparison as descriptive context; they do not
replace the point-estimate gates. Every full-stack cell is reported
individually.

### Analysis-harness maintenance and v1/v2 isolation

The analysis harness ships a maintenance change before any v2 run, repairing
and regression-testing five diagnosed v1 defects (frozen-formulation metadata
shadowing, equal-round comparison against a local skew winner instead of the
scalar frozen formulation, ablation-arm selection, retired Configuration 8,
round-completeness key collisions) and adding a v2-aware iso-byte entry point
that accepts an explicit experiment group and writes only new v2 outputs.
v1's discrete historical scorer is not silently changed, and no v1 output is
rewritten.

This maintenance work is isolation-gated, per `fedmaq-manuscript#16`:

- **Freeze and hash the v1 analyzer and its output bundle first**, before any
  v2 analyzer code is written, as a pre-work certificate.
- **v2 scoring lives in a separate module, command, and output namespace**
  from v1's — never inside the same entry point or writing to the same output
  tree.
- **v1 regression fixtures run before and after the v2 analyzer work.** Any
  changed v1 JSON other than the repaired round-completeness report is a
  breach of isolation and blocks the change.
- **Both certificates — the pre-work freeze and the post-work regression
  check — record the exact commit and dependency versions** they were taken
  against, so a later reader can reproduce either side of the comparison.

Regression coverage also required: conflicting per-skew winners against the
scalar frozen Formulation 2; formal-group-only ablation selection; no retired
Configuration 8; a synthetic mixed-dataset 105-run completeness fixture
asserting input count equals unique completeness keys; v2 explicit-group,
pairwise-interpolation iso-byte analysis including an interpolation-versus-
v1-discrete-scoring guard; failure (not silent intersection) on missing
expected seeds; and, added by `fedmaq-manuscript#9`, synthetic fixtures for
duplicate-cumulative-byte dedup (asserting last-value-wins) and
decreasing-cumulative-byte detection (asserting the cell blocks rather than
silently reordering).

Clean manifests, final checkpoints, 100/100 rounds, exact expected
candidate/baseline seed coverage per cell, and collision-free per-dataset
completeness are required before analysis runs.

### Literature and locked terms

The candidate literature (`fedmaq-literature/docs/audits/fedmaq-v2-server-kd-repair-candidates.md`)
is hypothesis support only, never a performance claim; supporting papers are
ingested into Zotero and `fedmaq-literature/kg` after results, without
turning rationale into a result. The v1 ablation falsifies the current fixed,
equal-weight server-KD path on CIFAR-10 specifically — it does not falsify
KD generally, and the first pass never combines repair mechanisms. Shared
terms are `CONTEXT.md` in the thesis root.

## Original consequences (historical except the dated amendment below)

- `fedmaq-manuscript#9` and `#16` close against this record; the plan no
  longer lives only in `fedmaq-experiments#15` and a Windows temp file.
- The freeze artifact and both isolation certificates are the auditable
  evidence that the advance rule and the v1/v2 boundary were both followed,
  not just designed. A freeze or v2 analyzer change that cannot produce them
  has not actually closed this ADR's requirements.
- The original design required a new ADR for protocol changes. The dated
  2026-09-23 amendment instead records the author's explicit post-campaign,
  pre-v2-dispatch revision in this ADR. No v2 result has been produced under
  the historical thresholds, so the next exact registration must be visible
  here before dispatch rather than silently changing a running branch.
