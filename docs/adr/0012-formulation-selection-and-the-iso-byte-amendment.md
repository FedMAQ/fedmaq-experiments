# ADR-0012 — Formulation selection: the criterion, its amendment, and the freeze

**Status**: Accepted · 2026-08-01, amended 2026-08-06, frozen 2026-08-06
**Supersedes**: `docs/DECISIONS.md` Decisions 64–66, 82–85 (file deleted; see ADR-0014)
**Superseded in part by**: [ADR-0021](0021-power-mean-formulation-family.md) (decided 2026-08-27, [#34](https://github.com/FedMAQ/fedmaq-experiments/issues/34)) — the **five-candidate set** this decision selected over was replaced by one swept power-mean family plus two structurally separate rules. The criterion, iso-byte amendment, frozen v1 verdict, and historical vocabulary remain authoritative for v1; **Formulation 2 is not assumed to survive re-selection.**

## Historical scope

This ADR is the frozen v1 decision record. It retains the numeric F0--F4,
`gamma1`/`gamma2`, and bytes-to-target vocabulary needed to interpret v1 configs and
evidence. The current recut is owned by [ADR-0021](0021-power-mean-formulation-family.md);
that recut does not rewrite this historical record or the v1 artifacts. Run-level
evidence remains with the pinned evidence records and Issues.

## Decision

### The pre-registered selection rules

**A winner that splits across skews resolves to the severe skew.** `select_winner`
is per-`(dataset, alpha)`, but the freeze takes one scalar for FedMAQ and its
ablation arms. The rule is **agreement-or-severe-skew**: agreement freezes directly;
otherwise the α = 0.1 winner freezes and the split is reported.

The decision is per skew because each skew has its own cumulative-MB scale; an
aggregate would introduce a second comparison parameter.

**Total disqualification is pre-registered.** A formulation that fails the
uncompressed-FedAvg accuracy floor is disqualified. The guard is applied per seed,
so one failing seed disqualifies a formulation rather than being averaged away.

If every formulation is disqualified at one skew, that skew defers to the other. A
wipeout at both falls back to highest mean top-1 at R=100 at α = 0.1 and withdraws
the framing of formulation selection as the primary methodological contribution.

This differs from ADR-0010's empty-freeze branch: a formulation must be selected or
the primary contribution claim is withdrawn, rather than silently inheriting a
default.

**The near-tie tie-break compares against within-candidate spread, never pooled
spread.** Use the `max` of the two top candidates' own standard deviations. A
pooled spread is self-referential because it includes the separation being judged;
the sharper standard-error statistic was rejected as false precision at the small
seed count.

### The criterion amendment

The pre-registered crossing rule produced a non-discriminating v1 verdict: the
crossing floor could rank a transient crossing above a lower final accuracy.

**Bytes-to-target is replaced as the primary criterion. This is a disclosed
post-hoc amendment to the pre-registration, not a correction of an implementation
error** — `first_crossing` was a faithful reading of the pre-registered prose. The
defect was in the criterion.

1. **The criterion feeds the whole reported grid, not just this stage.** Fixing the
   instrument before the main experiment runs is different from re-scoring a
   completed one.
2. **0.9 × FedAvg-at-equal-rounds is the wrong comparator for a
   communication-reduction method.** It charges FedMAQ for accuracy while crediting
   nothing for the communication reduction under study. Under this thesis's own
   memory-constrained premise, full-precision FedAvg
   cannot run on the target clients at all: it is an infeasible reference bound,
   not a competitor.
3. **In this regime the criterion has no discriminative power.** The floor sits
   inside the noise band of the final accuracies, so no variant of the crossing
   rule is adopted.

**The amended criterion, fixed before it was computed:**

1. **Primary comparison is the accuracy-vs-cumulative-MB curve.** No free
   parameters; already mandated by the evaluation-metrics rule.
2. **Where a scalar is required, it is top-1 accuracy at the minimum common
   cumulative-MB budget across the arms compared** — B = min over arms of that
   arm's final cumulative MB, determined by the data, chosen by nobody. This is
   what keeps the amendment from smuggling in the free parameter the k-consecutive
   rule was rejected for.
3. **The superseded verdict is reported alongside**, with first-touch /
   k-consecutive / final-round columns as a robustness table.

**Acknowledged risk:** a minimum-common-budget rule may favour the formulation that
transmits fewest bytes. The rule is fixed before that outcome is visible.

### Formulation 2 is frozen

Under the amended criterion the skews diverge. Formulation 2 wins the severe skew;
the other skew is a near-tie and is not read as discriminating. The severe-skew
tie-break therefore freezes `formulation: 2`; the contribution is not withdrawn.
The superseded robustness columns remain part of the historical report.

The v1 config froze Formulation 2. It is multiplicative in both Tier-2 signals;
Formulation 3 modulates a gradient primary, while Formulation 0 uses no soft signal.

**Not claimed:** that Formulation 2 is better than 1 or 0 in general. The defensible
statement is that it was the v1 winner under the amended criterion and severe-skew
tie-break.

### Historical ablation expression

The v1 ablation expressed signal removal as:

- **Configuration 3** (`fedmaq_no_data`): `lambda_val: 0.0` → `gamma2: 0.0`.
- **Configuration 4** (`fedmaq_no_state`): three keys collapse to `gamma1: 0.0`.
  The fallback-arm exception was retired because
  Formulation 3 carries no weight on its gradient term and so cannot express
  state-awareness removal at any parameter setting, forcing that arm onto a
  different formulation and a different parity anchor. Under the multiplicative
  form the removal is expressible in place, the arm nests like every other, and its
  anchor reverts to Configuration 7.

**No renormalization of the surviving exponent was deliberate.** Under the linear
form, removing one weight caps the survivor at half the soft range; under the
multiplicative form, `x^0.5` already spans [0, 1]. The arms are therefore mirror
images under the v1 formulation.

### The reserved recheck is degenerate and is not spent

The reserved recheck was empty. It compared surviving layer with unrefined under
the winning formulation, and [ADR-0008](0008-exploration-protocol-and-the-empty-refinement-layer.md)
froze the surviving layer empty, so both arms resolve to byte-identical configs.

This is the same empty-layer degeneracy that skips the corresponding confirmation;
it is not a substitute factorial or a new exemption.

The sequential design's possible formulation-dependent refinement gap is vacuous
when the surviving layer is empty. `recheck_discharged` and `recheck_note` record
that explanation beside `recheck_required`.

## Consequences

- The frozen formulation is a single scalar in `conf/algorithm/fedmaq.yaml`, behind
  the tag (ADR-0010). Changing it invalidates every ablation arm.
- Ablation arm differences are pinned by `ABLATION_ARM_DIFFS` and a test asserting
  both single-signal removals are exact and symmetric under the frozen formulation.
- Live results measured under this freeze — including how FedMAQ compares against
  the uncompressed control at equal bytes — are **findings, not decisions**, and
  live in the pinned results Issue, not in this directory.
