# ADR-0012 — Formulation selection: the criterion, its amendment, and the freeze

**Status**: Accepted · 2026-08-01, amended 2026-08-06, frozen 2026-08-06
**Supersedes**: `docs/DECISIONS.md` Decisions 64–66, 82–85 (file deleted; see ADR-0014)

## Context

FedMAQ's Tier-2 rule is one of five candidate *formulations* for computing the soft
quality target from the soft quality signal. Selecting among them is the thesis's
self-described primary methodological contribution, so the selection rule has to be
fixed before the result is visible — and has to still be defensible once it is.

## Decision

### The pre-registered selection rules

**A winner that splits across skews resolves to the severe skew.** `select_winner`
is per-`(dataset, alpha)` and the study runs CIFAR-10 at both skews, so two
verdicts are structural, not exceptional. The freeze takes one scalar, the grid
runs one FedMAQ, and every ablation arm inherits one formulation. The rule is
**agreement-or-severe-skew**: if both skews pick the same formulation it freezes;
if they diverge the α = 0.1 winner freezes and the split is reported as the finding
the methods chapter already promises. Severe skew breaks the tie because it is the
regime this thesis's claims are staked on, so selecting where the problem is
hardest is the conservative direction rather than the flattering one.

*A symmetric aggregate was rejected on technical grounds, not rhetorical ones*:
`compute_target_floor` is called *inside* the per-skew loop, so each skew's
cumulative MB is measured against its own floor at its own convergence speed, and
summing incommensurable quantities hands the decision to whichever skew has the
larger absolute MB scale. Normalizing to fix that introduces exactly the second
free parameter the design forbids.

**Total disqualification is pre-registered.** The accuracy floor is a fraction of
*uncompressed* FedAvg, and the study deliberately runs FedMAQ in its least
communication-efficient state — quantized under a memory ceiling with the
post-processing pipeline withheld so the formulas are judged on mathematical merit.
Clearing that at α = 0.1 is genuinely uncertain, so the branch is real, not
hypothetical.

The rule is also **stricter than the prose read**: the guard sits inside the
per-seed loop, so **one** failing seed of three disqualifies a formulation, where
"any formulation that fails to reach the target accuracy" reads naturally as a
claim about its mean. The strict rule is the right one — it refuses to average away
a seed that never converged. A wipeout at one skew defers to the other outright; a
wipeout at *both* falls back to highest mean top-1 at R=100 at α = 0.1 **and
withdraws the framing of formulation selection as the primary methodological
contribution.**

This deliberately does *not* mirror ADR-0010's empty-freeze branch, which withdraws
a claim rather than manufacturing a winner. The disanalogy: "ship unrefined" is a
coherent configuration and "ship no formulation" is not — `fedmaq.yaml` gets a
number either way, so the only real choice is whether it is picked empirically or
left at whatever the implementation already contained. Freezing the incumbent by
default would mean the thesis's primary contribution was settled by a default value.

**The near-tie tie-break compares against within-candidate spread, never pooled
spread.** The implementation concatenated both candidates' crossing values and took
the sample SD of the combined six, folding the *between*-candidate separation into
the threshold: with within-candidate sd `s` and separation `d`, the combined
variance is `(4s² + 1.5d²)/5`, so the rule fired at `d < 1.069s` and **the
threshold grew with the very margin it was judging.** A pre-registered rule cannot
be self-referential. It is now `max` of the two candidates' own SDs, applied to the
**top two** candidates only — which is what the code always did and the prose never
said. The sharper statistic (standard error of the difference) was rejected: at
n = 3 the σ estimate itself carries a roughly twelvefold interval, so a finer
instrument built on top of it is false precision.

### The criterion itself was wrong, and was amended before the verdict was computed

Under the pre-registered rule the verdict was Formulation 3 by one-sided
disqualification: every formulation disqualified at α = 0.1 (the *same* seed
failing for four of five — one hard Dirichlet partition, not four independent
method failures), and one survivor at α = 1.0.

That verdict exposed a pathology. **The winner had the lowest mean R=100 accuracy
of all five**, having won on a floor it crossed transiently and finished below. All
five ended within 2.7pp of one another and all below the floor.

**Bytes-to-target is replaced as the primary criterion. This is a disclosed
post-hoc amendment to the pre-registration, not a correction of an implementation
error** — `first_crossing` is a faithful reading of the pre-registered prose. The
defect is in the prose. Reasons, in the order that carries them:

1. **The criterion feeds the whole reported grid, not just this stage.** It is what
   every baseline's bytes-to-target column would be built on. Fixing the instrument
   before the main experiment runs is a different act from re-scoring a completed
   one.
2. **0.9 × FedAvg-at-equal-rounds is the wrong comparator for a
   communication-reduction method.** It charges FedMAQ for accuracy while crediting
   nothing for the 4–8× fewer bytes per round, which is the entire mechanism under
   study. Under this thesis's own memory-constrained premise, full-precision FedAvg
   cannot run on the target clients at all: it is an infeasible reference bound,
   not a competitor.
3. **In this regime the criterion has no discriminative power.** The floor sits
   inside the noise band of the final accuracies, so whether a seed "crosses" is
   close to a coin flip. First-touch, k-consecutive-rounds and a final-round gate
   are therefore not three rival operationalizations but three coin flips on the
   same near-tie — which is why *no* variant of the crossing rule is adopted.

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

   *Placement corrected 2026-08-08 (pass 15).* This clause read "in an appendix"
   until the manuscript was checked against it. The table is not in an appendix and
   never was: `chapter_4.tex:349` promises it in `sec:results_pilot`, `chapter_5.tex:23`
   slots it there, and `chapter_4.tex:384` carries the same columns beside the
   formulation-study results table. The manuscript's two appendices are ethics
   documents and resource persons, and both are commented out of `main.tex`, so an
   appendix was never a place this table could have gone. What is amended here is
   *where* the reporting happens, not *what* is reported — the three columns and the
   superseded verdict are unchanged and are delivered as promised.

**Acknowledged risk, stated before the result was known:** a minimum-common-budget
rule will tend to favour whichever formulation transmits fewest bytes, and which
one that is had not been computed. The rule was fixed precisely so it could not be
tuned once that became visible.

### Formulation 2 is frozen

Under the amended criterion the skews diverge — Formulation 2 wins α = 0.1 by a
real margin (~2.1pp over the runner-up, with Formulation 4 collapsing entirely, so
the threshold-staged rule does not survive severe skew), while α = 1.0's top four
span 0.45pp and **does not discriminate and is not read as if it did.** The
severe-skew tie-break fires. `frozen_formulation: 2`; the contribution is not
withdrawn.

The robustness columns emitted alongside show first-touch qualifying exactly one
cell, 5-consecutive none, and the final-round gate none. Three of the four rules
return no usable verdict at either skew. **The amended criterion is the only one
that ranks all five arms at both** — which is the argument made in advance and not
tuned afterwards.

`conf/algorithm/fedmaq.yaml` moved from `formulation: 3` to `2`. The contribution
claim survives and is *strengthened in kind* rather than merely rescued:
Formulation 2 is multiplicative in both Tier-2 signals, so multi-adaptive fusion is
what the freeze selected. Formulation 3 modulates a gradient primary; Formulation 0
uses no soft signal at all.

**Not claimed:** that Formulation 2 is better than 1 or 0 in any general sense. At
α = 1.0 it is inside noise of both. The defensible statement is that it is the
pre-registered winner under a criterion fixed before the result was visible, that
it wins the discriminating cell by a real margin, and that it is the only survivor
consistent with the multi-adaptive claim.

### The ablation arms were re-expressed — a contingency executing, not a repair

`conf/matrix/ablation.yaml`'s header carried this, dated, since before the study
ran. Both changes were made:

- **Configuration 3** (`fedmaq_no_data`): `lambda_val: 0.0` → `gamma2: 0.0`.
- **Configuration 4** (`fedmaq_no_state`): three keys collapse to `gamma1: 0.0`,
  and **the fallback-arm exception is retired.** It existed only because
  Formulation 3 carries no weight on its gradient term and so cannot express
  state-awareness removal at any parameter setting, forcing that arm onto a
  different formulation and a different parity anchor. Under the multiplicative
  form the removal is expressible in place, the arm nests like every other, and its
  anchor reverts to Configuration 7.

**No renormalization of the surviving exponent, deliberately.** Under the linear
form, removing one weight caps the survivor at half the soft range, so it had to be
lifted — a second difference forced by the form, and the reason the old arm needed
three keys. Multiplicatively `x^0.5` already spans [0, 1]; lifting the exponent
would change only response curvature and add a difference the ablation would then
have to attribute. The arms are now exact mirror images, which is a cleaner
leave-one-out than the design it replaces.

The completed ablation runs at the old formulation **are invalid, and their output
directory is moved aside rather than resumed** — `--skip_completed` keys on
per-run final-round checkpoints and would silently retain the old arms. Moved
rather than deleted: the Configuration 7 cell remains usable as a reference.

### The reserved recheck is degenerate and is not spent

`resolve_frozen_formulation` sets `recheck_required: true` mechanically whenever the
frozen formulation is not the incumbent, so the flag fired. **Its content is
empty.** The recheck is defined as surviving layer vs. unrefined under the winning
formulation, and ADR-0008 froze the surviving layer **empty** — both arms resolve
to byte-identical configurations.

This is the same degeneracy that skipped the Stage 3 confirmation, applied to the
same empty layer, so it is a precedent *inside* this pre-registration rather than a
new exemption invented for convenience. It cannot be substituted with a re-run
factorial: a veto can only remove mechanisms, and removing from the empty set
yields the empty set whatever formulation it runs under.

Disclosed as a limitation rather than hidden: the sequential design's known gap — a
refinement helping under one formulation and hurting under the winner would go
uncaught — is **vacuous here**, there being no surviving mechanism whose value
could depend on the formulation. `resolve_frozen_formulation` emits
`recheck_discharged` and `recheck_note` beside `recheck_required`, because the flag
alone reads as an owed recheck.

## Consequences

- The frozen formulation is a single scalar in `conf/algorithm/fedmaq.yaml`, behind
  the tag (ADR-0010). Changing it invalidates every ablation arm.
- Ablation arm differences are pinned by `ABLATION_ARM_DIFFS` and a test asserting
  both single-signal removals are exact and symmetric under the frozen formulation.
- Live results measured under this freeze — including how FedMAQ compares against
  the uncompressed control at equal bytes — are **findings, not decisions**, and
  live in the pinned results Issue, not in this directory.
