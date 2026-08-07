# ADR-0011 — Baseline matched tuning: the null result is the product

**Status**: Accepted · 2026-08-01, executed 2026-08-05, analyser committed 2026-08-06
**Supersedes**: `docs/DECISIONS.md` Decisions 67, 73, 81, 87 (file deleted; see ADR-0014)

## Context

ADR-0004 promises baseline parity through "matched light tuning," and the
manuscript promises the baseline hyperparameter table is locked and tagged at the
freeze. Nothing produced it. There was no matrix file, no stage in the runbook, and
the word "tuning" appeared in the methods chapter only in the phrase "rather than a
single unstructured tuning pass."

**The gap is transfer, not provenance.** Every baseline constant *was* sourced
carefully — FedProx's μ cited to the image benchmarks of its own paper's sweep,
DAdaQuant's φ derived from a published rule instantiated at R = 100, FedPAQ's
8 bits adopted from the precision another paper benchmarks it at (FedPAQ publishes
levels rather than bit-widths), FedKD's SVD endpoints declared as ours rather than
inherited. What no published value can establish is that it still holds **on
MobileNetV2GN at α ∈ {0.1, 1.0}**, which is a different architecture and a
different skew from any of the sources. FedMAQ meanwhile received a full
exploration phase and a formulation study on exactly this configuration.

*You verified yours on this grid and theirs on someone else's* is the surviving form
of the attack, and it had no answer.

## Decision

### The stage

`conf/matrix/baseline_tuning.yaml`, **Stage 1b**. Five baselines (FedAvg is the
uncompressed control and has no knob), one key hyperparameter each: a five-seed
reference cell at the shipped value plus two three-seed challengers. Run at the
held-out α = 0.3, under FedMAQ's own √2σ rule (ADR-0008), and uncounted among the
reported grid.

**R=100, not the R=50 every other exploration stage screens at.** Baselines get no
confirmation stage, so a truncated-horizon pick would ship into the reported grid
uncorrected. That is most of the stage's GPU cost and it is the one place worth
spending rather than economizing. Reference cells are deepened to five seeds by the
same argument applied per baseline; standardizing to three would leave the
baselines' margins estimated more coarsely than FedMAQ's own freeze gate, which is
an indefensible ordering.

**The knob is the one governing each baseline's own accuracy–communication
trade-off**, because that is the axis every claim rests on. This corrects an
earlier roster that named "FedDistill/FedKD distillation temp": FedDistill has no
temperature (its knob is `reg_alpha`), and FedKD's `temperature` governs
client-side mutual KD while `tmax`, the SVD energy cutoff, governs the trade-off
this grid compares on. Recorded rather than silently substituted.

**The null result is the product.** The likely outcome is that no challenger clears
and every baseline freezes at the value it would have had with no sweep — which
converts an appeal to authority into a measurement. A cheaper selective sweep
("tune only the knobs that look sensitive") was rejected for its *shape* rather
than its cost: the person whose algorithm benefits would be deciding which of his
competitors' knobs deserved tuning, which invites the exact suspicion the sweep
exists to dispel. **Uniform treatment has no soft spot.**

Stage 1b's placement is not an ordering constraint — it shares no configuration
with FedMAQ, so it is unordered with respect to the refinement search. The only
hard ordering the tag imposes is that it must finish before the tag.

### DAdaQuant was capped at five bits by a constant of ours that read as eight

`dadaquant.yaml` carried `q_max: 8` beside `fedpaq.yaml`'s `q: 8` and
`fedmaq.yaml`'s `q_max: 16`, and **the three do not denominate the same quantity.**
FedPAQ's `q` and FedMAQ's bounds are bit-widths; DAdaQuant's `q` counts
quantization levels *per sign* — codes in [−q, q], so 2q+1 levels and
⌈log₂(2q+1)⌉ bits per element. The hook's docstring states this plainly; nothing
outside that docstring did. Eight levels per sign is five bits, so the baseline a
reader would take to be matched to FedPAQ's 8-bit budget was running at roughly
five-eighths of it.

Two consequences, neither disclosed anywhere. The precision ceiling of a competitor
on the exact frontier every claim is drawn on was set by a constant we chose,
absent from the baseline table and not among the values the tuning stage tests. And
worse for the baseline's own integrity: `q_t` starts at `q_min = 1` and doubles, so
a ceiling of 8 is reached after three doublings — around round 30 of 100 — leaving
DAdaQuant effectively static for the remaining seventy rounds. **Time-adaptive
escalation is what DAdaQuant *is*; the cap suppressed most of it.**

**Resolved: `q_max = 127`.** That is 255 codes, exactly eight bits, so DAdaQuant's
ceiling equals FedPAQ's and the two pure-quantization baselines are separated by
*adaptivity* rather than by budget. `q_min = 1` is left alone — it is published.
Some upper bound is still required, so the change is which bound, not whether.

**Direction of the correction.** It strengthens a competitor. That is deliberate:
of the two available fixes — disclosing the 5-bit cap and leaving it, or matching
the budget and disclosing that — only the second cannot be read as the author of
FedMAQ choosing how much precision his competitors are allowed.

### Verdicts

Two of five constants moved off their published values; three were retained.

| Baseline | Knob | Reference | Verdict |
| --- | --- | --- | --- |
| FedProx | `mu` | 1.0 | **→ 0.01** (both challengers cleared; larger delta won) |
| FedDistill | `reg_alpha` | 1.0 | **→ 0.5** (cleared) |
| FedPAQ | `q` | 8 | retained (neither challenger cleared) |
| DAdaQuant | `phi` | 10 | retained (noisiest cell; margin correspondingly permissive) |
| FedKD | `tmax` | 0.95 | retained |

FedKD's absolute level sits well below the other baselines. That is architectural —
its student is a width-0.5 MobileNetV2GN against a full-size teacher (ADR-0005) —
**not a tuning failure. Do not read that row as a bug.**

### The analyser had to exist in committed code before the tag

`baseline_tuning.yaml`'s header named `scripts/analysis.py:exploration_noise_margin`
as its decision rule. That function filters `phase == EXPLORATION_PHASE` and
`algorithm == "fedmaq"` and keys cells on refinements, so it **structurally cannot
read Stage 1b's runs** and reports a completed 55-run stage as no runs at all. The
verdicts above were computed by an ad-hoc script pasted into an allocation session
and existed nowhere in the repository.

`baseline_tuning_margin()` now applies the same rule from committed code — σ from
each baseline's shipped-value reference cell, margin √2σ, adoption only on a
strictly clearing delta, tie-break by larger delta — writing to
`scripts/analysis_output/baseline_tuning_margin.json`. `RunRecord` gained `variant`
(ADR-0009) because Stage 1b's cells differ in nothing else the record holds: same
algorithm, same config name, same group, same skew.

Tests pin both outcomes reconstructed from the published statistics, including the
FedPAQ case where **both challengers score higher and neither is adopted** — the
retention case the rule exists for. A regression test pins the old defect directly.

**Why this had to precede the tag.** The tag freezes the baseline table. Tagging a
table whose values cannot be recomputed from the tagged tree is the weaker
artifact, and the tag is the thing one does not want to cut twice (ADR-0010).

## Consequences

- The revised constants ship in `conf/algorithm/{fedprox,feddistill}.yaml` and are
  frozen behind the tag; the manuscript's baseline table carries them.
- A future baseline added to the stack owes the same treatment — one knob, a
  five-seed reference, two three-seed challengers, √2σ — or an explicit statement
  of why it is exempt.
- `exploration_noise_margin` and `baseline_tuning_margin` are separate functions by
  necessity, not duplication. Do not merge them.
