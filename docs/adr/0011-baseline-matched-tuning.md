# ADR-0011 — Baseline matched tuning: the null result is the product

**Status**: Accepted · 2026-08-01, executed 2026-08-05, analyser committed 2026-08-06
**Supersedes**: `docs/DECISIONS.md` Decisions 67, 73, 81, 87 (file deleted; see ADR-0014)

## Current amendment — widened reporting stage before Stage 1a

The current campaign uses `conf/matrix/baseline_tuning_wide.yaml`. Each tunable
algorithm receives a shipped-reference cell and an equally specified challenger
curve; FedMAQ varies `q_max` while the hardware-grounded `c_unit` remains fixed.
The matrix is authoritative for the cell roster and metadata.

Adoption remains a strict delta greater than `sqrt(2) * sigma` of the reference
cell. Every challenger point is reported, and a highest point that does not clear
the margin is not adopted. The stage completes before
`pre-registration-stage1a`, because its verdicts configure the formulation study
and downstream benchmark. Shipped-reference and source-paper-default metadata are
kept distinct. Run state and evidence belong to the execution model and pinned
Issues, not this ADR.

## Context

ADR-0004's matched-light-tuning promise requires each baseline's own
accuracy--communication knob to be checked on the thesis configuration. Published
defaults do not establish transfer to this architecture and skew; this decision
therefore treats tuning as a uniform, pre-declared comparison rather than an
author-selected sensitivity search.

## Decision

### The stage

`conf/matrix/baseline_tuning.yaml`, **Stage 1b**. Five baselines (FedAvg is the
uncompressed control and has no knob), one key hyperparameter each: a five-seed
reference cell at the shipped value plus two three-seed challengers. Run at the
held-out α = 0.3, under FedMAQ's own √2σ rule (ADR-0008), and uncounted among the
reported grid.

The stage uses the horizon and reference depth declared by its matrix. Baselines
get no later confirmation stage, so the reference must support the adoption margin
before the reported grid is frozen.

**The knob is the one governing each baseline's own accuracy–communication
trade-off**, because that is the axis every claim rests on. This corrects an
earlier roster that named "FedDistill/FedKD distillation temp": FedDistill has no
temperature (its knob is `reg_alpha`), and FedKD's `temperature` governs
client-side mutual KD while `tmax`, the SVD energy cutoff, governs the trade-off
this grid compares on. Recorded rather than silently substituted.

**The null result is the product.** If no challenger clears, each baseline retains
its shipped value. Selective tuning is rejected because the author would choose
which competitors' knobs deserve attention; uniform treatment has no soft spot.

Stage 1b's placement is not an ordering constraint — it shares no configuration
with FedMAQ, so it is unordered with respect to the refinement search. The only
hard ordering the tag imposes is that it must finish before the tag.

### DAdaQuant's unit distinction

DAdaQuant's `q` counts quantization levels per sign, whereas FedPAQ's `q` and
FedMAQ's bounds are bit-widths. The durable correction is `q_max = 127`, giving
255 codes and an eight-bit ceiling while leaving the published `q_min` unchanged.
This strengthens the competitor and separates the pure-quantization baselines by
adaptivity rather than by an author-chosen precision budget.

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

### Reproducible analysis boundary

`baseline_tuning_margin()` is the committed implementation of the margin rule;
the matrix and analysis outputs are its execution/evidence owners. The analyser
must exist before the pre-registration tag so the frozen baseline table can be
recomputed from the tagged tree. The implementation details and regression checks
remain in code and tests rather than being duplicated here.

## Consequences

- The revised constants ship in `conf/algorithm/{fedprox,feddistill}.yaml` and are
  frozen behind the tag; the manuscript's baseline table carries them.
- A future tunable algorithm added to the stack owes the widened treatment — one
  trade-off knob, a five-seed shipped reference, four three-seed challengers,
  complete reporting, and the unchanged √2σ adoption rule — or an explicit
  statement of why it is exempt.
- `exploration_noise_margin` and `baseline_tuning_margin` are separate functions by
  necessity, not duplication. Do not merge them.
