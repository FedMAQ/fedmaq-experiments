# ADR-0011 — Baseline matched tuning: the null result is the product

**Status**: Accepted · 2026-08-01, executed 2026-08-05, analyser committed 2026-08-06;
amended 2026-09-13 ([#107](https://github.com/FedMAQ/fedmaq-experiments/issues/107))
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

### Verdicts (Stage A, widened) — supersedes the Stage 1b table below

`baseline_tuning_margin()` (`scripts/analysis.py:721-760,777`) re-anchors every
baseline to its source-paper default as the reference cell, discarding the
narrow Stage 1b table below as the baseline. Sigma and the `sqrt(2) * sigma`
margin are computed from that paper-default cell. This is a uniform, deliberate
property fixed at matrix-authoring time, not an incidental fact about any one
algorithm — see "Reference cell equals the paper default" below.

| Baseline | Knob | Reference (= paper default) | Verdict |
| --- | --- | --- | --- |
| FedProx | `mu` | 1.0 | retained — no challenger cleared the margin |
| FedDistill | `reg_alpha` | 1.0 | retained — no challenger cleared the margin |
| FedPAQ | `q` | 8 | retained (see evidentiary note) |
| DAdaQuant | `phi` | 10 | **→ 1** — a challenger cleared the margin by +3.81pp on MobileNetV2GN |
| FedKD | `tmax` | 0.95 | retained (see evidentiary note) |
| FedMAQ | `q_max` | 16 | retained (see evidentiary note) |

FedProx and FedDistill's retentions are directly evidenced: commit `11934ba`
rewrote `conf/algorithm/fedprox.yaml` (`mu: 0.01` → `1.0`) and
`conf/algorithm/feddistill.yaml` (`reg_alpha: 0.5` → `1.0`), each with a shipped
comment naming "no challenger cleared" under Stage A `baseline_tuning_wide`.
DAdaQuant's adoption is likewise directly evidenced: the same commit rewrote
`conf/algorithm/dadaquant.yaml` (`phi: 10` → `1`) with a shipped comment naming
the cleared margin and the +3.81pp figure.

**Evidentiary note — FedPAQ, FedKD, FedMAQ.** `git show --stat 11934ba` touched
only `dadaquant.yaml`, `feddistill.yaml`, `fedprox.yaml`, and
`docs/freeze/source_manifest.json` — it did not touch `fedpaq.yaml`, `fedkd.yaml`,
or `fedmaq.yaml`, and none of those three files carries a comment recording a
Stage A verdict. Their current shipped values (`q: 8`, `tmax: 0.95`, `q_max: 16`)
equal `BASELINE_TUNING_WIDE_REFERENCE_VARIANTS`' reference-cell entries for those
algorithms (`scripts/analysis.py:680-690`), which is consistent with retention,
and `retained_shipped_value` (`scripts/analysis.py:1020,1065`, computed as
`adopted is None`) is the field that would confirm it directly from a captured
margin computation. No such captured output for the real 145-cell Stage A run is
available locally — results live on the allocation
(`docs/agents/execution-model.md`), not in this tree — so this ADR states the
fact plainly rather than asserting the verdict: these three rows are retained
**by absence of an edit and by matching the reference-cell constant**, not by a
verdict comment or a locally re-derived analysis output. This is a weaker class
of evidence than FedProx/FedDistill/DAdaQuant's shipped comments, and is recorded
as such rather than smoothed over.

### Disclosed post-hoc amendment — issue #107

ADR-0011 and `conf/protocol/replacement-v1.yaml` were last edited 2026-09-04
09:52 UTC, ~7h before Stage A was dispatched (2026-09-04T17:06:46Z) — a genuine
pre-registration existed before Stage A ran. A 2026-09-09 audit
([#105](https://github.com/FedMAQ/fedmaq-experiments/issues/105)) found five gaps
in it and filed [#107](https://github.com/FedMAQ/fedmaq-experiments/issues/107)
(2026-09-09T06:01:30Z) asking that they close "before any Stage-A verdict is
read." Commit `11934ba` (2026-09-10T18:18:58Z, ~36h17m after #107, zero
intervening commits to either target file) read Stage A's verdicts into the
shipped algorithm configs before this amendment landed — #107's own deadline was
missed.

**This is a disclosed post-hoc amendment to the pre-registration, not a
correction of an implementation error**, following the precedent set by
[ADR-0012](0012-formulation-selection-and-the-iso-byte-amendment.md)'s criterion
amendment. Each of the five gaps below is a structural fact fixed at
matrix-authoring time, before any Stage A result existed — closing them late
documents what the pre-registration already meant, it does not select among
outcomes with the verdicts in hand. The distinction that would matter — and does
not apply here — is amending a rule *because* a result was seen; none of the five
gaps changes which challenger was adopted for any baseline.

1. **Iso-byte scalar.** `baseline_tuning_wide.yaml`'s scalar comparison point is
   the same minimum-common-cumulative-MB rule [ADR-0012](0012-formulation-selection-and-the-iso-byte-amendment.md)
   already defines for formulation selection, reused rather than redefined; this
   ADR did not separately restate it before now, which is the gap #107 named.
2. **`mean_terminal_mb` tie-break.** See "Tie-break registration" below.
3. **`post_process` convention.** `post_process: false` ships in every
   `conf/algorithm/*.yaml` used by this stage (confirmed by inspection: fedavg,
   dadaquant, cfd, fedmaq, feddistill, fedkd, fedmd, fedprox, fedavg_kd, fedpaq
   all ship `false`; only `fedpaq_pipeline.yaml` ships `true`, and that file is
   not part of this stage). `baseline_tuning_wide.yaml` overrides it to `true`
   for FedMAQ's four `q_max` arms specifically
   (`conf/matrix/baseline_tuning_wide.yaml:175,181,187,193`), enforced by
   `test_wide_baseline_tuning_adds_fedmaq_and_four_challengers`
   (`tests/test_config_and_dispatch.py:700-749`). This is the FedMAQ-only
   post-processing pipeline (§4.3) turned on for FedMAQ's own arms and left off
   for every baseline that has no such pipeline — not an inconsistency, but
   previously undocumented at the ADR level. See also the one-line addition to
   `docs/agents/execution-model.md`'s `post_process` ledger.
4. **Arm-count asymmetry.** FedMAQ contributes four arms (one reference + three
   challengers, `q_max ∈ {4, 6, 8, 16}`); every other baseline contributes five
   (one reference + four challengers). The asymmetry runs against FedMAQ, not in
   its favor. The four values are a subset of FedMAQ's canonical `bit_widths`
   ladder (`[2, 3, 4, 5, 6, 7, 8, 16]`, `conf/algorithm/fedmaq.yaml`) chosen at
   matrix-authoring time; no further subset-selection rationale is recorded in
   this repo, and none is invented here.
5. **Reference cell equals the paper default, uniformly.** For all five original
   baselines *and* FedMAQ, Stage A's reference cell is the source-paper default
   (`fedprox: mu1p0`, `fedpaq: q8`, `dadaquant: phi10`, `feddistill: a1p0`,
   `fedkd: t0p95`, `fedmaq: qmax16` — `BASELINE_REFERENCE_VARIANTS`/
   `BASELINE_TUNING_WIDE_REFERENCE_VARIANTS`, `scripts/analysis.py:680-690`).
   Stage A re-anchors every baseline to its source-paper default as the
   reference cell, discarding the narrow Stage 1b verdicts as the baseline;
   sigma and the margin are computed from that paper-default cell.
   `shipped_adopted_variant` is provenance metadata for the superseded Stage 1b
   pilot and is never consumed by the margin computation
   (`scripts/analysis.py:721-760,777`). This is not an incidental fact about two
   algorithms — it is the uniform mechanism that produced this campaign's one
   adoption (DAdaQuant, above): the challenger cleared margin computed from the
   paper-default `phi10` cell, not from Stage 1b's narrow verdict.

### Tie-break registration — `mean_terminal_mb`

`select_power_mean_degree_iso_byte`'s tie-break (mean terminal cumulative MB
across arms, used when the primary adoption margin does not discriminate) is
registered here in prose. `conf/protocol/replacement-v1.yaml`'s
`selection_domains` block is deliberately **not** amended to add it:
`preregistration_contract()` (`src/fedmaq/core/protocol.py:43-68`) includes
`selection_domains` in the hashed contract (line 67) — unlike `matrix_contracts`,
which [ADR-0023](0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md)
documents as excluded from the same hash. Amending `selection_domains` now would
move `stage_1a`'s `preregistration_sha256` away from the value the currently
in-flight 84-cell sweep already stamped. This mirrors ADR-0023's handling of the
identical collision one layer up (widening `power_mean_design` to 126 cells
without moving the hash, because that change lived in `matrix_contracts`): the
rule is disclosed here, in prose, precisely because the file it would otherwise
live in is frozen mid-campaign.

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

### Verdicts (Stage 1b, narrow campaign — superseded, historical record only)

**Superseded by Stage A's widened re-run; see "Verdicts (Stage A, widened)" above
for the current shipped state.** Retained below as the historical record of what
the narrow `baseline_tuning.yaml` campaign concluded at the time; it is not the
current shipped configuration — Stage A's widened re-run reversed both the
FedProx and FedDistill rows back to their paper defaults (`mu: 1.0`,
`reg_alpha: 1.0`), and DAdaQuant separately moved off this table's reference
value under Stage A. Do not read this table as describing any shipped config.

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

- The Stage A verdicts ship, not the Stage 1b table's: `conf/algorithm/fedprox.yaml`
  retains `mu: 1.0`, `conf/algorithm/feddistill.yaml` retains `reg_alpha: 1.0`, and
  `conf/algorithm/dadaquant.yaml` adopts `phi: 1` — all three behind commit
  `11934ba`, disclosed above as a post-hoc amendment. `conf/algorithm/fedpaq.yaml`
  (`q: 8`) and `conf/algorithm/fedkd.yaml` (`tmax: 0.95`) are unedited and match
  their reference-cell constants; `conf/algorithm/fedmaq.yaml` (`q_max: 16`) is
  likewise unedited. The manuscript's baseline table must carry the Stage A
  verdicts, not the superseded Stage 1b ones.
- A future tunable algorithm added to the stack owes the widened treatment — one
  trade-off knob, a five-seed shipped reference, four three-seed challengers,
  complete reporting, and the unchanged √2σ adoption rule — or an explicit
  statement of why it is exempt.
- `exploration_noise_margin` and `baseline_tuning_margin` are separate functions by
  necessity, not duplication. Do not merge them.
- `conf/protocol/replacement-v1.yaml`'s `selection_domains` is intentionally left
  unamended by this amendment; the `mean_terminal_mb` tie-break is registered in
  this document's prose only, per "Tie-break registration" above.
