# ADR-0008 — The exploration protocol, and the refinement layer that froze empty

**Status**: Accepted · 2026-07-18, protocol rewritten 2026-07-26, executed 2026-08-02
**Supersedes**: `docs/DECISIONS.md` Decisions 27–31, 33–35, 53, 61, 70, 79, 80 (file deleted; see ADR-0014)

## Context

FedMAQ carries a *refinement layer* — optional mechanisms (`soft_voting`,
`ema_student`, `grad_norm_ema`) layered on top of its two-tier precision scaling.
Exploration exists to decide which of them ship. ADR-0004 makes that decision
binding: whatever survives exploration is frozen behind the pre-registration tag
and the confirmatory grid runs it unchanged.

The first version of this protocol had no variance estimate and picked winners by
rank. It was replaced before it could reach a freeze.

## Decision

### Structure

**Explore-α = 0.3**, held out from the confirmatory report grid's {0.1, 1.0}, so
the single frozen configuration is not selected on the exact skews it will later be
reported at. **50 rounds, single seed** per screening run — cheap enough for
repeated passes. Mechanisms are grouped into passes rather than one joint factorial
(cost) or pure sequential coordinate descent (misses interactions), and every
mechanism setting includes its control/off arm.

FedMAQ's mechanisms are fully resolved before baseline matched-tuning (ADR-0011)
starts. Baseline hyperparameters do not depend on FedMAQ's mechanism choices, so
sequential ordering costs nothing and avoids redoing baseline tuning if a later
pass moves the FedMAQ config.

### The decision rule, and the rule that replaced it

The original rule was "keep/drop/revise only if the delta clears a noise margin."
No numeric margin was ever sourced for it — single-seed runs give no variance
estimate, and no repeated-seed characterization on MobileNetV2GN existed. The first
pass resolved this pragmatically by letting the empirical cluster band (~1.6pp
across 14 non-outlier sweep cells) stand in as a noise floor, and produced a
*provisional* pick of `entropy_weight=2.0`, `precision_weight=0.5`.

**That whole procedure is superseded.** σ is now measured from the **unrefined
reference cell** of the stage making the call, at the held-out α = 0.3, and the
keep-or-drop threshold is **√2σ**. It is implemented in
`scripts/analysis.py:exploration_noise_margin` and dispatched by
`conf/matrix/pass2_factorial.yaml`.

Two properties of that function are load-bearing and were once visible only in a
Python default argument:

- **σ is never pooled across stages.** The function takes an `experiment_group` and
  *refuses* to pool — the factorial and the R=100 confirmation each compute their
  own.
- **The rule states the mechanism, not seed counts.** Three sites once said "three
  seeds" while the reference cell was deepened to five, so they contradicted each
  other. The protocol states the rule; a single site owns the seed arithmetic.

Consequently the shipped refinement-flag values are an **exploration outcome**, not
a decision awaiting a flip. Do not cite the superseded provisional pick as settled,
and read `fedmaq.yaml`'s flags as findings only once exploration has run.

### `client_kd_reg` and `kd_prox_mu` are retired from the roster

An earlier decision put "client-KD-reg + proximal (μ)" on the Pass 2 roster and
another closed with "`client_kd_reg=true` runs as-is in Pass 2." It does not run:
the factorial is a 2³ over `soft_voting`, `ema_student` and `grad_norm_ema`, and
`fedmaq.yaml` ships `client_kd_reg: false`, `kd_prox_mu: 0.0`. A mechanism had left
the roster with a decision on record saying it would be evaluated and none retiring
it.

**Retired rather than reinstated.** Adding a fourth factor makes Stage 2 a 2⁴
factorial, doubling it from 26 runs to 50, for a mechanism whose only prior
evaluation (ADR-0001) concluded its implementation cost was not worth paying. The
code path remains for reproducibility, as with the dropped baselines.

### Outcome: the surviving set is empty, and FedMAQ ships unrefined

All 26 `pass2_factorial` runs completed clean and the noise-margin verdict is an
**empty surviving set** — no cell clears the margin. `soft_voting` alone posts the
best delta and does not clear it; `grad_norm_ema` and `ema_student` are each
individually *negative*, with `ema_student` destabilizing rather than merely flat.

This was checked against the two ways it could be an artifact:

- **Not a data-quality artifact.** Per-seed accuracies were pulled directly for all
  three single-mechanism cells. The unrefined cell's spread has no single outlier
  driving it — it is real seed-to-seed variance at α = 0.3. `soft_voting`'s three
  seeds sit *inside* the unrefined range and *below* unrefined's best seed. A
  mechanism that cannot beat the control's best draw has not distinguished itself
  from noise.
- **The margin was not relaxed post hoc.** `soft_voting` is the maximum delta of
  seven cells, and the expected maximum of seven noisy comparisons is positive even
  under a pure null. "The best cell looks promising" is the multiplicity trap the
  margin exists to control, not evidence the margin is miscalibrated. A Welch's-t
  reframing was considered and rejected: it ignores the best-of-7 selection
  entirely, and the only reason it reads as borderline is `soft_voting`'s tight but
  noise-unreliable 3-seed variance — exactly why the protocol keys the margin off
  the 5-seed reference arm.

**The pre-registered empty-freeze branch (ADR-0010) executed directly**, and the
Stage 3 confirmation was **skipped as degenerate**: with an empty surviving set,
both of its arms would have been config-identical to unrefined, spending ~10h of
allocation time re-testing unrefined against itself. This is a documented
interpretation rather than something the manuscript's stage description
anticipates, and is flagged as such rather than silently elided.

What landed:

- `conf/algorithm/fedmaq.yaml`: all three flags `false`, with an inline comment so
  a future edit does not casually flip them back to chase an improvement.
- **Ablation Configuration 8 dropped** — nothing left for it to remove.
  `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` already
  enforced this and started passing once the freeze landed.
- `conf/algorithm/fedavg_kd.yaml` (Configuration 6): `ema_student` → `false`, per
  the existing rule that ablation arms mirror the frozen value. Not a new decision
   — the existing rule's output changing with the freeze.
- The ablation grid test derives its expected run count from
  `_frozen_refinements()` instead of a literal, so it needs no hand-edit at the
  next freeze.

## Consequences

- **No subset retries, no tuning rescue.** The empty set is the verdict, and
  re-opening the factorial is pre-registered against.
- The reserved refinement recheck is also degenerate under this freeze — see
  ADR-0012, which records why removing from an empty set yields the empty set.
- Anything that would re-populate the refinement layer would gate the ablation,
  which inherits the layer through `defaults: [fedmaq, _self_]`. Nothing can, so
  the ablation is not gated from this direction.
- Confirmatory run counts moved when Configuration 8 dropped. Counts live in the
  pinned dispatch Issue, not here — see ADR-0014.
