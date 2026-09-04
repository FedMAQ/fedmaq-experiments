# ADR-0004 — Iso-architecture, the confirmatory grid, and the explore/confirm boundary

**Status**: Accepted · 2026-07-16, extended 2026-07-26 and 2026-08-01
**Supersedes**: `docs/DECISIONS.md` Decisions 1–13, 52, 75 (file deleted; see ADR-0014)

## Context

The thesis compares FedMAQ against a baseline stack on image classification under
non-IID partitioning. Before any run was dispatched, three things had to be fixed
together: what every arm trains, what the reported grid contains, and where the
line falls between exploration (adaptive, cheap) and confirmation (frozen, reported).
Leaving any of them implicit would let a later result reshape the design that
produced it.

## Decision

**Iso-architecture.** Every algorithm — FedMAQ and all baselines — trains
**MobileNetV2GN (~2.24M params)**. Baselines were re-run from scratch on it; the
older ResNet18GN standings are retired. The switch is justified by *edge realism
only*: compression ratio (~1.7×) is model-independent at iso-architecture
(ratio = `avg_bits/32`), so no "improved communication savings" claim rests on it.
FedKD is a deliberate compact-student exception (ADR-0005).

**FedMAQ-Lite is dropped** from the formal thesis. At iso-architecture its
SimpleCNN (2.16M) is the same size as the main model, so its size-contrast story
*was* the confounded cross-architecture comparison being retired. Smoke results
move to the exploration appendix.

**Contribution is mechanism-primary**: quantization-robust accuracy under
heterogeneity. Communication savings are reported honestly as secondary (~1.7×).
The ablation table is the headline evidence, not the communication number.

**Methodology.**

- **One frozen FedMAQ setting**, held across datasets and α. Regime dependence is
  reported as a sensitivity study and transfer failure as a finding, never used
  to re-select a downstream configuration.
- **Paired seeds, descriptive interpretation**: every arm shares the same three
  seeds with *identical partitions*. Per-seed deltas, their mean, and their range
  describe direction and stability; three seeds do not support an inferential
  significance claim. The determinism this rests on is ADR-0006.
- **Baseline parity is matched light tuning**: each baseline gets an equal small
  budget on its key hyperparameter, frozen before confirmation (ADR-0011).
- **Hard explore/confirm freeze**: exploration is adaptive, single-seed, cheap, and
  mechanisms are up for debate. It ends by pre-registering a frozen configuration
  and a fixed mechanism set behind a git tag. The confirmatory grid runs frozen.
  A surprise during confirmation becomes a documented finding or a new labelled
  exploration round — never a silent edit. The machinery enforcing this is ADR-0010.

**The grid.** CIFAR-10, CIFAR-100 and FEMNIST, **3 seeds × 100 rounds**.
α ∈ {0.1, 1.0} for the CIFAR datasets (severe and moderate extremes; intermediate
values dropped to cut runs); FEMNIST uses writer partitioning and has no α.
Exploration and freeze happen on **CIFAR-10 as primary**, transferring the frozen
setting to CIFAR-100/FEMNIST. Transfer failure is documented as a finding and
does not permit a per-dataset re-freeze. The ablation is an additive ladder
(narrative) plus leave-one-out (rigorous attribution), run on CIFAR-10 at both
skews only.

**Run counts are not recorded here.** They have moved repeatedly (183 → 177 when
Configuration 8 dropped, ADR-0008) and belong to live dispatch state, not to a
decision record. The pinned dispatch Issue is the single source.

**Config-as-code registry.** A manifest enumerates every formal run
(algorithm × dataset × α × seed), hashes frozen configs, and drives the
process-isolated runners. Hydra `--multirun` is never used. The two-phase layout
enforces the freeze boundary structurally rather than by convention. Output-path
and run-identity mechanics are ADR-0009.

### The comparison regime governs `post_process`, not the algorithm config

The error-feedback / difference-coding / zlib pipeline belongs to the primary
benchmarking grid and to nothing else. It shipped `false` in every
`conf/algorithm/*.yaml`, so the grid would have reported communication numbers
without the pipeline it claims to run — but flipping it in `fedmaq.yaml` is worse
than the bug, because that file is also the formulation study's config and Ablation
Configuration 7, and the ablation inherits Configuration 7 from the primary grid.
A pipeline present there but not in the arms puts every arm two removals from its
reference rather than one, with the damage landing on the communication axis where
the ablation's claims live.

**The governing rule: a FedMAQ run carries the pipeline if and only if the runs it
is compared against carry it.** The flag is overridden per matrix file, never set
in an algorithm config. The control arm's partner is the primary grid, so it
carries the pipeline; the formulation study and every ablation arm are contrasted
within their own sets, so they do not. Both directions are enforced in
`tests/test_config_and_dispatch.py`.

### The uniform-memory control arm is the memory-blind condition

§4.1 once claimed this arm "isolates the accuracy-recovery capabilities of
server-side knowledge distillation from the confounding variable of memory-based
gradient quantization." It does not and cannot: distillation is active and
identically configured on both sides of that contrast, so the contrast holds it
fixed rather than pricing it. The arm that prices distillation is Configuration 5
(`fedmaq_no_kd`).

At `c_unit = 1024` MB, `floor(16384 / 1024) = 16`, which equals `fedmaq.yaml`'s
`q_max`; the Tier-2 target is itself clamped to `q_max`, so `min(Q_k^max, q_hat)
= q_hat` for every client. The Tier-1 ceiling does not merely stop *varying* — it
never binds at all. The arm is FedMAQ in the memory-blind condition, which is
Ablation Configuration 2, and the two assign **identical bit-widths**.

**The arm is set to 16384 MB and reframed.** This is the smallest value under the
current calibration that makes the clamp non-binding at `q_max = 16`. It is the
upper endpoint of the variable-capacity law, not a central-tendency control. Its
purpose is to price memory blindness on FedMAQ itself.

Its independent content is the **coding regime**, not the condition: the ablation
runs pipeline-free while every headline comparison runs with error compensation
active, and error compensation is the one downstream mechanism that could absorb a
clamp-induced loss of precision. Agreement between the two transfers the ablation's
price for memory blindness to the regime the reported claims live in; disagreement
localizes the difference to the pipeline.

## Consequences

- `test_uniform_memory_arm_is_the_memory_blind_condition` composes the shipped
  configs through Hydra and asserts the equivalence across all five formulations,
  plus that a low-capacity client *is* clamped. The claim holds only while
  `floor(uniform_memory_mb / c_unit) >= q_max` — raise `c_unit`, lower the
  capacity, or raise `q_max` and the test fails rather than the manuscript
  silently going false.
- Never set `post_process` in an algorithm config. Decide it where the comparison
  is defined.
- Ordinary FedPAQ never receives FedMAQ's post-processing pipeline. The separately
  coded FedPAQ-pipeline matrix is a diagnostic control and is reported as such,
  not counted as an additional baseline algorithm.
- The paired-seed methodology is load-bearing for every ablation claim; anything
  that breaks partition determinism (ADR-0006) invalidates the design, not just a
  run.
