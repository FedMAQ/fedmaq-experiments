# ADR-0025 — The FedMAQ `q_max` grid boundary is disclosed, not extended

**Status**: Accepted · 2026-09-14
**Related**: ADR-0011 (the matched-tuning stage this concerns), ADR-0023 and ADR-0024
(the all-or-nothing closure rule this decision is priced against), ADR-0021 (the
power-mean family, which inherits the same interpolation shape)

## Context

[Issue #111](https://github.com/FedMAQ/fedmaq-experiments/issues/111), a child of the
#105 Stage-A audit, records a real asymmetry: FedMAQ's matched-tuning ladder is
`q_max ∈ {4, 6, 8, 16}` and the adopted value is 16, the maximum. Every other algorithm
in the stage adopts an interior value. A stage whose winner sits on its own boundary
cannot discover that a larger value would have been better, and that is worth saying
out loud.

The issue proposes closing this by registering an extension arm at `q_max = 24` or `32`.
Its argument for why the arm is substantive — not a formality — is that raising `q_max`
"cannot raise the ceiling, but materially changes which clients reach it," because at
`q_max = 32` a client with `term` near 0.47 reaches `q_hat ≈ 17`, which snaps to 16.

**That mechanism does not hold in the condition this thesis reports.** Client capacity is
drawn `rng.uniform(2048.0, 16384.0)` (`src/fedmaq/core/strategy.py:112`), and
`np.random.Generator.uniform` excludes its upper endpoint, so `c_k < 16384`. With
`c_unit = 1024`, the Tier-1 cap `q_k_max_raw = max(1.0, np.floor(c_k / c_unit))`
(`quantization_planner.py:222`) is therefore an integer in `[2, 15]`. Under
`resource_aware: true` the tiers combine as `min(q_k_max_raw, q_hat)`
(`quantization_planner.py:256`), so the combined target never exceeds 15 whatever `q_hat`
is. `_snap_floor` (`quantization_planner.py:138-141`) maps every value in `[8, 15]` to 8,
because `DEFAULT_BIT_WIDTHS` (`quantization_planner.py:24`) jumps straight from 8 to 16.

The conclusion is deterministic rather than statistical: **under variable memory, every
client receives at most 8 bits, in every round, at every value of `q_max`.** The
`q_hat ≈ 17` the issue describes is clamped to 15 by Tier 1 and then snapped to 8. It
never becomes 16. This was confirmed by exhaustive enumeration over all fourteen
reachable Tier-1 caps crossed with a 101 × 101 grid of the two normalized Tier-2 signals,
run at the shipped `gamma1 = gamma2 = 0.5` (`conf/algorithm/fedmaq.yaml:29-30`); no
combination produced a 16-bit assignment at `q_max ∈ {16, 24, 32}`.

**The extension arm is nonetheless not a no-op, and this ADR does not claim it is.** The
same enumeration shows the arm would shift assignments for thirteen of the fourteen
reachable caps — at a Tier-1 cap of 8, a client whose normalized signals are
`(tilde_g, tilde_n) = (0.00, 0.17)` receives 3 bits under `q_max = 16`, 4 bits at 24, and
5 bits at 32. Raising `q_max` compresses the interpolation so that a
given `term` maps higher within `{2, …, 8}`. What it cannot do is reach the 16-bit
endpoint, which is the specific thing #111 argues it buys.

The decision therefore does not rest on the arm being inert. It rests on what the arm
would actually be.

## Decision

### D1 — The `q_max = 24 / 32` extension arm is declined

Three reasons, in descending weight.

**`q_max > max(Q)` is a different rule, not a larger setting of the same knob.** `q_max`
is the upper endpoint of the Tier-2 interpolation `q_hat = q_min + round((q_max −
q_min)·term)`. Setting it above 16 means interpolating toward a target the permissible set
`Q` cannot represent and the protocol does not evaluate: the Tier 2 subsection of Gradient
Quantization Bounds fixes the range at `q_min = 2`, `q_max = 16` and states plainly that
this is "a deliberate protocol boundary: this thesis does not evaluate 1-bit or 32-bit
transmission." An arm at 24 or 32 does not test a larger ceiling — it tests a
reparameterized mapping from the same signals into the same eight levels. That is a
formulation change, and formulation changes belong to the Formulation Study's registered
family (ADR-0021), not to a matched-tuning ladder whose stated job is "tuning one declared
trade-off knob per algorithm." #111's own scope excludes "changing the permissible set `Q`
itself"; interpolating past `Q`'s top member is adjacent enough to that exclusion to
require its own design decision rather than an extra ladder rung.

**The published rationale for the arm is factually wrong in the reported condition.** An
amendment justified by a mechanism that does not occur would be recorded in the protocol
under a reason the evidence contradicts. If the arm is ever wanted, it should be requested
on the correct rationale — the within-`{2,…,8}` redistribution above — and argued on its
merits against the Formulation Study, which is the stage that owns how signals map to
levels.

**The cost is a stalled Stage-A verdict, not merely GPU time.** Registering the arm moves
`baseline_tuning_wide` from `cell_count: 145` to 150 and moves its sha256 at
`conf/protocol/replacement-v1.yaml:48`. `closure_certificate` certifies a group as a unit,
and ADR-0024 records the consequence in the Stage-1a case: the moment the manifest widens,
a single incomplete new cell blocks the whole stage's verdict **including every cell that
already ran**. Those 145 cells are complete and their verdicts are already applied to
`conf/algorithm/{fedprox,feddistill,dadaquant}.yaml` at `11934ba`, the same commit
`pre-registration-stage1a` points at. Widening now would retroactively un-certify the
stage that the Gate-1 tag freezes.

**No run is invalidated by this decision, and none is re-run.** The ladder's four arms
remain genuinely distinct from one another: at `q_max = 4` the interpolation spans
`{2, 3, 4}` and Tier 1 never binds, while at `q_max = 16` it spans `[2, 16]`, is clipped
to 15, and snaps across `{2, …, 8}`. The stage compared four real configurations and
`qmax16` won on the registered `√2·σ` margin. What the boundary limits is the stage's
reach above its top rung, which is exactly what D2 discloses.

### D2 — The boundary is disclosed in the matched-tuning prose; the unreachability finding is already disclosed

#111's acceptance requires, if declined, that "FedMAQ's tuned `q_max` was selected from a
grid whose maximum it occupies" appear somewhere a reader will find it. It lands in the
Baseline Algorithms subsection of Validation and Benchmarking, in the paragraph that
introduces the ladders and already names `q_{\max} ∈ {4,6,8,16}`, immediately before the
adoption-rule paragraph. That is where a reader meets the grid, so that is where the
caveat belongs.

**The related unreachability limitation needs no new text.** The Tier 1 subsection of
Gradient Quantization Bounds already states it, in the thesis as it stands: "A
variable-memory client cannot receive 16 bits because its raw cap remains below 16 and the
final floor-snap maps raw caps from 9 through 15 to 8 bits. The separate uniform-memory
control assigns 16384 MB exactly, making its raw cap 16 and rendering Tier 1 non-binding."
The adjacent Tier 2 subsection scopes the claim correctly — "with the 16-bit endpoint
realized only by the uniform-memory control under the current capacity model" — and the
System Heterogeneity subsection of Data Collection and Environment Setup carries the
matching statement for the control arm. This ADR records that the disclosure was checked
against the manuscript rather than assumed missing; the investigation behind D1 initially
treated it as unrecorded, and it is not.

The scope of that existing disclosure is correct and must not be widened. Sixteen bits is
unreachable specifically for **variable-memory arms under `resource_aware: true`**. It is
reachable in the uniform-memory control (`floor(16384/1024) = 16` exactly) and in Ablation
Configuration 2 (`conf/algorithm/fedmaq_no_resource.yaml`, which lifts Tier 1 so `q_hat`
alone governs). A blanket claim that FedMAQ cannot emit 16 bits would be false.

## Consequences

- `conf/matrix/baseline_tuning_wide.yaml` is unmodified.
  `matrix_contracts.baseline_tuning_wide` at `conf/protocol/replacement-v1.yaml:48` keeps
  `cell_count: 145` and sha256
  `41a4456817dd410f6685feabcce0e8600e36d1489d3bdf7c80e6a208b26603c0`. No freeze artifact
  moves, so this decision does not consume the `AGENTS.md` authorization that #111's
  Authorization section anticipated — that authorization was required only for the register
  branch.
- The Stage-A verdict stands as certified. The 145 cells remain closed and the applied
  verdicts at `11934ba` remain valid.
- No GPU time is spent, and the Stage-1a seed extension currently in flight (ADR-0024) is
  untouched. This decision was reached without reading any Stage-A outcome for the FedMAQ
  arms, satisfying #111's final acceptance bullet; the argument rests on config and source
  alone. Per-round telemetry would corroborate it
  (`src/fedmaq/core/telemetry.py:230-234` records realized capacities) and is available if
  an examiner asks, but no result was consulted in reaching the decision.
- `chapter_4.tex` gains the grid-boundary caveat in the Baseline Algorithms subsection.
  No other manuscript change is required; the Tier 1 and Tier 2 subsections of Gradient
  Quantization Bounds, and the System Heterogeneity subsection of Data Collection and
  Environment Setup, already carry the capacity-model limitation.
- **If a future contributor wants the upward redistribution tested**, that is a Formulation
  Study question about how signals map into `Q`, requiring its own ADR and its own
  registered arm — not a rung appended to a closed matched-tuning ladder. The correct
  rationale is recorded above so it need not be rediscovered.
- The same interpolation shape governs the shipped power-mean branch
  (`q_hat = q_min + round((q_max − q_min)·M_p)`), so `q_max = 16` bounds Stage 1a and
  Stage 1b identically. Nothing in this decision is specific to Formulation 2.
