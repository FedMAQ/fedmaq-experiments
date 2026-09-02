# Pipeline justification audit — independent confirmation and dispatch verdict

Date: 2026-09-02. Scope: independent re-audit of the 2026-09-01
[justification audit](2026-09-01-experimental-pipeline-justification-audit.md) and
[precision/memory-path audit](2026-09-01-precision-memory-path-audit.md), plus a
fresh sweep for what they missed. The initial pass was read-only. This copy now
includes a separately labelled independent re-confirmation addendum requested by
the author; no production configuration, code, or experiment evidence was changed.
Conducted while the 102-cell `baseline_tuning_wide` stage
(issue #22 step 1) is in flight, at the author's statement that pipeline freeze is
declared and step 1 is dispatched.

## Superseded initial verdict: kill the 17 FedKD cells; let the other 85 run

> **Revised 2026-09-02, after the fresh-eyes sweep completed.** The original
> verdict of this document was "do not stop the runs," on the finding that the
> step-1 gate scores accuracy alone and no defect reached `src/`. That reasoning
> holds for **85 of the 102 cells** and is retained below. It does not hold for
> FedKD's 17: finding **N10** is a genuine `src/` defect inside the running stage.
> The corrected verdict is scoped, not reversed.

> **Superseded by N18 below.** The independent re-confirmation found a second,
> more serious FedKD lifecycle defect. No FedKD accuracy, final checkpoint, or
> bidirectional-byte result from the affected pipeline is admissible.

**N10 blocks the FedKD arm only.** `client_hooks/fedkd.py:66-70` constructs its
`torch.optim.SGD` with no `momentum=` argument — PyTorch then defaults it to 0.
Verified as the **sole** exception across all nine SGD constructions in `src/`
(`standard.py:98`, `feddistill.py:124`, `cfd.py:85`, `fedmd.py:63,128`,
`kd_utils.py:93`, `strategy_hooks/cfd.py:229` all pass it). `exp_config` is already
in scope at `fedkd.py:63`, so the repair is the two-line `standard.py:91` pattern.

This is not merely a defect but an affirmative misstatement in the evidence chain:
[ADR-0005](../adr/0005-baseline-stack-membership.md):108-115 attributes FedKD's
"remaining 15–27pp gap against every other baseline" to SVD lossiness and instructs
the reader that it is "architectural, not a tuning failure … do not read it as a
bug." That attribution is confounded by an optimizer defect. FedKD's absolute level
is not reportable until this is fixed; its **`tmax` ranking may survive**, since all
five variants share `momentum=0` and the adoption rule compares them only to each
other.

Every GPU-hour spent on FedKD after this point produces a row that must be
withdrawn or re-run. **Kill those 17 cells; the other 85 are unaffected**, because
`fedkd.py` is reached only by the FedKD arm.

### The retained verdict, for the other 85 cells

**No other finding in this audit invalidates a single cell of the stage now
running.** Every other defect found is one of three kinds:

1. **Prose or claim-scope** — the implementation is self-consistent; the wording
   describing it is not.
2. **Post-hoc interpretation rules** — `scripts/analysis.py` consumes the step-1
   CSVs *after* they are written. Those rules can be revised without re-running a
   cell, because the cells log the inputs a revised rule would need.
3. **Second-freeze repairs** — bound to issue #22 step 5, structurally unable to
   reach step 1 or Stage 1a.

Nothing *else* requires a change to `src/`, and that is the load-bearing fact: a
cost-model or quantizer edit would break the freeze certificate
(`scripts/check_freeze.py:34` hashes `src/**/*.py`) and trip the golden-diff gate
(`scripts/golden_diff.py:52` ignores only wall-clock columns, so every simulated
value is compared). No finding demands one.

**But a window is closing, and it is not the dispatch window.** Three decisions
below are pre-registration-integrity decisions: the evidence survives either way,
but deciding them *after* reading the step-1 results converts a pre-specified rule
into a post-hoc one. That window closes when you look at the tuning curves, not
when the GPUs stop. Settle them while the cells burn.

## What the prior audits got right

All four headline findings confirmed against source. Two are sharper than stated,
one is materially incomplete, one is defused.

### 1. Precision is not wire packing — CONFIRMED, and disclosed

`src/fedmaq/baselines/quantization.py:67` is the whole serialization:
`codes.astype(np.int64).tobytes() + np.float32(scale).tobytes()`. Raw storage is
`8N + 4` bytes per non-empty tensor, independent of nominal `q`. Every arm routes
here; the only genuinely bit-packed coder in the repo
(`dadaquant_coder.py:127-145`, Elias omega) feeds `secondary_bytes` only, never
`measured_bytes`, exactly as [ADR-0020](../adr/0020-secondary-byte-axis.md) says.

**The contamination check comes back clean.** This was the highest-stakes
question: whether any *decision rule* derives bytes from nominal `q`. It does not.
`iso_byte_scores` (`scripts/analysis.py:1158+`) budgets on
`communication/cumulative_mb`, written by `telemetry.py:346-353` from
`UploadReport.measured_bytes`. The planner does no byte arithmetic at all. The
only nominal-bit expression in `src/` is a teacher-confidence weight at
`kd_utils.py:119-122`, dead under the frozen `soft_voting: false`. No CSV column
is a nominal-bit derivation. **Path 1 of the precision audit's recommendation is
already what the code does** — only the prose needs to catch up.

### 2. Hardware labels need temporal correction — CONFIRMED, self-inconsistent

`docs/adr/0002-hardware-telemetry-grounding.md` heads its client table "Edge
Client Fleet (Oct 2023)" and "Raspberry Pi 5 Series (Released Oct 2023)", then
lists 2 GB and 16 GB tiers. The 2023 Pi 5 shipped 4 GB and 8 GB only; 2 GB arrived
August 2024, 16 GB January 2025. The ADR fails its own stated requirement 1,
"Temporal Alignment" (`:20`).

**A second defect in the same table, not previously noted:** `:22` claims the
capacity distribution "align[s] cleanly with real physical single-board computer
configurations", but `strategy.py:112` draws `rng.uniform(2048.0, 16384.0)` —
*continuously*. Raw Tier-1 caps of 3, 5, 6, and 7 bits arise routinely and
correspond to no SKU. The discrete-tier table describes a fleet the code never
samples.

### 3. Memory does not create stragglers — CONFIRMED, but the audit's phrasing is refutable

`client_round_delay` (`strategy.py:126-149`) takes `cid, model_size_bytes,
bytes_uploaded, train_sample_count, compute_scale`. No memory term. Bandwidth and
compute are `np.full` scalars (`strategy.py:94-96`);
`tests/test_timing_golden.py:14-38` constructs the cost model *without a memory
argument at all* and passes. The simulated straggler — `max(round_delays)` at
`telemetry.py:293` — is the client with the largest Dirichlet shard, a data-volume
straggler.

**An indirect path exists and runs opposite to the prose.** memory → Tier-1 cap →
`q` → code alphabet → zlib compressibility → `measured_bytes` → `standard.py:151`
→ `telemetry.py:248` → `t_upload` (`strategy.py:146`). It is FedMAQ-only;
baselines never receive a capacity-derived `q`. Its sign: lower memory → lower `q`
→ **fewer** bytes → marginally *faster*. A defense phrased as "memory has no
effect on time" is false as stated and refutable by reading one line. The
defensible claim is: *capacity governs precision; its only effect on simulated
time is a second-order payload effect that makes constrained clients slightly
faster.*

### 4. The active formulation is not the historical configuration — CONFIRMED, and DEFUSED for step 1

`conf/algorithm/fedmaq.yaml:26` still reads `formulation: 2`. Issue #45's claim is
confirmed empirically: `FORMULATION_CONSTANTS["power_mean"] = ("p","omega")`
(`quantization_planner.py:39`) plus the silent `.get(key, default)` fallback
(`:87-90`) means `gamma1`/`gamma2` are **read but never used** under the power
mean. At materialization, `fedmaq_no_data.yaml:25` and `fedmaq_no_state.yaml:31`
become null overrides and Configurations 3 and 4 collapse into Configuration 7 —
12 of the 147 downstream cells silently reporting the full method as
component-contribution arms. The arms are **live and correct today**; the collapse
is a future event, gated to step 5 exactly as issue #22 states.

**The step-1 exposure is defused.** A concern was raised that FedMAQ's `q_max` is
being tuned under a retired formulation, since `q_hat = q_min + round((q_max −
q_min) · term)` (`quantization_planner.py:229`) makes `q_max` a direct multiplier
on the formulation's output — so the optimum is not formulation-invariant a
priori. But F2 at `γ1 = γ2 = 0.5` is `g̃^0.5 · ñ^0.5`, and the power mean at
`p = 0, ω = 0.5` is `g̃^ω · ñ^(1−ω)` (`quantization_planner.py:178`). These are the
same function. Verified over 320 combinations of `(g̃, ñ, q_max)` spanning the full
tuning ladder: **320 checked, 0 mismatches**, on both `q_hat` and the snapped `q`.

So `q_max` is being tuned at the exact centre of the Stage-1a power-mean ladder,
not under an orphaned formulation. The residual assumption — that the `q_max`
optimum at `p = 0, ω = 0.5` transfers to whatever `p` Stage 1a selects — is a real
one and is undocumented, but it is a narrow, statable assumption rather than a
methodological break. **State it; do not re-run for it.**

## New findings the prior audits missed

### N1. The headline communication metric is bidirectional, and the download leg imposes a floor

`telemetry.py:286` accumulates `round_bytes_downloaded += model_size_bytes`
*inside the per-client loop* — one full broadcast per sampled client — and `:298`
sums both legs into `round_bytes`, which becomes `cumulative_mb`, which is the
iso-byte budget and the headline axis.

Every sampled client pays a model broadcast in addition to its upload. In the
ordinary shared-hook path that broadcast is serialized through the same zlib seam
(`strategy_hooks/base.py:127-143`). **Consequence: even at zero uplink cost, the
bidirectional ratio cannot fall below the download share of the chosen baseline.**
If the baseline spends `D` bytes downloading and `U` uploading, the exact lower
bound is `D/(D+U)` and the exact maximum reduction factor is `1 + U/D`. It is only
`2.0x` when `D = U`; there is no universal numeric ceiling.

**Quantified on real data (`outputs/golden/step2/`, final round, `client_gpus=1.0`,
`num_clients=2`, `client_fraction=0.1` → exactly 1 sampled client/round).** Solving
`round_bytes = D + n·upload` across three arms gives `n = 1` and a shared download
constant `D = 8,946,728` B — exactly FedAvg's raw `payload_bytes`, i.e. the
uncompressed model:

| arm | `round_bytes` | upload | download `D` | headline vs FedAvg | upload vs FedAvg |
|---|---|---|---|---|---|
| fedavg | 16,854,347 | 7,907,619 | 8,946,728 | 1.00× | 1.00× |
| fedpaq (q=8) | 11,471,880 | 2,525,152 | 8,946,728 | **0.68×** | 0.32× |
| dadaquant | 9,246,599 | 299,871 | 8,946,728 | **0.55×** | **0.038×** |
| fedmaq | 14,566,747 | 5,620,019 | 8,946,728 | 0.86× | 0.71× |

For this historical capture only, the floor is `8,946,728 / 16,854,347` =
**53.1%**, corresponding to a maximum **1.88x** reduction. DAdaQuant already sits
at 1.034x that floor: a **26x uplink reduction buys a 1.82x headline reduction.** The
dynamic range separating quantizers on `cumulative_mb` is compressed roughly
two-fold by a constant every arm pays identically — which is also the axis
[ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md)'s
iso-byte rule selects on. These numbers are diagnostic, not a current certificate:
the capture predates the candidate fixes and was made from a dirty historical tree.

This is the most consequential new finding. It does not invalidate anything; it
bounds what the results can show. A pre-registered expectation written without it
will read as a failure against an unreachable target.

**Correction to the former N1a:** current FedDistill and the base hook both use
`UploadReport.from_payloads`, so the historical arithmetic is not evidence of a
current FedDistill raw-download asymmetry. Withdraw that claim. FedKD does have a
special SVD broadcast path, but N18 shows that its current download telemetry is
coupled to the wrong lifecycle state and must be repaired before comparing it.

### N2. The step-1 adoption rule scores a communication knob on accuracy alone

`baseline_tuning_margin` (`scripts/analysis.py:591`) adopts on
`accuracy_at_round(df, 100)` (`:649`) with `margin = sqrt(2) * sigma` and
`adopted = max(clearing, key=delta)` (`:741-757`). **No byte term anywhere in the
path.** Meanwhile [ADR-0011](../adr/0011-baseline-matched-tuning.md) declares the
tuned knob to be the one "governing each baseline's own accuracy–communication
trade-off, because that is the axis every claim rests on."

Three of the six ladders now running are precision/communication knobs:
`fedpaq.q` = {2,4,8,16,32}, `fedmaq.q_max` = {4,6,8,16,32}, `dadaquant.phi` =
{1,5,10,20,40} (`conf/matrix/baseline_tuning_wide.yaml:27-54`). Because accuracy
rises with `q`, the rule hands each quantized baseline its most byte-hungry
setting — and the downstream headline comparison is *iso-byte*, which then
penalises exactly that setting. The procedure gives baselines their worst
configuration under the metric they are reported on.

The rule is procedurally symmetric (FedMAQ's own `q_max` is adopted the same way),
so this is a validity problem, not favouritism by construction.
`accuracy_at_budget` already exists at `scripts/analysis.py:912` and reads
`communication/cumulative_mb` — the repair is available in-repo and is purely
post-hoc.

### N3. The `q32` / `qmax32` rungs cost more than sending nothing compressed

At `q = 32` the int64 codes measure *above* the uncompressed float32 control —
int64 storage plus a code alphabet too large for zlib to exploit. Both ladders now
running reach it: `baseline_tuning_wide.yaml:32` (`q32`) and `:193-197`
(`algorithm.q_max=32`). Separately, `qmax32` exceeds ADR-0002's stated intentional
Tier-2 ceiling of 16, under which "no client transmits at FP32" (`:155-161`).

**Basis — structural, then measured.** `symmetric_levels(q) = 2^(q-1) − 1`
(`quantization.py:37`), so `q=32` gives **2,147,483,647** positive levels: codes
occupy ~31 bits of each int64, leaving only the four high-order zero bytes for zlib
to reclaim. The floor is therefore ~4 B/element against an fp32 control that
*measures* 3.71 B/element. The crossover is a property of the code width, not of
the delta distribution.

Confirmed by executing the real hook over the real architecture. The CIFAR-10 model
factory yields 158 tensors / 2,236,682 elements / 8,946,728 raw bytes — matching the
golden `payload_bytes` **exactly**, so the shape list is the one actually shipped.
Running `FedPAQCompressionHook` + `measure_bytes` across a range of delta tail
weights (Student-t, df 1.5 → 30):

| df | fp32 control | q=8 | q=16 | q=32 | **q32 / fp32** |
|---|---|---|---|---|---|
| 1.5 | 8,387,673 | 346,322 | 2,903,153 | 8,343,792 | 0.995 |
| 3.0 | 8,345,281 | 767,424 | 3,913,615 | 9,055,374 | 1.085 |
| 30.0 | 8,305,304 | 841,385 | 4,015,540 | 9,112,051 | 1.097 |

The fp32 control lands within **+4.9%** of golden's measured 7,907,619 B, so the
control leg is faithful. The q=8 legs run **66–86% *below*** golden's 2,525,152 B —
synthetic deltas compress better than real SGD updates at every tail weight tried.
That bias is one-directional and in the conservative direction: **the true q=32
figure is higher than the table shows, and the table already exceeds fp32.**

`_serialize_codes`' own docstring (`quantization.py:46-63`) documents the fixed
int64 width as a deliberate deferral under **#25**, predicts this exact
high-bit-width outcome, and states it "must be checked per #25 AC 3, not assumed
from this docstring or from synthetic test data." So this is a *tracked* deferral
with an open acceptance criterion, not a hidden defect — the finding here is that
AC 3 is now answered, and answered in the direction the docstring flagged as
possible. FedMAQ inherits it: `baselines/__init__.py:66-68` dispatches `fedmaq` to
`FedPAQCompressionHook` (and `postprocess.py:68` uses the same `symmetric_levels`),
so the `qmax32` rung hits the identical crossover whenever the planner emits 32.

Adopting a compression baseline at a setting more expensive than no compression
would poison the downstream grid. Under N2's accuracy-only rule that outcome is
reachable, because `q32` will plausibly score highest on accuracy.

### N4. `baseline_tuning_margin` has no round-completeness guard

`accuracy_at_round` (`scripts/analysis.py:889-895`) silently falls back to the
last logged round when round 100 is absent. Cell *counts* are guarded — n=5
reference, n=3 challengers, else "No verdict is issued" (`:706-724`) — but a cell
with all five runs present, one truncated at round 40, averages that run's
round-40 accuracy into the mean with no error.

The repo already has the guard (`has_expected_round` / `incomplete_runs`,
`:1027-1037`) and applies it to the iso-byte rule (`:2165`) and the study runs
(`:2307`). The tuning CLI path (`:2422`) does not call it. This is an
inconsistency in the repo's own established pattern, not a missing idea — and it
matters now, because a partially recovered sweep is exactly the condition it fails
silently in.

### N5. The ω-polarity of the ablation repair is inverted from the intuitive reading

`quantization_planner.py:169-171`: `if omega == 0.0: return tilde_n` /
`if omega == 1.0: return tilde_g`. ω weights `tilde_g`, the **state/gradient**
signal. So the correct binding is `fedmaq_no_data → omega: 1.0` and
`fedmaq_no_state → omega: 0.0` — the inverse of what the names suggest.
[ADR-0021](../adr/0021-power-mean-formulation-family.md) D4 (`:51`) says only "the
ω=1 / ω=0 endpoints" without disambiguating. A swap yields two arms that *look*
live and are silently mislabeled — strictly worse than the inert failure #45
describes.

**And the existing guard cannot catch either failure.**
`tests/test_config_and_dispatch.py:78-103` asserts key-set parity and value diffs;
it has no knowledge of `FORMULATION_CONSTANTS`, so it passes on two inert arms if
the gammas are left in place, and passes again on a zero-removal Config-7
duplicate. The repair needs a formulation-aware assertion, not just corrected
YAML. (Note: issue #45 cites `tests/test_simulation.py` for `ABLATION_ARM_DIFFS`;
the guard actually lives in `test_config_and_dispatch.py:67-103`.)

**Second-order:** at either ω endpoint the short-circuit precedes all `p` handling
(`:169-181`), so Configurations 3 and 4 never exercise the selected degree `p`.
Defensible by design, but recorded nowhere.

### N6. Tier-1 can bind on at most 43% of clients

Tier-1 never binds when `floor(c_k / 512) >= q_max = 16`, i.e. when
`c_k >= 8192` MB. Under `c_k ~ U(2048, 16384)`,
`P(c_k < 8192) = 6144 / 14336 = 42.9%`. So the capacity rule — the study's central
mechanism — is inert for a majority of clients by construction, and ADR-0002
already concedes the endpoint case: "8 GB and 16 GB Pi 5 clients are **functionally
identical** in achievable precision" (`:161`).

Not a defect. But "how often does your mechanism actually fire?" is an obvious
panel question with a specific numeric answer, and that answer should be in your
pocket rather than derived at the podium.

### N7. `q=1` is ternary and cannot be represented by a truthful one-bit wire code

`symmetric_levels(1) == symmetric_levels(2) == 1` (`quantization.py:37-39`), but
the two branches are **not the same quantizer**: `q=1` is deterministic sign/zero,
while `q=2` uses the stochastic L2 branch. The `q=1` code alphabet is
`{-1, 0, +1}`. Preserving all three symbols requires more than one bit per
coordinate unless an additional zero mask or equivalent side channel is charged.
Therefore the planned real-packing repair creates an author decision: remove the
one-bit rung, redefine it as binary and change its numerics, or retain the ternary
scheme under an honestly costed representation.

### N8. Diff-coding subtracts codes across a `q` change

`postprocess.py:138-139` guards on `prev_codes.shape == codes.shape` only. When
FedMAQ's adaptive `q` moves between rounds, codes from different alphabets are
subtracted. Lossless and exactly invertible — correctness is fine — but it
inflates measured bytes precisely when adaptation fires, charging FedMAQ's own
mechanism. Report it; do not repair it.

### N9. ADR-0002 carries three further statements the code does not support

- `:12` gives round time as `t_download + t_train + t_upload`, omitting the server
  term that `telemetry.py:295-297` adds (FedMAQ's KD cost is real,
  `strategy_hooks/fedmaq.py:198-211`).
- `:53` justifies 10 Mbps by "multi-client channel contention". No contention is
  modelled; `strategy.py:94-95` sets identical constant symmetric links and delays
  are computed per-client independently. The *value* may stand; the cited
  *mechanism* is not implemented.
- `:173`'s "7.2 s download + 7.2 s upload" symmetry holds only for an
  identity-transport arm; for every compressed arm both legs are zlib-measured and
  unequal.

Also noted: the dead `heterogeneity.bandwidth` / `heterogeneity.compute` branches
at `strategy.py:85-92` match no key anywhere in `conf/`, and their `min_*` naming
is vestigial of an abandoned range design — the most likely reason a reader would
believe heterogeneous bandwidth is configurable.

## Claim census: 65 sites, 33 must-fix

A full census of `fedmaq-manuscript`, `fedmaq-journal-article`,
`fedmaq-presentations`, `oce-prep` and `fedmaq-experiments/{conf,docs}` returned:

| Family | Sites | Must fix | Should rephrase |
| --- | --- | --- | --- |
| A — nominal-bit / ratio communication | 14 | 4 | 6 |
| B — Late-2023 Pi 5 fleet | 5 | 2 | 3 |
| C — memory causes latency | 8 | 4 | 3 |
| D — F1–F4 / `formulation: 2` as selected | 19 | 15 | 4 |
| E — config comments asserting results | 9 | 5 | 3 |
| Other overclaims | 10 | 3 | 4 |
| **Total** | **65** | **33** | **23** |

Four things about that table matter more than its size.

**The thesis contradicts itself in print.** `chapter_4.tex:349` establishes an 8:1
compute-to-communication ratio and concludes byte reduction "shortens the round
only marginally". `chapter_5.tex:101` instructs the author to report whether
precision scaling "reduces the straggler effect and total estimated wall-clock
training time". Both cannot stand. This is the one census item where the
manuscript is not merely overstating but self-refuting, and it is the most likely
thing a panelist reading two chapters in sequence will find. `chapter_4.tex:349`
is already the correct position — the repair is a deletion, not new analysis.

**`chapter_5.tex:98-101` is a stated research objective, not a stray sentence.**
It commits the study to a question the instrument cannot answer, since the only
memory→time channel is the inverted second-order one in finding 3. This is the
single highest-priority prose item in the audit.

**Family D is bimodal, which is good news.** Chapters 3 and 4 are already fully
recut to ADR-0021's power-mean family. Chapters 5–6 and the *entire* journal
article are not — and in the journal, Formulation 2 appears as a numbered
*contribution* (`01_introduction.tex:42-44`), not just a reporting instruction.
The work is concentrated in about six files, not diffuse across the corpus.

**Your defense-prep material carries the false hardware claim verbatim.**
`oce-prep/GLOSSARY.md:84-86` states the 2/4/8/16 GB Pi 5 tier mapping as fact.
That is material you would quote under questioning, so it is higher-risk than the
manuscript instance. `oce-prep/lessons/0003-...html:54` carries a disclaimer that
defends the wrong thing — it defends `c_unit`, not the nonexistent tier set.

Two incidental census results worth knowing: the **defense deck does not yet
exist** (`fedmaq-presentations/` is an unmodified beamer template with generic
FedAvg/FedProx filler and zero FedMAQ claims), and three cell-count contradictions
survive in `chapter_5.tex` — a "105-run grid" at `:47` and a "177 reported runs"
denominator at `:226`/`:239`, against the 147/243 the same chapter opens with.

**Family E deserves separate emphasis.** `conf/algorithm/fedmaq.yaml:19-26` is a
frozen config comment asserting an empirical verdict from a superseded campaign
("Formulation 2 wins the severe skew by 2.1pp"), and closing with a standing
instruction: "Do not revert this to 3 to match an older doc; the docs are what is
stale." Under ADR-0021 that instruction is now inverted — the config is the stale
party. The same file already demonstrates the correct repair pattern at `:42-46`,
where a retired argument is named, marked as no longer holding, and kept for
history. Apply that pattern to `:19-26`.

## Gate ledger

| Finding | Gate | Why |
| --- | --- | --- |
| **N18 FedKD evaluates/checkpoints the pre-fit broadcast and misaccounts its download** | **BLOCKS ALL FedKD EVIDENCE** | Accuracy, final model, and bidirectional bytes are coupled to the wrong lifecycle state. Repair and test the two-round production seam. |
| **N19 packed-wire protocol is underspecified** | **BLOCKS ALL REPLACEMENT COMMUNICATION EVIDENCE** | A serializer substitution cannot define signed codes, metadata, post-process widths, or `q` transitions. Specify and round-trip-test the wire format first. |
| **N20 partial checkpoint can count as complete** | **BLOCKS RECOVERY/DISPATCH** | Atomic publication and structural completion validation are required before `--skip_completed` is safe. |
| **N21 analysis can continue after closure failure** | **BLOCKS EVERY VERDICT** | Missing/duplicate/truncated/unscorable inputs must exit nonzero without winner artifacts. |
| **N22 memory probe crosses the stated trust boundary** | **BLOCKS FedMAQ PROTOCOL FREEZE** | Model it as charged client-local preflight emulation or implement an actual preflight protocol. |
| **N24 transition diff confused with golden PASS** | **BLOCKS SECOND FREEZE** | Preserve the transition diagnostic separately; only two independent clean candidate captures can pass. |
| **N25 FedPAQ `post_process` flag is a silent no-op** | **BLOCKS THE NEW CONTROL ARM** | Register and test a distinct algorithm identity; YAML alone would reproduce ordinary FedPAQ. |
| **N10 FedKD trains with `momentum=0`** | **BLOCKS ALL FedKD EVIDENCE** | A real optimizer inconsistency, now compounded by N18. Every FedKD row must be withdrawn and rerun after both repairs. |
| N13 iso-byte budget pooled across seeds, vs ADR-0016 | **Decide before reading step-1 results** | A pre-registered protocol deviation running in the proponent's direction. Code or ADR must move — choosing after seeing the numbers is what a panel will call it. |
| N14 FedKD continuous inverse-loss weighting already implemented; config text is false | **Before FedKD grid rows are cut** | Preserve and regression-test the implementation; remove the nonexistent binary correctness-gate description. |
| N11 frozen per-round batch ordering | **Before second freeze** (disclosure) | Arm-neutral, biases nothing. But by the repo's own #24 standard this is the same bug unfixed one line away — if the author holds that standard binding, it escalates. |
| N15 FedMAQ-only post-processing pipeline | **Before second freeze** (claim scope) | The byte win is not decomposable into mechanism vs. coding. One pipeline-equipped FedPAQ arm closes it permanently. |
| N17 no validation split; selection on test accuracy | **BLOCKS REPLACEMENT CAMPAIGN** | Introduce a selection-only validation split and reserve test data for final reporting; invalidate and rederive every test-selected decision. |
| N12 additive seed composition | Prose only | No material effect established. |
| `chapter_5.tex:98-101` straggler/wall-clock research objective | **Decide before reading step-1 results** | Commits the study to a question the instrument cannot answer. The alternative repair — making it true by editing the cost model — breaks the freeze certificate and the golden gate, so foreclosing it is the decision that must not drift. Minutes of prose work. |
| N2 adoption rule scores accuracy only | **Decide before reading step-1 results** | The 102 runs are rule-agnostic raw evidence and can be re-scored without re-running a cell; but choosing the rule after seeing the curves makes it post-hoc against a pre-registered margin. |
| N3 `q32` / `qmax32` rungs exceed the uncompressed control | **Decide before reading step-1 results** | Same decision as N2. Data valid either way; the *rule* for handling a rung that costs more than no compression must be written down first. |
| N4 no round-completeness guard in the tuning verdict | **Before the step-1 verdict is computed** | Post-hoc analysis fix. Bites only if a cell is truncated — which a partially recovered sweep produces. |
| Precision is not wire packing | Prose only | Implementation self-consistent and disclosed; no decision rule contaminated. |
| Pi 5 temporal label + continuous-vs-discrete sampling | Prose only (before second freeze) | Changes no number. ADR-0002 is the grounding document; correct it, not the config comments it protects. |
| Memory-not-a-straggler + the inverted indirect path | Prose only | Code is correct. The absolute phrasing is what must not be written. |
| N1 bidirectional metric / download-leg floor | **Before second freeze** | Bounds what results can show; must scope effect-size language before the 147-cell grid is frozen. |
| Former N1a FedDistill asymmetry | **WITHDRAWN** | Current code uses the shared compressed payload-report seam; the old capture cannot establish a current asymmetry. |
| #45 gamma keys go inert at materialization | **Before second freeze** (step 5) | Confirmed; structurally cannot reach step 1 or Stage 1a. Arms are live today. |
| N5 ω-polarity + guard cannot detect the collapse | **Before second freeze** (step 5) | A polarity swap is worse than the inert failure; the test needs a formulation-aware assertion. |
| Superseded `formulation_study` auditor group | **Before second freeze** (step 5) | `scripts/audit_study1_artifacts.py:29-34` hard-raises at `:716-718`. |
| N6 Tier-1 binds on <=43% of clients | Prose only | Not a defect. Prepare the number. |
| N7 ternary `q=1`, N8 diff-coding across `q` | **BLOCKS PACKED-WIRE SPEC** | Define a versioned, round-trippable code format and settle the one-bit alphabet and cross-`q` state transition before implementation. |
| N9 ADR-0002 formula, contention, symmetry defects | **Before second freeze** | Changes no number; the ADR cannot be frozen as evidence while it misdescribes the code. |
| 65 claim-census sites | Prose, staged | 33 must-fix. Concentrated in `chapter_5.tex`, `chapter_6.tex`, four journal files, `oce-prep/GLOSSARY.md`, and five config files. |

## Panel posture, revised

The prior audit's posture stands and gains two sentences:

> We fixed a controlled, reproducible comparison environment and separated the
> mechanism-level question from deployment certification. We evaluate whether a
> capacity-derived precision rule changes the measured accuracy–communication
> trade-off under stated non-IID conditions. **Our communication axis is the
> bidirectional measured serialized byte count, so the achievable reduction is
> bounded below by an uncompressed download leg we do not claim to compress.
> Capacity governs precision only; its sole effect on simulated time is a
> second-order payload effect in the direction opposite to a straggler.** We do
> not claim the simulation certifies end-device memory feasibility, real-world
> latency, or universal superiority.

## Findings from the fresh-eyes sweep (N10–N17)

Added 2026-09-02 after the adversarial sweep completed. N10, N11 and N13 were
re-verified against source directly; the rest are recorded as reported.

### N10. FedKD trains without SGD momentum — the only arm that does

See the revised verdict above. `client_hooks/fedkd.py:66-70`. **Blocks the FedKD
arm of the in-flight stage.** Contradicts both the prior audit's "shared optimizer
settings" claim and ADR-0005's architectural attribution of the 15–27pp gap.

### N11. Local training replays the same batch ordering every round

`simulation.py:254` computes `client_seed = plan.seed + partition_id` — **no round
term** — and feeds it to both `configure_torch_determinism` (`:255`) and
`get_client_loader(..., seed=client_seed)` (`:262`). Flower rebuilds `client_fn`
per message, so every round replays one permutation.

The repo diagnosed this exact bug and fixed it *one line away*: `standard.py:79`
reseeds the compressor as `np.random.default_rng((seed, partition_id, server_round))`
under #24, with a comment naming the training RNG as a separate stream. The fix
was never applied to the training path.

Blast radius is bounded and was checked: `models.py` has no dropout and
`partitioning.py:51-62` applies no augmentation (`ToTensor + Normalize` only), so
the batch permutation is the *only* per-round stochastic input to local training —
and it is frozen for all 100 rounds. **Arm-neutral**, so it biases no comparison;
it is a disclosure defect. By the repo's own #24 standard this is the same bug left
unfixed, and if the author holds that standard binding it becomes a dispatch
blocker. It is not classified as one here because it changes no arm's standing
relative to another.

### N12. Additive seed composition aliases RNG streams across runs

`simulation.py:254, 274` sum `plan.seed + partition_id`; `standard.py:79`
deliberately uses tuple composition instead, for stated reasons. With seeds
{0,7,21,42,123} the client-seed ranges overlap in 58 values. No material effect
established — partitions differ per seed, so aliased clients hold different shards.
Recorded as the third instance of a pattern the repo has twice called a bug.

### N13. The iso-byte budget is pooled across seeds, against ADR-0016

`analysis.py:1195`: `budget_mb = min(finals.values())`, where `finals` spans both
arms **and all common seeds** (`:1192-1194`).
[ADR-0016](../adr/0016-v2-evaluation-protocol-and-advance-rule.md)
specifies a per-seed `B*_s`. The pairing logic at `:1834-1855` is correct; the
budget inside it is not.

Direction is one-way: pooled-min ≤ per-seed-min for every seed, so each pair is
scored at or before its registered budget — earlier on the accuracy/byte curve,
where the more byte-efficient arm has completed more rounds. That is usually
FedMAQ. **Magnitude is unquantified** (needs run artifacts). A deviation from a
pre-registered protocol that runs in the proponent's direction is the worst class
of deviation regardless of size. Either the code or the ADR must move — and must
move *before* the numbers are seen.

### N14. FedKD's continuous inverse-loss weighting is implemented; its config text is wrong

`conf/algorithm/fedkd.yaml` states that mutual distillation "is gated by prediction
correctness." That binary gate exists neither in the implementation nor in the
accepted source-method description. `client_hooks/fedkd.py:95-112` already applies
the continuous loss-inverse weighting in both directions by dividing each KL term
by `loss_s_task + loss_t_task + 1e-6`. Preserve that behavior, encode it in a
regression test, and correct the config/manuscript/knowledge-graph prose. Do not add
a binary gate.

### N15. FedMAQ alone carries error feedback, diff coding, and zlib on the primary grid

`baselines/__init__.py:107-110` returns `FedMAQPostProcessCompressionHook` only for
`fedmaq` with `post_process=true`, which every `benchmark_grid*.yaml` sets; every
baseline ships `post_process: false` and gets an identity hook. None of the three
mechanisms is adaptive or uses the capacity signal, and FedMAQ's base quantizer
*is* FedPAQ's (`baselines/__init__.py:66`).

The regime-matching is applied knowingly and correctly throughout
(`conf/algorithm/fedmaq.yaml:57-68`, `conf/matrix/uniform_memory_control.yaml:30-37`),
so this is claim-scope, not unfairness. But the grid's byte advantage is not
decomposable into "adaptive quantization" versus "coding the baseline was denied."
**One extra arm — a pipeline-equipped FedPAQ in the grid regime — closes this
permanently**, and it is the most likely place for the headline number to be
attacked.

### N16. Verified clean, recorded so they are not re-litigated

- **Proxy set** — see Coverage below. Probe-verified end to end.
- **Partitions and cohorts are arm-invariant.** `generate_partition_indices`
  (`partitioning.py:284`) takes no algorithm argument; `client_manager.py:179-181`
  samples as a pure function of (seed, round). Paired-arm comparability holds.
- **KD student and teachers are honest.** Student is Flower's data-size-weighted
  aggregate (`kd_utils.py:187-189`); teachers are the genuinely *dequantized*
  reconstructions (`training_skeleton.py:126-142`), not pre-quantization originals.
- **Cell counts reconcile:** 102 (6 × 17) + 84 (14 × 3 × 2) + 12 (2 ω × 3 × 2) +
  147 (42+42+21+36+6) = **345**. The 12 is derivable only from issue #22 step 4,
  not from `conf/matrix/` — state that provenance if asked.
- **FedProx, FedPAQ, DAdaQuant, FedDistill** faithful within declared scope.
- `_partition.py:65`'s unseeded `hash()` fallback is **unreachable** — `cid` is
  always `str(partition_id)` (`simulation.py:280`). Dead code, not a live hazard.
- The partition cache lacks a code-version key (`partitioning.py:310-315`,
  gitignored) but is **not currently stale**: cache mtime predates the last
  partition-*logic* commit's successor, and the intervening commit is
  docstring-only. Clear it before the freeze anyway.

### N17. No validation split anywhere — every selection is on test accuracy

`simulation.py:279` passes `testloader=train_loader` and `fraction_evaluate=0.0`;
server evaluation uses the torchvision test split. No third split exists. The
accuracy floor (`analysis.py:884-886`), formulation selection (`:1236-1240`),
`select_winner`'s tie-break (`:1409+`) and knob adoption (`:648`) all read test
accuracy.

Real mitigations, and they should be said out loud: knob tuning runs at α=0.3, a
skew the reported grid never uses; formulation selection is scoped to CIFAR-10.
**The mitigation that does not hold:** Stage 1a runs at α ∈ {0.1, 1.0} with seeds
{0,42,123} — the same skews and seeds the headline grid reports. So `p` is selected
on test accuracy in the exact cells whose test accuracy is then reported.

For the historical campaign this was initially classified as disclosure-only. The
author has instead accepted a validation split. It therefore **blocks the
replacement campaign** and invalidates the old test-selected decisions; Stage 1a,
tuning, ablation, and final reporting must be rerun/rederived under a versioned
selection-validation versus reserved-test contract.

## Coverage and limits

Confirmed by four independent sub-audits plus direct verification: the byte and
precision path, the memory/time cost model, the formulation recut and ablation
binding, and a 65-site claim census across five repos. The F2 ≡ power-mean(p=0,
ω=0.5) equivalence was verified by execution, not inspection.

**Proxy-set integrity: checked directly, and clean.** This was the one outstanding
item that could have inverted the verdict, so it was verified rather than deferred.
Two disjointness properties hold by construction:

1. **Proxy ∩ test = ∅.** `get_server_loaders` (`partitioning.py:420-432`) builds
   `public_subset` from `_load_dataset_cached(dataset_name, True)` — the *train*
   split — and `test_loader` from `_load_dataset_cached(dataset_name, False)` — the
   *test* split. Separate corpora; no overlap is possible.
2. **Proxy ∩ client-train = ∅.** The public pool is carved out *before*
   partitioning, and removed from the pool it was drawn from:
   `class_indices[c] = np.setdiff1d(class_indices[c], selected)`
   (`partitioning.py:355`), with the Dirichlet and writer partitions both running
   on the reduced `class_indices` afterward (`:358-370`).

The pool is also class-stratified by construction (`base_per_class` +
remainder-distribution, `:332-343`), so the KD arms are not distilling on a
label-skewed proxy. **No distillation-set leakage. The verdict stands.**

**The outstanding sweep has since completed** and its findings are folded in as
N10–N17: paired seeds and partitions (clean, N16), baseline mechanism fidelity
(N10 plus the corrected N14), evaluation protocol (N13, N17), determinism (N11,
N12), and the 345-cell arithmetic (reconciles, N16). The later independent
re-confirmation found the additional blockers N18–N25 recorded below; its revised
dispatch verdict supersedes this historical sweep conclusion.

**Still unverified, and named rather than implied:**

- Whether FedKD's 17 tuning cells have already completed. No campaign artifacts
  exist on this machine; remote job state was not queried. This sets the cost of
  N10 at 17 cells or fewer.
- The magnitude of N13's pooled-budget deviation. Direction is determinable from
  the code; size needs run artifacts.
- Whether concurrent Ray actors (`client_gpus=0.5`, the production regime) actually
  break bit-exactness. The golden gate runs `client_gpus=1.0`, 2 rounds,
  `experiment=ci` (`golden_diff.py:56, 71-83`), so it does not cover the regime the
  campaign uses. **No positive evidence of breakage** — only that the gate's scope
  is narrower than the claim it underwrites. Verifying it requires a dispatch,
  which agents in this repo do not perform.
- Whether FedKD with momentum closes its 15–27pp gap. Requires a run. What is
  established is only that the gap is currently *confounded*, so ADR-0005's
  architectural attribution is unsupported as written.

## Independent re-confirmation addendum (2026-09-02)

This addendum is authoritative over conflicting statements above. Three
independent code/manuscript passes were reconciled with a direct source inspection.
The local CPU suite established a clean pre-change baseline: **415 passed, one
third-party Ray warning, in 308.44 seconds**. The repository currently contains 35
pytest source files, so the earlier plan's count of 34 is stale.

### N18. FedKD evaluates and checkpoints the pre-fit broadcast, not the aggregate

`strategy.py:214-238` calls `pre_configure_fit`, whose FedKD hook stores the
reconstructed broadcast in `_last_reconstructed` (`strategy_hooks/fedkd.py:141-150`).
After client aggregation, `pre_evaluate` returns that cached pre-fit vector and
ignores the post-aggregation `parameters` argument (`:176-190`). `simulation.py`
then evaluates and writes the final checkpoint from the returned vector
(`:319-327`). Therefore each reported FedKD round and its final checkpoint score
the model sent *into* the round, not the student aggregated *from* that round.

The same state transition corrupts download accounting. `_svd_compress_delta`
advances `_reference` during `pre_configure_fit` (`fedkd.py:95-139`), but telemetry
asks `download_size_bytes` only after aggregation (`telemetry.py:215-218`). The
reported size is recomputed against the advanced reference and the aggregate, not
read from the actual broadcast payload. **This is a dispatch blocker for all FedKD
results**, independent of N10. Repair the lifecycle, cache the exact broadcast
report before advancing state, evaluate the real aggregate, and test both vectors
and byte reports at the production seam.

### N19. Real bit packing requires a wire protocol, not a serializer substitution

`quantization.py:42-67` currently serializes codes as `int64`. Replacing the dtype
with a generic bit packer is insufficient. The format must define a version,
signed-code mapping, bit order and endianness, tensor boundaries and shapes, scale
metadata, empty tensors, padding, and exact round-trip validation. Error-feedback
and diff coding enlarge the code domain; `postprocess.py:133-154` can require
`q+1` bits, and a precision change makes the previous and current alphabets
different. The safest semantics are to reset/cold-start differential state when
`q` changes and encode same-`q` differences with an explicitly derived width.

The `q=32` FedMAQ rung should still be removed under ADR-0002. A packed `q=32`
FedPAQ rung is a **32-bit quantized bit-parity reference**, not an uncompressed
control: it still carries quantizer metadata and may pass through zlib. FedAvg is
the uncompressed control. Budget-parity tests must pin `ceil(log2(255)) = 8` for
the DAdaQuant/FedPAQ comparison.

### N20. A partial final checkpoint can be mistaken for a completed run

`checkpoint.py:56-64` writes directly to the final path and catches write errors.
`scripts/common.py:149-163` treats the path's existence as completion, and the
current test suite pins even a zero-byte sentinel as complete. A killed write can
therefore cause `--skip_completed` to skip a failed run. Write to a same-directory
temporary file and atomically replace the final path; then require a loadable
state-dict plus a final-round telemetry/manifest agreement before completion.
Failures must propagate or leave no valid sentinel.

### N21. The selection pipeline can emit a verdict after closure has failed

The existing analysis path contains the per-seed defect in N13 and also permits
later output generation after warning about missing closure. The replacement path
must fail closed on missing expected identities, duplicates, truncated round
series, non-monotone cumulative bytes, non-finite values, and budgets outside an
arm's observed support. No winner or comparison artifact may be written after a
closure failure. Historical integer-formulation analysis must remain labelled and
isolated; the power-mean selector must emit the actual `p` and `omega` identity and
be the sole authority for the replacement campaign.

`accuracy_at_budget` also needs a complete preregistered contract: budget scope,
interpolation, seed balance, no-extrapolation behavior, and the adoption margin.
Those are author decisions, not implementation details.

### N22. The memory-capacity probe crosses the stated server/client trust boundary

`quantization_planner.py:378-423` reads a client's private training batch and runs
a forward/backward pass in server-side planning code. That contradicts the stated
system model even if it is convenient in a centralized simulation. A defensible
repair must either implement a real client-local preflight protocol or explicitly
model the current operation as an emulation of that client-local step, charge its
client computation and scalar/control communication, and remove every claim that
the deployable server observes raw client data.

### N23. The memory model is synthetic, not a Raspberry Pi deployment trace

`strategy.py` samples continuous `U(2048, 16384)` MiB capacities, while the ADR and
manuscript sometimes describe discrete Raspberry Pi SKUs. The `512 MiB/bit`
conversion is a design normalization, not an empirically grounded hardware law.
The cost model also contains no channel contention and does not establish
memory-caused stragglers. Either switch to a defensible discrete hardware model or
state the continuous synthetic abstraction, justify the normalization, and add a
predeclared sensitivity analysis before making robustness claims.

### N24. The golden transition and the final golden gate are different artifacts

An old-versus-new comparison is expected to differ after semantic repairs and can
only classify the transition; it is never a PASS. The final gate must compare two
independent, clean-tree captures of the same candidate semantics and pass bit
exactly. Preserve both artifacts, enforce clean source state in the harness, and
add a repeated determinism probe at the production `client_gpus=0.5` regime. A
later pass cannot retroactively validate an earlier change.

### N25. A `post_process: true` FedPAQ YAML is currently a silent no-op

`baselines/__init__.py:98-110` only constructs the post-processing hook when the
algorithm name is `fedmaq`; `tests/test_postprocess.py:218-229` explicitly pins the
FedPAQ flag as ignored. The requested comparison therefore needs a distinct,
registered run identity (for example `fedpaq_pipeline`), dispatch validation,
analysis labels, manifest coverage, and a test proving that the pipeline is active.
A configuration-only change would silently reproduce ordinary FedPAQ.

### Revised dispatch verdict

**Do not dispatch or reuse any replacement campaign row until N10-N30 and the
accepted protocol decisions are represented by code tests, configuration-schema tests,
analysis closure checks, updated ADR/manuscript contracts, and a clean candidate
golden PASS.** Existing raw artifacts may be retained for historical diagnosis,
but test-selected decisions, FedKD results, and communication values affected by
packing/accounting changes cannot be promoted into the replacement evidence set.

### Author resolutions recorded 2026-09-02

The author accepted the six recommended branches left open by the re-confirmation:

1. deterministic stratified 50/50 CIFAR-10 selection-validation/reserved-test split;
2. remove ternary `q=1` and set `q_min=2`;
3. charge and describe the gradient probe as simulation emulation of client-local
   preflight behind a summary-only semantic boundary;
4. retain Raspberry Pi 5 and NVIDIA L40S as client/server **compute-modeling
   references**, while removing literal deployment-population claims; keep memory
   capacity as a separately declared continuous synthetic variable, identify
   `c_unit` as a normalization, and run the `{256,512,1024}` sensitivity;
5. balance every matched-tuning candidate at five seeds and apply the per-seed
   common validation-byte/interpolation/no-extrapolation contract; and
6. limit the registered FedPAQ-pipeline comparison to the 15-cell primary grid,
   using the FedPAQ `q` selected by matched tuning so that the shared coding pipeline
   is the only treatment difference.

The hardware distinction is deliberate: using a named device to ground a throughput
parameter does not imply that every synthetic client has that device's installed
RAM SKU or that the experiment validates a real Raspberry Pi/L40S deployment.

## Adversarial plan-review addendum (2026-09-02)

A fresh isolated Luna-high review compared the complete remediation plan against
`CONTEXT.md`, ADR-0021, the execution model, the matrix definitions, and the current
golden-gate skill. Direct parent-agent verification accepted the following findings.

### N26. The 415-cell total was right, but its stage labels and dispatch topology were wrong

The prior plan reached 415 only because its row labelled `ablation = 12` was really
the mandatory selected-`p` omega follow-up. The existing 147-cell downstream set
already contains 105 benchmark, 36 ablation, and six uniform-memory cells. Treating
the campaign as one frozen dispatch also violated the selection dependencies in
[the execution model](../agents/execution-model.md): matched-tuning verdicts configure
Stage 1a; Stage 1a selects `p` before the omega matrix can be materialized; Stage 1b
selects `(p,omega)` before downstream configs and ablations can be frozen.

The corrected mutually exclusive scientific ledger is:

| Ordered stage | Cells | Role |
| --- | ---: | --- |
| balanced matched tuning | 145 | 29 candidates x five selection-validation seeds |
| Stage 1a power-mean degree | 84 | 14 arms x two skews x three selection-validation seeds |
| Stage 1b selected-`p` omega | 12 | two net-new omega values x two skews x three seeds |
| downstream core | 147 | 105 benchmark + 36 ablation + six uniform-memory reserved-test cells |
| FedPAQ-pipeline control | 15 | five primary dataset/skew cells x three seeds |
| net-new memory sensitivity | 12 | 18 sensitivity cells minus six `c_unit=512` benchmark overlaps |
| **total** | **415** | exact-set expansion must still prove this before freeze |

This table records design arithmetic, not live dispatch state. The pinned issue and
generated stage manifests remain authoritative for actual run status and counts.

The working total is unchanged, but it is now dispatched as `145 -> 84 -> 12 ->
174` with a fresh manifest/envelope gate at every selection-dependent boundary.
Golden, concurrency, and smoke executions remain a separately counted assurance
ledger rather than being mislabeled as scientific cells.

### N27. The current golden skill cannot implement the required replacement PASS

The current `jupyterhub-golden-gate` skill and `golden_diff.py` define PASS as an
old-commit capture matching a new-commit compare. Intentional RNG, packing, and
FedKD lifecycle repairs must differ, so that operation can only be a transition
diagnostic. A new blocker now requires two explicit operations: preserved
old-to-new change classification that never emits PASS, and two independent clean
captures of the same candidate with separate roots and bit-exact repeatability PASS
semantics. The skill, harness, evidence schema, and cleanup behavior must be revised
and tested before they are used.

### N28. Completion must prove every round, not only the final artifact

A loadable final checkpoint plus a final-round row can coexist with missing rounds
1 through `R-1`. If `--skip_completed` accepts that state, the runner skips a
truncated run and analysis discovers the hole only after the recovery opportunity.
The completion contract now requires an atomic checkpoint, matching identity,
contiguous unique telemetry for every round `1..R`, finite required values, and
monotone cumulative metrics. Missing-intermediate-round and duplicate-round tests
are mandatory negative cases.

### N29. Exact-set and treatment contracts need to be preregistered, not inferred

Analysis must enforce `discovered_ids == expected_ids` within an explicit stage
evidence root; rejecting only missing/duplicate cells still admits historical or
unexpected rows. The implementation plan now also fixes these formerly ambiguous
contracts:

- FedMAQ adaptive `Q = {2,3,4,5,6,7,8,16}` and matched-tuning
  `q_max in {4,6,8,16}`;
- FedPAQ matched-tuning `q in {2,4,8,16,32}`;
- FedPAQ-pipeline inherits the selected FedPAQ `q`, seeds, model, rounds, and primary
  cells, changing only the registered coding pipeline;
- the gradient-preflight summary uses a versioned, byte-tested control-message
  schema with declared direction and multiplicity;
- the FedMAQ smoke fixture must prove an actual `q` transition in telemetry;
- a fresh replacement source manifest and assurance envelope supersede, but do not
  rewrite, the historical authority; and
- `just check` plus a fresh independent post-implementation review are explicit
  pipeline-freeze gates.

### N30. Selection rules are now stage-specific and validation-only

“Accuracy at budget” is not one underspecified universal selector. The revised plan
defines three versioned selectors. Matched tuning uses five paired seeds per
candidate and preserves ADR-0011's strict `sqrt(2) * SD(reference)` adoption margin
on interpolated validation accuracy. Stage 1a ranks only the seven eligible `p`
values at per-seed common validation-byte budgets; structural controls remain
descriptive. Stage 1b applies the same support/interpolation rule to
`omega in {0.25,0.5,0.75}` at the selected `p`. None uses an extra accuracy floor;
complete in-support validation curves are the eligibility condition. Test-split
inputs to any selector are a hard failure.

### Plan-review disposition

All critical and significant review findings were either incorporated above or
resolved by an explicit contract. No production code, configuration, frozen
artifact, or experiment output was changed during this review. On 2026-09-02, the
author confirmed sole authority over the experimental pipeline and authorized
implementation without an external adviser gate. The plan is therefore ready for
implementation, but remains neither implementation-complete nor dispatch-ready. A
fresh independent review of the implemented diff and generated manifests is still
required before the author can declare pipeline freeze.

### N31. A second full audit is a mandatory implementation-completion gate

The author requires a new audit after the remediation plan is implemented. This is
broader than the focused independent diff review: it must again inspect the entire
experiments codebase and the full thesis manuscript for logical, methodological,
implementation, configuration, analysis, telemetry, run-ledger, hardware-model, and
claim inconsistencies. Every finding receives a durable disposition. Every
substantiated in-scope defect must be corrected in code, configuration, tests, ADRs,
analysis, or manuscript prose as appropriate, followed by affected verification and
another audit pass. Pipeline-freeze consideration is blocked until the report is
bound to the exact candidate and assurance envelope and records no undisposed
critical/significant finding or known correctness/methodology error.

The implementation contract and live execution state are owned by
[issue #86](https://github.com/FedMAQ/fedmaq-experiments/issues/86).

### N32. The standalone type gate is currently red and absent from `just check`

The 2026-09-02 pre-publication verification confirmed that `just check` passes its
freeze, Ruff, and 415-test suite, but `just typecheck` fails in the Flower
client-properties compatibility path. The current typed `ClientProxy` interface
requires `group_id`; the legacy-signature fallback omits it and mypy reports the
missing positional argument. This is pre-existing source behavior, not introduced by
the audit document. B11 must repair or isolate the compatibility boundary, add the
type gate to enforced CI/repository checks, and reach a green standalone mypy run
before pipeline-freeze consideration.
