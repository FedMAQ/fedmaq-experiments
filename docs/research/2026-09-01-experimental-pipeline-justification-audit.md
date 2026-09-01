# Experimental-pipeline justification audit

Date: 2026-09-01. Scope: the current FedMAQ experimental design, its stated
rationale, and implementation alignment. This is a static audit, not a results
review. It distinguishes a design that can be defended with a narrow claim from
a design that has already established an empirical outcome.

## Verdict

The study has a coherent comparison architecture: a shared image-classification
task, common training settings, paired seeds and partitions, mechanism-defined
baselines, a controlled FedMAQ memory-blind ablation, and measured serialized
communication. Those choices are generally defensible at thesis scope.

It is not yet defensible to present the whole pipeline as deployment-feasible or
fully frozen. Four items must be carried as explicit limitations or repaired
before any final methodology/results defense:

1. the nominal bit-widths are not packed into nominal-width payloads;
2. the claimed Late-2023 Pi 5 capacity fleet is historically inconsistent;
3. the implementation does not make low-memory clients slower, despite prose
   that says it does; and
4. the current power-mean recut has not yet selected and materialized its
   downstream formulation and ablation bindings.

The companion [precision and memory-path audit](2026-09-01-precision-memory-path-audit.md)
documents the first two issues in detail.

## Decision ledger

| Decision | Evidence | Classification | Defensible statement and boundary |
| --- | --- | --- | --- |
| Common task and backbone | The formal grid uses CIFAR-10, CIFAR-100, and FEMNIST; CIFAR clients use MobileNetV2GN and FEMNIST uses SimpleCNN. The iso-architecture decision applies the full MobileNetV2GN to FedMAQ and the ordinary baselines, with a deliberate compact-student exception for FedKD ([ADR-0004](../adr/0004-confirmatory-grid-design.md), [ADR-0005](../adr/0005-baseline-stack-membership.md)). | Grounded | The ordinary-arm comparison controls the shared model architecture. FedKD is a published-mechanism bundle, not an iso-architecture causal comparison. CIFAR-to-FEMNIST transfer also changes both architecture and partition type. |
| Statistical heterogeneity | CIFAR-10/100 use Dirichlet `alpha` values 0.1 and 1.0; FEMNIST retains its writer partition ([heterogeneity configs](../../conf/heterogeneity/), [experiment-design rule](../../.agents/rules/experiment-design.md)). | Grounded | The study examines severe and moderate synthetic label skew on CIFAR, then a natural writer-partitioned task. It does not isolate all possible non-IID dimensions or claim that either CIFAR split reproduces a deployment population. |
| Shared training schedule | The default configuration fixes `K=100`, `C=0.1`, `E=5`, batch size 64, 100 rounds, and shared optimizer settings ([default config](../../conf/experiment/default.yaml)); the manuscript states that common settings are applied across arms ([Chapter 4](../../../fedmaq-manuscript/chapter_4.tex)). | Qualified | Equal settings control a comparison and keep the run budget tractable. They are not jointly proven optimal for every baseline or dataset; the baseline tuning policy addresses one declared trade-off knob per method, not every possible hyperparameter. |
| Server-only proxy set | The configuration holds 3,000 proxy samples; the manuscript specifies a label-stratified training-pool split before client partitioning and discards labels for server KD ([default config](../../conf/experiment/default.yaml), [Chapter 4](../../../fedmaq-manuscript/chapter_4.tex)). | Grounded for protocol; qualified for privacy | This isolates a common server-side distillation reference and prevents client-shard overlap in the stated split. Server-only access is a design boundary, not a formal privacy guarantee or a claim that such proxy data is available in every deployment. |
| FedAvg plus ensemble KD | The FedMAQ hook receives Flower's data-size-weighted aggregate, uses it as the student initialization, and distills reconstructed selected-client teachers into it ([hook](../../src/fedmaq/core/strategy_hooks/fedmaq.py), [KD routine](../../src/fedmaq/core/kd_utils.py)). | Grounded as implementation | KD is a testable functional-refinement mechanism after parameter aggregation. It is not a proof that the aggregate will improve, recover quantization loss, or solve non-IID drift in every condition. |
| Memory-aware precision rule | The planner computes a Tier-1 `min()` clamp from sampled capacity and a Tier-2 target from configured client signals ([planner](../../src/fedmaq/core/quantization_planner.py)); the control/ablation deliberately remove the Tier-1 rule ([ADR-0004](../adr/0004-confirmatory-grid-design.md)). | Qualified | This supports a mechanism comparison: whether the capacity-derived precision allocation changes the measured accuracy--communication trade-off. It is not full RAM accounting, client admission control, or deployment certification. |
| Standard-baseline comparison | The six baselines occupy distinct mechanism categories; baseline knobs receive a declared matched-tuning stage and all reported arms use the held-constant byte-measurement seam ([experiment-design rule](../../.agents/rules/experiment-design.md), [ADR-0011](../adr/0011-baseline-matched-tuning.md), [transport seam](../../src/fedmaq/baselines/transport.py)). | Grounded with stated asymmetry | Baselines need not be rewritten to consume FedMAQ's capacity signal. The within-FedMAQ no-resource control is the appropriate estimate of the Tier-1 rule's contribution. This does not establish that each baseline is physically feasible on every modelled client. |
| Communication measurement | Each arm routes serialized payloads through one zlib measurement function; a distinct DAdaQuant as-published axis is retained ([transport seam](../../src/fedmaq/baselines/transport.py), [ADR-0018](../adr/0018-byte-accounting-seam.md)). | Grounded for the defined measurement | Comparative claims may concern the logged measured serialized bytes under this held-constant transport. They may not substitute nominal quantizer width for actual wire cost; see the companion audit. |
| Estimated time | The cost model applies uniform bandwidth and compute arrays, then uses payload size and local sample count to form each client's delay ([strategy](../../src/fedmaq/core/strategy.py), [telemetry](../../src/fedmaq/core/telemetry.py)). | Problem in current prose alignment | The model can compare algorithmic cost under stated fixed bandwidth/compute assumptions. It does **not** currently make the sampled memory capacity itself a straggler-time input. Remove or repair any statement that it does. Absolute latency remains a cost-model output, not deployment observation. |
| Formulation selection and freeze | ADR-0021 makes the power-mean family the current formulation authority, replacing the historical F1--F4 candidate set. The live campaign requires selection, downstream materialization, and an ablation-binding repair before reported runs ([ADR-0021](../adr/0021-power-mean-formulation-family.md), [issue #22](https://github.com/FedMAQ/fedmaq-experiments/issues/22), [issue #45](https://github.com/FedMAQ/fedmaq-experiments/issues/45)). | Incomplete by design | Teach the power-mean family as the current planned selection protocol. Do not teach the checked-in historical `formulation: 2` as the final selected method, and do not make performance claims before the required stages and freeze are complete. |
| Three seeds and paired comparison | The protocol shares seeds and partitions across arms and reports descriptive paired deltas, mean, and spread rather than formal significance ([ADR-0004](../adr/0004-confirmatory-grid-design.md), [Chapter 4](../../../fedmaq-manuscript/chapter_4.tex)). | Grounded with a clear limitation | Pairing reduces avoidable seed/partition noise. Three seeds support descriptive stability evidence, not a general claim of statistical significance or population-wide superiority. |

## Findings that require action or explicit limitation

### Precision is not wire packing

The implementation represents all quantization codes as `int64` values before
zlib compression. A selected 3-bit or 7-bit resolution changes the quantizer and
may change zlib compressibility, but it does not deterministically send a 3-bit or
7-bit packed code per element. Retain the current implementation only if the
study reports measured serialized bytes and removes nominal-bit payload-ratio
claims. A true packed-code redesign changes the protocol and requires a new
validation/freeze path.

### Hardware labels need temporal correction

Raspberry Pi's 2023 launch offered 4- and 8-GB Pi 5 variants; the 2-GB model was
announced in August 2024 and the 16-GB model in January 2025. The 2--16 GB range
can remain a simulated capacity calibration, but it is not a contemporaneous
Late-2023 Pi 5 fleet. The protocol must reframe that wording before it becomes a
defense claim. [Raspberry Pi's 2-GB announcement](https://www.raspberrypi.com/news/2gb-raspberry-pi-5-on-sale-now-at-50/)
and [16-GB announcement](https://www.raspberrypi.com/news/16gb-raspberry-pi-5-on-sale-now-at-120/)
are the primary evidence.

### Memory does not presently create stragglers

The implementation creates a uniform array for bandwidth and compute speed.
Sampled memory reaches the FedMAQ quantization planner, not the delay formula.
If the intended study claim is only that capacity limits precision, this is a
prose correction. If the intended claim is that lower memory itself creates longer
round time, the model needs an explicit memory-to-delay relation and a new
validation path.

### The active formulation is not the historical configuration

The old numeric formulation remains in `fedmaq.yaml` to preserve the frozen v1
record. The current power-mean design is explicitly a recut. Its selection and
the repair that binds the ablations to the selected `(p, omega)` are still future
gates. This is healthy experimental discipline, but it means the next OCE lesson
must distinguish historical implementation from the live planned protocol.

## Panel posture

The honest high-level defense is not “every choice is physically exact.” It is:

> We fixed a controlled, reproducible comparison environment, grounded the
> components we could ground, and separated the mechanism-level question from
> deployment certification. We evaluate whether a capacity-derived precision rule
> changes the measured accuracy--communication trade-off under stated non-IID
> conditions. We do not claim that the simulation certifies end-device memory
> feasibility, real-world latency, or universal superiority. Where implementation
> and wording diverge, we correct the wording or revalidate the mechanism before
> making the claim.

That posture is defensible only after the identified precision, hardware-label,
time-model, and current-recut boundaries are carried into the manuscript and oral
preparation.

## Next teaching step

Do not create a blanket “why every choice is justified” lesson. Create a
methodology-defense lesson only after these findings are accepted. It should use
three labels for each decision: **design rationale**, **claim it supports**, and
**claim it does not support**. Its first topic should be the distinction between a
simulated precision-allocation cap and complete device-memory feasibility.
