# Precision and memory-path audit

Date: 2026-09-01. Scope: the configured FedMAQ precision/memory path only:
`q_max`, `c_unit`, sampled capacity, the permissible precision set, and the
actual byte representation. This is a source-and-configuration audit, not an
experiment and not a protocol change. It does not alter frozen configuration,
code, or results.

## Bottom line

The current configuration intentionally makes **16 bits the maximum Tier-2
request**. The `32` in the permissible set is therefore unreachable in the
default FedMAQ path; it remains available only to configurations that raise
`q_max`.

More importantly, a configured `q=3` through `q=7` is a real change to the
quantizer's number of levels and therefore to reconstruction fidelity, but it
is **not a 3- through 7-bit packed wire format**. FedMAQ serializes every code
as `int64`, then measures the result after zlib compression. Raw
`payload_bytes` are independent of `q`; measured bytes can vary with the code
distribution, but need not be monotone. Any nominal claim such as "a 3-bit
client sends 3/32 of the update" is false for this implementation.

The capacity rule is an interpretable *precision-allocation calibration*, not
a measurement of model-training RAM feasibility. It currently gives FedMAQ a
Tier-1 cap, while standard baselines remain memory-blind controls. That
comparison can support a mechanism-level claim about the rule; it cannot
certify that any arm can run a full training process on the modelled hardware.

## Evidence and classification

| Point | Evidence | Classification | Audit conclusion |
| --- | --- | --- | --- |
| `q_max = 16` is intentional | [`conf/algorithm/fedmaq.yaml:1-14`](../../conf/algorithm/fedmaq.yaml) sets `q_min: 1`, `q_max: 16`, and the permissible set. [`docs/adr/0002-hardware-telemetry-grounding.md:156-168`](../adr/0002-hardware-telemetry-grounding.md) explicitly calls this the Tier-2 interpolation cap and says FP32 is not assigned. [`quantization_planner.py:184-256`](../../src/fedmaq/core/quantization_planner.py) clamps `q_hat` to `[q_min, q_max]` before the Tier-1 `min()` and discrete snap. | **Grounded** as configured intent | The 1--16 Tier-2 range is a deliberate design decision, not a mathematical consequence of `c_unit`. Under the default, Tier-1 may have a raw 32-bit ceiling, but the final `min()` cannot exceed 16. |
| 16-bit rather than 32-bit is justified by the stated rationale | ADR-0002 says FP16-to-FP32 gains are expected to be marginal relative to doubled communication cost, but supplies no experiment-specific measurement or precise citation for that broad assertion ([ADR-0002:164-168](../adr/0002-hardware-telemetry-grounding.md)). | **Qualified** | Defend this as a conservative, pre-specified communication--fidelity choice, not as a proved universal sufficiency result. The study tests the configured policy; it does not establish that FP16 always matches FP32. |
| `c_unit = 512 MB` maps capacity to a raw cap | The ADR defines `floor(c_k / 512)` and maps 2/4/8/16 GB to 4/8/16/32 ([ADR-0002:41-52](../adr/0002-hardware-telemetry-grounding.md)). `PhysicalCostModel` samples continuous `U(2048, 16384)` MB when no uniform control is selected ([`strategy.py:78-123`](../../src/fedmaq/core/strategy.py)); the planner computes `floor(c_k / c_unit)` ([`quantization_planner.py:215-256`](../../src/fedmaq/core/quantization_planner.py)). Tier-1 telemetry tests demonstrate a 4-GB client capped at 8 and a 16-GB client capped at 16 under `q_max=16` ([`tests/test_issue28_tier1_telemetry.py:16-85`](../../tests/test_issue28_tier1_telemetry.py)). | **Qualified** | This is a reproducible calibration that gives the named endpoints intuitive ceilings. It does not derive MB-per-bit from model weights, activations, optimizer state, or operating-system memory. Continuous sampling also creates intermediate raw caps such as 3--7, which are policy values rather than named Pi memory SKUs. |
| The claimed Late-2023 Pi 5 capacity tiers are historically accurate | ADR-0002 calls the fleet a Late-2023 ecosystem and labels all 2/4/8/16-GB tiers as Raspberry Pi 5 ([ADR-0002:26-52](../adr/0002-hardware-telemetry-grounding.md)). Raspberry Pi's primary announcements state that the original 2023 Pi 5 had 4- and 8-GB choices; the 2-GB variant launched on 19 August 2024 ([Raspberry Pi, 2024](https://www.raspberrypi.com/news/2gb-raspberry-pi-5-on-sale-now-at-50/)), and the 16-GB variant launched on 9 January 2025 ([Raspberry Pi, 2025](https://www.raspberrypi.com/news/16gb-raspberry-pi-5-on-sale-now-at-120/)). | **Problem** | The 2- and 16-GB capacities are valid Pi 5 configurations today, but not in a Late-2023 fleet. Before defense/results prose, reframe the hardware grounding as a cross-generation Pi 5 capacity range or revise the time label; do not call all four tiers contemporaneous late-2023 hardware. |
| `[1,2,3,4,5,6,7,8,16,32]` permits fine-grained assignment | The set is configured in `fedmaq.yaml` and is the planner default ([`quantization_planner.py:23`](../../src/fedmaq/core/quantization_planner.py)); `_snap_floor` returns the largest permissible value at or below the combined target ([`quantization_planner.py:137-140`](../../src/fedmaq/core/quantization_planner.py)). Numerical-safety tests exercise every listed value ([`tests/test_models_and_algorithms.py:919-932`](../../tests/test_models_and_algorithms.py)). ADR-0002 identifies 4/8/16/32 as conventional hardware formats while retaining fine granularity in 1--8 ([ADR-0002:170-177](../adr/0002-hardware-telemetry-grounding.md)). | **Grounded** for the policy; **qualified** for the hardware wording | Explain 1--8 as fine-grained *quantizer-resolution candidates*, with 4/8/16/32 the familiar format landmarks. Do not describe 3, 5, 6, or 7 as hardware-native formats. A raw target between 8 and 16 snaps to 8; it does not select 9--15. |
| Lower irregular `q` deterministically lowers uplink payload | `_serialize_codes` converts all codes to `int64` and appends a float32 scale, explicitly regardless of nominal `q` ([`quantization.py:42-67`](../../src/fedmaq/baselines/quantization.py)). The primary FedMAQ post-processing hook uses that serializer ([`postprocess.py:110-154`](../../src/fedmaq/baselines/postprocess.py)). `UploadReport` defines raw `payload_bytes` as the sum of payload lengths and measured bytes as the sum of `len(zlib.compress(payload))` ([`transport.py:22-66`](../../src/fedmaq/baselines/transport.py)). | **Problem** for a nominal-bit communication claim | Per non-empty tensor, raw code storage is `8N + 4` bytes regardless of `q`; no 1--8, 16, or 32-bit packing occurs. Lower `q` can still change the code alphabet, quantization error, error-feedback residual, and zlib compressibility. Its measured-byte effect is data-dependent, not a guaranteed `q/32` saving. |
| The data-dependent measured-byte distinction is actually exercised | The serializer documents that zlib may recover low-bit padding but may not at higher precision, and directs empirical checking rather than an assumed drop ([`quantization.py:48-65`](../../src/fedmaq/baselines/quantization.py)). Payload tests verify that reported raw and measured totals reproduce from actual payloads ([`tests/test_payload_capture.py:34-61`](../../tests/test_payload_capture.py)). A local seeded, non-campaign check over an 8,192-element delta produced the same raw `payload_bytes` (65,540) for every `q`, while measured bytes varied: q=3: 678, q=6: 2,116, q=7: 1,958, q=8: 1,971. | **Grounded** for the accounting behavior; **qualified** for general performance | The check confirms the code path, not a general ordering of bit-widths on real model updates. The experiment's primary communication axis remains valid when it is described as measured serialized/zlib bytes. |
| Memory is a real deployment-feasibility cap across the comparison | The planner says `resource_aware=False` lifts the Tier-1 ceiling and describes capacity as a simulated scalar consumed only there ([`quantization_planner.py:201-209`](../../src/fedmaq/core/quantization_planner.py)). The no-resource ablation makes that removal explicit ([`conf/algorithm/fedmaq_no_resource.yaml`](../../conf/algorithm/fedmaq_no_resource.yaml)); ADR-0004 calls the uniform-memory arm FedMAQ in a memory-blind condition and specifies the control equivalence at 8 GB ([ADR-0004:88-126](../adr/0004-confirmatory-grid-design.md)). | **Qualified** | The design can test whether the FedMAQ Tier-1 rule changes the accuracy/communication trade-off relative to its memory-blind ablation and standard baselines. It does not model full client RAM consumption, drop infeasible baseline clients, or certify deployment feasibility. Bandwidth/time accounting is a different simulated quantity and does not enforce RAM feasibility. |

## Consequences for claims and comparison

### Defensible now

- FedMAQ chooses a discrete **quantizer resolution** from the configured set,
  with a Tier-1 capacity-derived ceiling and a Tier-2 quality target.
- The primary communication measurement is the actual post-serialization,
  zlib-compressed byte count, held constant across the ordinary comparison
  arms by the transport seam ([ADR-0018](../adr/0018-byte-accounting-seam.md)).
- Lower `q` may trade reconstruction fidelity against measured communication
  because it alters code distributions; the magnitude and monotonicity are
  empirical outcomes to report, not assumptions.
- The resource-aware arm is a mechanism comparison, bounded to the stated
  simulated precision-allocation rule.

### Do not claim without a repair

- A selected `q` is an actual packed `q`-bit payload or gives a deterministic
  `q/32` communication ratio.
- The 512-MB divisor measures a client process's RAM consumption or makes a
  run physically feasible on that device.
- The 2/4/8/16-GB Pi 5 fleet is contemporaneous Late-2023 hardware.
- A memory-blind baseline has been penalized, dropped, or feasibility-checked.

## Recommendation

There are two coherent paths; do not mix their claims.

1. **Keep the frozen implementation and restrict communication claims.** Treat
   `q` as quantizer resolution, report the logged measured serialized bytes,
   and state their content-sensitive encoder. This path requires the historical
   Pi-5 wording to be corrected/reframed, but it does not require a wire-format
   change.
2. **Repair nominal-bit communication before making nominal-bit claims.** Define
   and implement a true signed-code packing format (or a q-appropriate integer
   representation), include scale/framing overhead, route it through the same
   accounting seam for every affected arm, add round-trip and byte-size tests,
   then regenerate frozen/golden evidence. This is a protocol and measurement
   change, not an accounting-only cleanup, so it must not be slipped into the
   current frozen pipeline.

For the current study, path 1 is the only defensible description until a
separately authorised repair is evaluated. The memory mechanism remains
testable as a simulated allocation policy, but it should not be presented as a
full deployment-feasibility mechanism.
