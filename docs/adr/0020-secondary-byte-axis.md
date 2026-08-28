# ADR-0020 — Secondary byte axis: DAdaQuant's as-published coder, parallel to the measured seam

**Status**: Accepted · 2026-08-27
**Related**: ADR-0018 (the primary `measure_bytes` seam this is deliberately *not* folded into); 2026-08-26 baseline-implementation audit (X1 finding, the source of the keep decision)

## 2026-08-28 disclosure correction

The methods chapter now reports this coder as an implemented DAdaQuant-only
secondary axis rather than saying it was not reproduced. It also separates the
paper's model-update delta transmission from FedMAQ's thesis-introduced temporal
code differencing. The latter subtracts prior-round integer codes and is not
attributed to Hönig et al.

## Context

The primary byte seam (ADR-0018, #25) holds one encoder constant across every arm so the comparison isolates compression *policy*, not transport engineering. That is the correct primary axis, but it invites one specific, legitimate objection: DAdaQuant's own paper (Hönig et al. 2022) specifies 0-run-length encoding plus Elias omega coding as its transport stage, and measuring it under a generic zlib pass instead compares against a weaker DAdaQuant than the one actually published. The 2026-08-26 audit's X1 finding names this directly — DAdaQuant is the one baseline whose paper mandates entropy coding, and the implementation grants it none while granting FedMAQ its own real compressor. The objection runs in FedMAQ's favor, which is exactly the direction this project committed to not being casual about.

Issue #26 scoped this as droppable-by-schedule, with keep-or-drop routed through a `wayfinder` Decision separate from implementation (per the #24/#25/#26 cluster's `to-spec` → `to-tickets` → `implement` routing, itself adopted to avoid three independent sessions reproducing X1 on one seam). The decision — made 2026-08-27, in the same session as implementation, with no schedule pressure reported — was **keep**: the scope is bounded (one coder module, one test file, a few lines of plumbing) relative to the 109-hour rerun it sits inside, and dropping it would leave a known pro-FedMAQ gap disclosed but unaddressed rather than closed.

## Decision

**A second, parallel measurement, not a substitute for the primary one.** `dadaquant_coder.py` implements 0-RLE + Elias omega as `dadaquant_pack`/`dadaquant_unpack`, exercised only by `DAdaQuantCompressionHook.compress()` via an `on_codes` callback threaded through `_quantize_deltas` (the shared FedPAQ/DAdaQuant skeleton stays unaware of *why* the callback exists — it just exposes each tensor's `(codes, scale)`, matching the primary axis's coverage of both the `scale > 0` and all-zero branches). The result is stashed on `CompressionHook.last_secondary_bytes: int | None`, `None` by default so "not applicable" is distinguishable from a measured zero — every hook except DAdaQuant's leaves it at that default.

**Scale metadata travels outside the coder, not through it.** The primary seam's contract (ADR-0018) puts metadata *inside* the encoded payload so nothing routes around `measure_bytes`. This axis does the opposite deliberately: DAdaQuant's paper codes quantization *levels*, not a single per-tensor float32 normalization constant, so the scale is added as 4 raw bytes on top — the same `+4` convention every analytic-formula baseline already uses. Applying Elias omega to a value that isn't a quantization code would misrepresent what the paper's coder actually does.

**Difference coding — the third component the paper names alongside 0-RLE and Elias omega — needed no new code.** Every arm already quantizes parameter *deltas* (the difference between incoming and locally-updated weights), not raw weights, so this requirement was already structural before #26 existed. Issue #26's own scope reflects this: it names only 0-RLE and Elias omega as implementation work.

**The audit requirement is satisfied by evidence already in hand, not re-derived.** #26's scope calls for auditing whether any baseline besides DAdaQuant has a paper-specified transport coder. The 2026-08-26 audit already checked all seven baselines against their source papers for exactly this and found DAdaQuant alone (X1) — FedPAQ, FedKD, FedDistill, FedAvg and FedProx are unflagged. No new audit pass was run; this ADR is the citation.

**Logging is additive and conditional, not a schema change for every arm.** `secondary_bytes_uploaded` is excluded from the generic per-client numeric-key auto-average (alongside `bytes_uploaded`/`payload_bytes`) and instead summed explicitly into `RoundSnapshot.round_secondary_bytes: int | None`, `None` when no client in the round reported one. `communication/round_secondary_bytes` is in the CSV's stable schema, but stays blank (not `0`) on every non-DAdaQuant row via `restval=""`.

## Consequences

- Any future baseline found to have its own paper-specified transport stage gets the same treatment: a hook-local coder module, `last_secondary_bytes` set on its `CompressionHook`, a `ClientFitStrategy._extra_fit_metrics` override — not a change to the primary seam or to arms that don't need it.
- `communication/round_secondary_bytes` is available per round for DAdaQuant runs without re-running training, satisfying #26's own "must land before dispatch" constraint — both totals are logged together going forward.
- §4 must report both axes for the DAdaQuant comparison, not just the primary one, per #26's own AC — if the two disagree qualitatively, that disagreement is itself a finding, not a discrepancy to resolve away.
- This lands last in the #24/#25/#26 cluster (per the combined `to-spec`), after both landed. Encoding an already-computed `codes` array a second way draws no new RNG and reorders no existing draws, so ADR-0006's own re-baseline trigger ("any change that reorders RNG-consuming operations") does not fire — **but `golden_diff.py`'s gate is broader than that: it also asserts exact CSV column-set equality (`scripts/golden_diff.py::_diff`)**, and `communication/round_secondary_bytes` is a new column in every algorithm's schema (`_COMMON_CSV_FIELDNAMES`), not just DAdaQuant's. `outputs/golden/step2/` (gitignored, local-only, not part of this commit) needs a fresh `capture` before the next `compare` — otherwise every arm in `GOLDEN_SET`, not only `dadaquant`, will show a spurious "column set differs" failure that has nothing to do with a real regression.
