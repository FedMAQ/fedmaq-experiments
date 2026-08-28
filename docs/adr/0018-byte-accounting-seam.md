# ADR-0018 — Byte-accounting seam: one `measure_bytes`, and why counts alone don't satisfy reproducibility

**Status**: Accepted · 2026-08-27
**Related**: ADR-0006 (the golden-diff gate this required a re-capture against); ADR-0012 (the cumulative-MB terms this seam ultimately feeds)

## 2026-08-28 correction — the held-constant seam covers both directions

The original decision closed the upload seam but overstated its coverage of the
headline total-byte metric. `communication/round_bytes` is bidirectional: it sums
one server broadcast per sampled client plus every client upload. The default
download hook still returned raw `nbytes`, while FedKD alone measured its SVD
download through `measure_bytes`. The primary axis therefore mixed two instruments.

The default download path now serializes each non-empty tensor as float32 and
routes every per-tensor payload through `measure_bytes`, matching the identity
upload convention. FedKD retains its method-specific SVD payload shape but uses
the same encoder. With `log_payloads=true`, exact download payloads are stored
separately as `payloads/download_round_NNNN.pkl`, keyed by recipient; replaying
every stored call reproduces the aggregate download leg. The established upload
file `payloads/round_NNNN.pkl` remains backward compatible.

This is an accounting change across every ordinary-download arm and a model-output
change for none. It invalidates prior communication-byte goldens and requires a
fresh GPU-host capture and compare at the exact pre-registration commit.

### Same-day follow-up — FedDistill's download override

That sweep enumerated two download paths, the default hook and FedKD, and missed
a third. `strategy_hooks/feddistill.py` overrides `download_size_bytes` and still
returned raw `nbytes` for both the weight broadcast and the global logit matrix,
even though FedDistill's *upload* leg already measured the same logit payload
through `measure_bytes`. The override now delegates the weight leg to the default
path and measures the broadcast logit payload as its own call — `measure_bytes`
is not additive over concatenation, so the two legs are measured separately and
summed — adding that payload to the returned report so the replay identity holds
for this arm too. On CIFAR-10/MobileNetV2GN this moves
FedDistill's per-round download leg from 8,947,128 raw bytes to 8,176,145
measured: the old figure over-charged a baseline by 9.4%, in FedMAQ's favor.
FedDistill falls under the same fresh-capture requirement as the arms above.
CFD remains the only download path outside the shared transport seam: it uses
its published soft-label coder, and is represented by a report with no
`measure_bytes` payloads. FedMD inherits the shared default download path; its
raw prediction-array accounting remains on its upload leg.

## 2026-08-28 architecture pass 1 — typed reports on both legs

The source re-derivation required by #49 and #50 produced this enumeration,
rather than copying the earlier handoff:

- Upload implementations are `CompressionHook.compress`,
  `FedPAQCompressionHook.compress`, `DAdaQuantCompressionHook.compress`,
  `FedKDCompressionHook.compress`, and
  `FedMAQPostProcessCompressionHook.compress`. The base identity implementation
  is included in the coverage test alongside the four concrete overrides.
- Download implementations are `StrategyHook.download_size_bytes`,
  `FedKDHook.download_size_bytes`, `FedDistillHook.download_size_bytes`, and
  `CFDHook.download_size_bytes`. `PassthroughHook`, `DAdaQuantHook`,
  `FedAvgKDHook`, `FedMAQHook`, and `FedMDHook` inherit the shared default.

Both legs now return the existing frozen `UploadReport` value. Its contract is
`measured_bytes` for the reported communication total, `payload_bytes` for the
pre-encoding companion, nullable `secondary_bytes` for the DAdaQuant-only
as-published axis, and `payloads` preserving one byte string per
`measure_bytes` call. The name is retained to reuse #49's type rather than
introduce a second transport abstraction. Telemetry consumes the returned
download report directly; no string-keyed payload attribute remains.

CFD constructs a report whose measured and payload totals are the output of its
own published soft-label coder and whose payload tuple is empty because that
path does not call `measure_bytes`. This records its ADR-0005 exclusion without
pretending that its coder is replayable through the shared transport. The two
dropped upload arms remain explicit as well: FedMD reports raw prediction-array
bytes, while CFD reports the output of its published soft-label coder. Only
FedMD computes a raw array byte count; the distinction is intentional.

## Context

Byte accounting had grown four independent implementations, not two as first scoped: `_quantize_deltas`'s `ceil(size*bits/8)+4` and `postprocess.py`'s `len(zlib.compress(payload))+4` were the two named in Issue #25's original framing, but FedDistill carried a fifth, undocumented path of its own. Every arm computing its own transmitted-byte arithmetic meant AC1 ("no per-arm arithmetic outside one function") was failing silently — a new baseline could add a sixth path and nothing would flag it.

## Decision

**One function, `measure_bytes(payload: bytes) -> int`, in a new `transport.py`, is the only place transmitted-byte arithmetic happens.** Every `CompressionHook` serializes its payload through a per-hook `serialize(payload_parts) -> bytes` first (metadata travels through the encoder, not around it as a separately-added constant), then calls `measure_bytes` on the result. `cfd.py`/`fedmd.py` are untouched — they're dropped baselines (ADR-0005) and out of scope.

This is deliberately a **measured**, not modeled, quantity: `measure_bytes` runs the real encoder (zlib for the compressed arms) rather than estimating a size from parameters, which is why its output is content-sensitive and not reconstructible from a byte count alone (see AC2 below). Fixed int64 code width is kept by design — not folded into the seam — since narrowing it is a quantization-format change, not an accounting change.

A `payload_bytes` telemetry companion (pre-encoding size, logged as `communication/round_payload_bytes`) ships alongside `measure_bytes`'s post-encoding output, so both the input and output of the content-sensitive step are visible. FedKD's server→client download leg was missed in the first pass (upload legs only) and needed a follow-up companion (`algorithm/fedkd/download_payload_bytes`) once noticed.

**AC2 ("byte totals reproducible offline from logged telemetry") is not satisfied by counts.** zlib's output size depends on the payload's actual bytes, not just its length — a summed `measure_bytes` total can't be re-scored against a hypothetical alternative encoder after the fact, and per-client counts or hashes were both rejected as insufficient for the same reason. The seam's own logged counts satisfy AC1 and AC3 but leave a real reproducibility gap.

That gap is closed by an **opt-in** mechanism, not a default one: `experiment.telemetry.log_payloads` (default `False`) gates whether each hook's actual payload bytes (`last_payloads: list[bytes]`), not just their measured length, are retained and persisted per round (`payloads/round_{NNNN}.pkl`). `attach_payloads_if_enabled` in `client_hooks/base.py` is the single gate checkpoint — every hook's capture path routes through it, so the on/off behavior can't diverge per-arm the way the original byte arithmetic did. Off by default because a multi-MB blob per client per round is a real cost over Flower's simulated Ray object-store channel; no run pays it unless it specifically wants full AC2 reproducibility.

**AC4 (§4 manuscript disclosure of the accounting) was closed on 2026-08-28.** The
methods chapter defines cumulative MB as aggregate bidirectional client--server
traffic and states that both directions use the held-constant encoder while FedKD
changes only its method-specific download payload shape.

## Consequences

- Any future baseline's byte accounting must route through `measure_bytes`/`serialize`, never compute its own arithmetic — that's the whole point of AC1, and it's enforceable by review (one call site) rather than by convention.
- Because `measure_bytes`'s output is what golden-diff bit-exactness is checked against, unifying it required a full golden-diff re-capture (ADR-0006): the old baseline at `outputs/golden/step2/` no longer reflects current behavior and was overwritten 2026-08-27 (pre-unification copy kept at `outputs/golden/step2_old/`).
- `log_payloads` defaults off, so **most runs get AC1/AC3-grade counts only, not AC2-grade reproducibility** — a run intended to support an offline re-score against a different encoder must explicitly opt in before it executes, not after.
- Adding raw-payload capture (commit `90a8d5d`) added no RNG draws, so it did not itself require a further golden-diff re-baseline — only accounting/architecture changes that touch `measure_bytes` or the training loop do.
- The headline quantity is aggregate bidirectional traffic, not uploaded bytes or
  MB per client. `communication/client_bytes_uploaded_mean` remains a separate
  per-client upload descriptor.
