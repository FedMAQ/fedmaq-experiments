# ADR-0003 — TrainingSkeleton seam: a narrow `run_epochs` atom, not one class per baseline

**Status**: Accepted · 2026-07-23
**Related**: `docs/plans/architecture-deepening.md` Step 2 (Candidate A); `DECISIONS.md` Decision 40 (bit-exact golden-diff gate).

## Context

The plan's original framing for Step 2 was one deep `TrainingSkeleton` owning "the SGD+CE+accuracy loop, delta→compress→reconstruct tail, and metrics dict," shared across all five client hooks (`standard`, `fedkd`, `feddistill`, `cfd`, `fedmd`). Reading all five in full before implementing showed that framing only fits two of them:

| Hook | Loops per `fit()` | Optimizer | Upload payload |
| :-- | :-- | :-- | :-- |
| `standard.py` | 1 | single-model SGD | reconstructed weights (delta→compress→reconstruct) |
| `fedkd.py` | 1 (joint) | joint student+teacher SGD | reconstructed student weights (delta→compress→reconstruct) |
| `feddistill.py` | 1 | single-model SGD | **raw** updated weights (uncompressed) + serialized logit matrix |
| `cfd.py` | 2 (distill phase, then CE phase) | single-model SGD (same instance across phases) | quantized soft-label codes on a public set (no weights at all) |
| `fedmd.py` | up to 4 (pub-pretrain, priv-pretrain, digest, revisit), conditional on round/disk state | single-model SGD | public-set predictions (no weights at all) |

Forcing one class to own the loop *and* the tail *and* the metrics shape across all five would require the abstraction to accommodate joint optimizers, variable loop counts, and three incompatible upload payload shapes — a "broad" seam that widens to fit its least-similar member, which is the failure mode the plan's own Candidate D writeup (StrategyHook seam, deferred past thesis) already warns against.

## Decision

Split into two independent, narrow pieces instead of one skeleton class:

1. **`run_epochs(model, loader, optimizer, epochs, step_fn, on_after_backward=None) -> AggregatedMetrics`** — the single-model batch-loop atom (`zero_grad → forward → step_fn(outputs, labels) → backward → [on_after_backward()] → optimizer.step()`, with loss/accuracy accumulation). This is the truly universal repeated structure:
   - `standard.py` calls it once. `step_fn` wraps the existing `client.loss_hook.compute_loss(...)` call, so FedProx/KD/DAdaQuant/FedMAQ variants keep routing through their existing loss-hook seam unchanged — `run_epochs` does not need to know about them.
   - `feddistill.py` calls it once, with `LogitTracker.update()` folded into `step_fn`.
   - `cfd.py` calls it twice (distill-phase `step_fn` returns KL loss against server soft labels over `public_loader`, no labels/accuracy; CE-phase `step_fn` returns cross-entropy over `trainloader`, with accuracy).
   - `fedmd.py` calls it up to four times (pub-pretrain, priv-pretrain over `trainloader`; digest over `public_loader` with L1 loss, no labels; revisit over `trainloader`), each single-model.
   - **`on_after_backward`** is an optional no-op-by-default, read-only, post-backward/pre-step callback — the slot for `StandardFit`'s FedProx grad-norm instrumentation (`standard.py:98-104`, which reads `p.grad` between `backward()` and `step()`). It must not touch gradients or otherwise affect the optimizer step, only observe.
   - **Accuracy is optional in the aggregated result.** Phases that iterate a loader without labels (cfd's distill phase, fedmd's digest phase) pass a `step_fn` that returns loss only; `run_epochs` must not assume `(loss, correct, total)` is always available.
   - **The aggregated result carries `last_loss`** (final batch's loss, not just the epoch-average) — `FedMAQFit._reported_local_loss` reports this, not the mean.
2. **`fedkd.py` keeps its own hand-rolled loop.** Its joint student+teacher optimizer over combined parameter lists does not fit a single-model atom; forcing it in would re-widen the seam for one baseline's benefit. It is migrated in S2a alongside `standard` only via piece 3 below, not via `run_epochs`.
3. **`compress_and_reconstruct(original_params, deltas, compressor_hook) -> (reconstructed_params, byte_size)`** — a small shared helper for the delta→compress→reconstruct tail, used only by `standard` and `fedkd` (the only two baselines that have this tail at all). Takes the original (pre-training) params, not just deltas — reconstruction is `[o + cd for o, cd in zip(original_params, compressed_deltas)]`, which needs both.

No shared "return upload payload" abstraction is imposed beyond piece 3. `feddistill`/`cfd`/`fedmd` each keep their own upload-assembly code (raw weights + logits; quantized codes; predictions) — these are genuinely different payload types, not a common interface hiding behind naming.

No unification of the metrics dict. Step 1 (`StrategyHook.metric_keys()`) already lets each algorithm declare its own CSV columns; a common metrics shape here would be scope creep on top of an already-closed seam.

## Consequences

- **`standard.py` is not one behavior for golden-diff purposes.** It's the shared `fit()` entry for FedAvg, FedProx, FedPAQ, FedAvgKD, FedDistill's non-hook path, plus the `DAdaQuantFit`/`FedMAQFit` subclasses — each exercises a different branch (FedProx: `on_after_backward` instrumentation + prox-penalty loss hook; KD: `ClientKDLossHook`; DAdaQuant: `_pretrain_local_loss` + dynamic `q`; FedMAQ: `_reported_local_loss` reading `last_loss`). The S2a golden set must run **one config per branch**, not just one algorithm — a harness that only diffs plain FedAvg passes green while silently breaking FedProx or KD, since those branches are never exercised by the diff.
- S2a scope (per the plan) is: build `run_epochs` + `compress_and_reconstruct`, migrate `standard.py` onto both (all its branches), migrate `fedkd.py` onto `compress_and_reconstruct` only (its loop stays hand-rolled). Bit-exact golden-diff each branch against pre-refactor output (Decision 40).
- S2b scope is: migrate `feddistill`/`cfd`/`fedmd` onto `run_epochs` only (called 1/2/4 times respectively), with each hook's own upload-assembly code untouched. Bit-exact golden-diff each.
- `run_epochs` must preserve the exact existing per-batch sequence (`zero_grad → forward → loss → backward → step`) and epoch/batch nesting order — no reordering of RNG-consuming operations — since this is the mechanism the golden-diff gate checks.
- If a future baseline needs a joint-optimizer loop like `fedkd`'s, it follows `fedkd`'s precedent (hand-rolled, no forced fit into `run_epochs`) rather than widening the atom.
