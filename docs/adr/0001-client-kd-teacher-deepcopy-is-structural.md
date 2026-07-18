# ADR-0001 — The client-KD teacher deepcopy is structural, not a defect

**Status**: Accepted · 2026-07-18
**Related**: `DECISIONS.md` Decision 32 (F9 WONTFIX); audit `docs/audits/fedmaq-code-audit.md` (F9); superseded plan candidate "C" in `docs/plans/architecture-deepening.md`.

## Context

`ClientKDLossHook.on_train_begin` (`src/fedmaq/core/kd_loss_hook.py:54`) does `copy.deepcopy(model)` to snapshot a frozen teacher for the KD-regularization term. Recurring architecture reviews flag this as per-round waste and propose "widen the `LossHook` contract so the hook owns its teacher lifecycle and caches across rounds." This ADR records why that proposal is a dead end, so it stops being re-suggested.

Decisive facts (verified in code):

1. **The hook is rebuilt every round.** `client_fn` (`simulation.py:163`) calls `get_loss_hook(...)` fresh each round; Flower simulation destroys and recreates the client per round. `ClientKDLossHook._global_model` therefore cannot survive across rounds — the *instance* is gone. The hook has no cross-round lifecycle to "own."

2. **The only persistence path is `context.state` / a Ray process-global.** To keep a teacher alive past client re-instantiation it must be stashed in per-node state (as the compressor's error-feedback accumulator is, `simulation.py:168 state=context.state`) or a Ray global. Any "widen the contract" framing collapses into exactly this — the already-rejected prototype (a partition-id-keyed teacher-shell cache).

3. **That prototype was reviewed and reverted the same day** (Decision 32). It keeps a GPU-resident model copy alive per client per Ray worker for the whole run — the Ray/PyTorch VRAM-accumulation class the process-isolated runners exist to prevent (`hydra-config.md`, `flower-patterns.md`). Cache hit-rate isn't even guaranteed: flwr simulation does not pin partitions to actors.

4. **The prize is negligible.** The deepcopy is one ~9MB allocation per client per round. Even a perfect cache cannot skip the per-round `load_state_dict` — the teacher must equal *this* round's incoming global model, which changes every round. Meanwhile the real KD cost is the teacher **forward pass every batch** (`standard.py:87`), inherent to KD and untouched by any caching. The optimization targets the cheap part.

## Decision

**Keep the per-round `deepcopy`. Do not thread `context.state` into the loss hook. Do not cache the teacher across rounds.** The deepcopy is the minimum structure for a frozen teacher held separate from the training student, given a per-round client lifecycle.

## Consequences

- `client_kd_reg=true` runs as-is; no code change.
- Future architecture reviews should treat this seam as **resolved** and not re-propose teacher caching or a `LossHook` lifecycle widening. If a review believes the trade-off has changed, the burden is to refute facts 1–4 above (specifically: show cross-round persistence without per-worker VRAM accumulation, and a payoff that beats the per-batch teacher forward).
- If Flower's simulation model ever changes so clients persist across rounds (facts 1–2 no longer hold), this ADR may be revisited.
