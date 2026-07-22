# Architecture Deepening Plan

**Status**: Scoped via architecture audit + grilling, 2026-07-18. **Active.** No open questions — Candidate C resolved (grilled 2026-07-18 → `docs/adr/0001-client-kd-teacher-deepcopy-is-structural.md`).
**Origin**: Read-only deepening audit (`/improve-codebase-architecture`), 2026-07-18. Vocabulary: `/codebase-design`. Domain: `CONTEXT.md`.
**Scope decided**: Candidates B (telemetry) + A (client skeleton). Candidate C (loss-hook deepcopy) **closed** — structural, not a defect (ADR-0001). Candidate D (StrategyHook seam) **out** — speculative, highest risk, deferred past thesis.

> [!IMPORTANT]
> **Hard timing constraint — freeze gap only.** This plan executes in the window **after exploration finishes, before the confirm-phase freeze** — so all formal-grid runs share one code version. **Never interleave with a live campaign.** As of 2026-07-23, Pass 2 exploration is **paused (not complete)** at run 3/6 pending datacenter compute (local machine underpowered for the remaining sweeps) — no campaign is actively running, so the interleaving risk this gate exists to prevent is moot for now, and Step 1/2 work below may proceed. Re-check this note before resuming Pass 2/3: if a sweep is live again, stop code changes here until it finishes. See `DECISIONS.md` (explore/confirm freeze).

---

## Invalidation risk (why the gate exists)

Every candidate is *meant* to be behavior-preserving. Behavior-preserving in intent ≠ in fact. Candidates that touch the numeric FL path must be proven **bit-exact against golden outputs** before any formal run trusts them.

| Candidate | Touches | Risk | Gate |
| :-- | :-- | :-- | :-- |
| B — Telemetry seam | comms-bytes + wall-clock (both are `evaluation-metrics.md` metrics) | Low (pure relocation, identical formula) | Telemetry tests + spot-check one run's numbers unchanged |
| A — Client skeleton | client loss / optimizer / RNG draw order | **Real** | **Bit-exact golden diff** per baseline before trusting |
| C — LossHook deepcopy | KD teacher identity at loss-compute time | — | **Closed WONTFIX** — structural, see ADR-0001 |
| D — StrategyHook seam | every baseline server hook (aggregation/quant) | High | **Deferred past thesis** — not in this plan |

---

## Sequencing (safe → risky)

### Step 1 · Candidate B — Telemetry seam — **DONE (2026-07-23)**
- Relocated the client-metric-averaging + byte/delay-simulation block out of `strategy.aggregate_fit` into `TelemetryManager.record_fit_round()`; `aggregate_fit` now only dispatches to hooks and calls this one method. `evaluate()` reads `last_round_*` snapshots off `telemetry_manager` instead of `self`.
- Replaced the hardcoded per-algorithm CSV key list with `StrategyHook.metric_keys()`: each hook declares its own keys (FedMAQ, FedAvg+KD, FedKD, DAdaQuant — the last was previously missing from the schema entirely, a real gap this closed), registered via `TelemetryManager.register_hook_metric_keys()` in `TelemetryFedAvg.__init__`. The CSV header now only carries the *active* algorithm's columns instead of every baseline's.
- Validated via a pre/post-refactor `ci_test` matrix diff (fedavg + fedmaq, 2R): `communication/*` bytes and `system/cumulative_time_sec`/`client_time`/`server_time` (the deterministic simulated-time columns) are byte-identical. Only real `wall_time_sec` and model test-loss/accuracy (GPU/cuDNN non-determinism, untouched by this refactor) vary between runs, as expected.
- Added `tests/test_telemetry.py` covering the previously-untested gate: a hook-declared key reserves its column before it first appears, and an undeclared key is folded in once and never duplicates/reorders the header. Full suite green (111 passed).
- Restored `TelemetryFedAvg.simulated_time` as a read-only property delegating to `telemetry_manager.cumulative_time` — it looked like dead state (write-only in `src/`) but `tests/test_environment.py` reads it directly as a public signal in 4 tests.

### Step 2 · Candidate A — Client training skeleton (~2 Sonnet 5 sessions)
Highest value, gated behind golden validation. **This is the hot path of the sweep — freeze-gap only.**
- **S2a**: Build one deep `TrainingSkeleton` owning the SGD+CE+accuracy loop, delta→compress→reconstruct tail (`standard.py:118-130` ≡ `fedkd.py:139-151`), and the metrics dict (`standard.py:135`, `fedkd.py:164`, `feddistill.py:171`, `cfd.py:177`, `fedmd.py:192`). Seam ≈ "given a batch → return loss term" + "return upload payload." **Build the golden-output harness first.** Migrate `standard` + `fedkd`; bit-exact diff each.
- **S2b**: Migrate `feddistill`, `cfd`, `fedmd` (fedmd currently repeats the loop 3×). Bit-exact diff each baseline vs pre-refactor golden.
- **Hard gate**: any baseline whose golden diff is non-zero blocks the merge. Preserve RNG draw order and optimizer construction exactly.

### Step 3 · Candidate C — CLOSED (no work)
Grilled 2026-07-18. The "widen the `LossHook` contract" framing collapses into the already-rejected Ray/`context.state` cache: the hook is rebuilt every round (`simulation.py:163`), so it has no cross-round lifecycle to own, and the deepcopy is dominated by the inherent per-batch teacher forward pass. **Structural, not a defect** — recorded in `docs/adr/0001-client-kd-teacher-deepcopy-is-structural.md` so future audits stop re-suggesting it.

---

## Not in scope
- **Candidate D — StrategyHook seam narrowing.** Deepest finding, widest blast radius, thesis-imminent. Revisit post-thesis as its own decision.
- **fedmaq.py Tier1/Tier2 + formulation `if/elif`** — strongest deep module already; a dispatch table would only move branches (fails deletion test). Leave it.

---

## Session budget
- Scope (B + A): **~3 Sonnet 5 sessions**. (Candidate C closed → no session.) Step 1 done in 1 session as scoped; Step 2 (~2 sessions) remains.
- Golden-validation is baked into Step 2, not a separate session — but it is the merge gate.

## Resolution
When all steps land (or are recorded as ADRs), merge outcomes into `DECISIONS.md` and delete this plan file (`docs-management.md`: plans are active-only).
