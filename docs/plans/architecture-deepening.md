# Architecture Deepening Plan

**Status**: Scoped via architecture audit + grilling, 2026-07-18. **Active.** No open questions — Candidate C resolved (grilled 2026-07-18 → `docs/adr/0001-client-kd-teacher-deepcopy-is-structural.md`).
**Origin**: Read-only deepening audit (`/improve-codebase-architecture`), 2026-07-18. Vocabulary: `/codebase-design`. Domain: `CONTEXT.md`.
**Scope decided**: Candidates B (telemetry) + A (client skeleton). Candidate C (loss-hook deepcopy) **closed** — structural, not a defect (ADR-0001). Candidate D (StrategyHook seam) **out** — speculative, highest risk, deferred past thesis.

> [!IMPORTANT]
> **Hard timing constraint — freeze gap only.** This plan executes in the window **after exploration finishes, before the confirm-phase freeze** — so all formal-grid runs share one code version. **Never interleave with a live campaign.** Pass 1 exploration is running as of this writing; do not start any code change here until the exploration campaign is complete. See `DECISIONS.md` (explore/confirm freeze).

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

### Step 1 · Candidate B — Telemetry seam (~1 Sonnet 5 session)
Isolated from FL math, do first.
- Relocate byte/wall-clock **measurement** out of `strategy.py:223-337` (the ~115L inline block in `aggregate_fit`) behind the `TelemetryManager` interface, which already owns accumulation + emission.
- Replace the hardcoded 40-key CSV schema (`telemetry.py:158-197`) with a per-baseline **metric registry**: each hook declares the keys it emits; telemetry composes the schema.
- **Formula must stay identical** — byte-count and wall-clock math unchanged, only relocated. Verify: one run's `communication/*` and wall-clock numbers match pre-refactor.
- `aggregate_fit` shrinks back to hook dispatch only.
- Gate: telemetry tests green + assert unknown-key → stable CSV header (currently untested, `extrasaction="ignore"`).

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
- Scope (B + A): **~3 Sonnet 5 sessions**. (Candidate C closed → no session.)
- Golden-validation is baked into Step 2, not a separate session — but it is the merge gate.

## Resolution
When all steps land (or are recorded as ADRs), merge outcomes into `DECISIONS.md` and delete this plan file (`docs-management.md`: plans are active-only).
