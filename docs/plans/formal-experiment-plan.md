# FedMAQ Formal Experiment Plan

**Status**: Aligned via grilling session, 2026-07-16; exploration-campaign process scoped via grilling session, 2026-07-18 (see §3). Pre-confirmation (exploration phase in progress).
**Supersedes**: ad-hoc smoke-test progression. All prior ResNet18GN standings deprecated.

> [!IMPORTANT]
> Structure below is **tentative** and open to restructuring during planning. The *decisions* (§1) are settled; the *pipeline/organization* (§4) may still be refined.

---

## 1. Settled Decisions

See [docs/DECISIONS.md](../DECISIONS.md) — full resolved-decisions log (framing/grid, architecture refactor, baseline drops, exploration-campaign scoping).

---

## 2. Mechanisms Under Deliberation (MobileNetV2GN)

The mechanisms below are **guides, not commitments** — they were tuned for SimpleCNN / ResNet18GN and may not all earn their place on MobileNetV2GN (deep but small, ~2.24M ≈ SimpleCNN param count). Exploration decides keep/drop/revise per mechanism.

| Mechanism | Prior status | MobileNetV2GN question | Pass |
| :-- | :-- | :-- | :-- |
| Soft-voting (entropy × precision weights) | Tuned per α | Re-sweep natively; entropy_weight transfer failed on ResNet18GN | 1 — **done, provisional** (ew=2.0/pw=0.5/sv_on, pending multi-seed re-verification) |
| Capacity-EMA duality | Helps small (SimpleCNN), hurts large (ResNet18GN) | **Open**: MobileNetV2 is small-but-deep — EMA on or off? | 2 |
| Grad-norm smoothing (β=0.7) | Keep (measurement noise, not dynamics) | Isolation ablation still owed | 2 |
| Client KD reg + proximal (μ=0.1) | Stacked reg best on ResNet18GN | Still needed at lower capacity? | 2 |
| Dual-tier precision scaling (Formulation 3) | Robustly optimal (SimpleCNN/ResNet) | Still optimal at this capacity? | 3 |

Pass order + explore-α=0.3 + decision rule: `DECISIONS.md` 2026-07-18 entry (29–30). Pass 1 complete 2026-07-18 (`scripts/run_soft_voting_explore.py`, `multirun/2026-07-18/03-30-59-soft-voting-explore-mobilenetv2/`) — results + tentative pick: `DECISIONS.md` entries 33–35.

**Early signal (2026-07-16)**: FedMAQ first 40 rounds on MobileNetV2GN look promising — preliminary mechanism validation. FedAvg, FedProx smoke tests finished.

---

## 3. Deferred Sub-Details

Process questions (selection rule, sweep structure, decision rule, baseline-tuning budget) are resolved — see [DECISIONS.md](../DECISIONS.md) 2026-07-18 entry (27–32). Still open, pending Pass 1–3 results:

- **Capacity-EMA resolution**: Pass 2 answers whether EMA is in the frozen config.
- **Pareto plot**: compare FedMAQ vs pure-quant baselines (FedPAQ, DAdaQuant) at **matched bit budgets**, else accuracy-vs-compression frontier is apples-to-oranges. Not yet built.

---

## 4. Execution Structure (tentative)

Two-phase, freeze-enforced — design summary only; concrete build steps (config-as-code registry, seed-determinism check) are tracked in [HANDOFF.md §5](../../HANDOFF.md).

- **Exploration phase** (now): adaptive, single-seed, mechanisms in flux. Output = one pre-registered frozen config (CIFAR-10) + baseline HP table + fixed mechanism set. Git-tag the pre-registration.
- **Confirmation phase**: config-as-code manifest → process-isolated runners → WandB (fixed project/group/tag scheme). Read-only configs after launch (hashed in manifest).

Open for restructuring: directory layout (`exploration/` vs `formal/`), manifest schema, WandB namespacing. To be planned before confirmation launch.

---

## 5. Metrics (per `evaluation-metrics.md`)

Top-1 accuracy; CE + distillation loss; macro P/R/F1; cumulative comm (MB/GB per-client + aggregate); wall-clock; convergence curves (acc vs rounds, acc vs transmitted bytes).
