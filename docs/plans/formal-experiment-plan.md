# FedMAQ Formal Experiment Plan

**Status**: Aligned via grilling session, 2026-07-16. Pre-confirmation (exploration phase in progress).
**Supersedes**: ad-hoc smoke-test progression. All prior ResNet18GN standings deprecated.

> [!IMPORTANT]
> Structure below is **tentative** and open to restructuring during planning. The *decisions* (§1) are settled; the *pipeline/organization* (§4) may still be refined.

---

## 1. Settled Decisions

See [docs/DECISIONS.md](../DECISIONS.md) (2026-07-16 entry) for the full list of 13 resolved decisions.

---

## 2. Mechanisms Under Deliberation (MobileNetV2GN)

The mechanisms below are **guides, not commitments** — they were tuned for SimpleCNN / ResNet18GN and may not all earn their place on MobileNetV2GN (deep but small, ~2.24M ≈ SimpleCNN param count). Exploration decides keep/drop/revise per mechanism.

| Mechanism | Prior status | MobileNetV2GN question |
| :-- | :-- | :-- |
| Dual-tier precision scaling (Formulation 3) | Robustly optimal (SimpleCNN/ResNet) | Still optimal at this capacity? |
| Soft-voting (entropy × precision weights) | Tuned per α | Re-sweep natively; entropy_weight transfer failed on ResNet18GN |
| Capacity-EMA duality | Helps small (SimpleCNN), hurts large (ResNet18GN) | **Open**: MobileNetV2 is small-but-deep — EMA on or off? |
| Grad-norm smoothing (β=0.7) | Keep (measurement noise, not dynamics) | Isolation ablation still owed |
| Client KD reg + proximal (μ=0.1) | Stacked reg best on ResNet18GN | Still needed at lower capacity? |

**Early signal (2026-07-16)**: FedMAQ first 40 rounds on MobileNetV2GN look promising — preliminary mechanism validation. FedAvg, FedProx smoke tests finished.

---

## 3. Deferred Sub-Details

Design-level open questions this plan still owes an answer to (the *what to decide*, not *what to do next* — for the task list, see [HANDOFF.md §5](../../HANDOFF.md), which is canonical for next-agent action items):

- **Single-config selection rule**: pick config maximizing mean accuracy over the sweep; ideally selected on a validation α distinct from the reported {0.1, 1.0} so even the "single config" isn't fit to the reported grid.
- **Baseline key-HP enumeration** for symmetric matched tuning: FedProx μ, FedPAQ bit-width, DAdaQuant schedule, FedMD/FedDistill/FedKD/CFD distillation temps. One key HP each, equal budget.
- **Pareto plot**: compare FedMAQ vs pure-quant baselines (FedPAQ, DAdaQuant) at **matched bit budgets**, else accuracy-vs-compression frontier is apples-to-oranges.
- **Capacity-EMA resolution**: exploration answers whether EMA is in the frozen config.

---

## 4. Execution Structure (tentative)

Two-phase, freeze-enforced — design summary only; concrete build steps (config-as-code registry, seed-determinism check) are tracked in [HANDOFF.md §5](../../HANDOFF.md).

- **Exploration phase** (now): adaptive, single-seed, mechanisms in flux. Output = one pre-registered frozen config (CIFAR-10) + baseline HP table + fixed mechanism set. Git-tag the pre-registration.
- **Confirmation phase**: config-as-code manifest → process-isolated runners → WandB (fixed project/group/tag scheme). Read-only configs after launch (hashed in manifest).

Open for restructuring: directory layout (`exploration/` vs `formal/`), manifest schema, WandB namespacing. To be planned before confirmation launch.

---

## 5. Metrics (per `evaluation-metrics.md`)

Top-1 accuracy; CE + distillation loss; macro P/R/F1; cumulative comm (MB/GB per-client + aggregate); wall-clock; convergence curves (acc vs rounds, acc vs transmitted bytes).
