# FedMAQ Project Status

Single source of truth for current project state. Updated after each experiment batch.

**Last updated**: 2026-07-16

---

## 1. Important Context

> [!IMPORTANT]
> **All experiments conducted so far are exploratory smoke tests** — short-round sweeps (40–50R) on single seeds to validate the algorithm direction and identify which hyperparameters matter. They are **not** the formal thesis results. The formal experiment grid (multi-seed, multi-α, 100+ rounds) has not been executed.

> [!WARNING]
> **Model architecture switched to MobileNetV2GN.** As of 2026-07-15, the default CIFAR model has been changed from ResNet18GN (~11.17M params) to MobileNetV2GN (~2.24M params) for **edge realism** (deployable ~2.24M model on Pi/Jetson tiers). Note: this does **not** improve the compression *ratio* — at iso-architecture the ratio (~1.7×) is set by bit-width allocation, not param count (see §5, Decision 1). All prior ResNet18GN smoke test results (§3) are **deprecated** and must be re-run with MobileNetV2GN. ResNet18GN remains available via `model_name="resnet18gn"` config override. A full hyperparameter sweep on MobileNetV2GN is required before formal experiments.

---

## 2. Algorithm Variants

FedMAQ has been formally partitioned into two variants:

| Variant                         | Client Model  | Params | Status                                          | Primary Use Case                                    |
| :------------------------------ | :------------ | :----: | :---------------------------------------------- | :-------------------------------------------------- |
| **FedMAQ** (`fedmaq`)           | MobileNetV2GN | ~2.24M | Active development — needs MobileNetV2GN tuning | Iso-architecture baseline comparison (edge model)   |
| **FedMAQ-Lite** (`fedmaq_lite`) | SimpleCNN     | ~2.16M | Smoke tests complete — tuned                    | Demonstrates even small models beat large baselines |

---

## 3. Best-Known Accuracy Standings (ResNet18GN era — deprecated)

> [!WARNING]
> All standings and configs from the ResNet18GN/SimpleCNN era have moved to [docs/experiments/archive/RESNET18GN-SUMMARY.md](experiments/archive/RESNET18GN-SUMMARY.md) — retained for historical reference only, since the default CIFAR model is now MobileNetV2GN (see §5, Decision 1). No MobileNetV2GN standings exist yet; this section will be repopulated once formal runs land.

---

## 4. Critical Decisions — RESOLVED (2026-07-16)

All framing/methodology decisions were resolved in a grilling session on 2026-07-16. Full list + rationale: **[docs/DECISIONS.md](DECISIONS.md)**. Grid design detail: [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md).

---

## 5. Key Novel Findings (Smoke Tests)

Full list with rationale: [docs/experiments/archive/RESNET18GN-SUMMARY.md](experiments/archive/RESNET18GN-SUMMARY.md) §"Key Novel Findings". Flagged for MobileNetV2GN re-validation per [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md) §2.

---

## 6. What Remains

See [HANDOFF.md §5](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md) for the current next-agent action list.

---

## 7. Reference Links

| Document                                                                                                                                      | Purpose                                                      |
| :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ |
| [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)                                           | Resolved decisions log (single source of truth)             |
| [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md)                                                         | Next-agent instructions and immediate action items          |
| [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)                         | Chronological experiment registry with per-experiment links |
| [docs/audits/fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md)                       | Full algorithm audit with line-level code references        |
| [docs/audits/fedmaq-audit-recos.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit-recos.md)           | Actionable audit recommendations with priority table        |
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                         | Canonical glossary (resolves naming drift between repos)    |
