# FedMAQ Project Status

Single source of truth for current project state. Updated after each experiment batch.

**Last updated**: 2026-07-15

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

## 3. Best-Known Accuracy Standings

### FedMAQ-Lite (SimpleCNN) — Smoke Test Results

| Algorithm                | α=0.1 (Severe) | α=1.0 (Moderate) | Comm Footprint | Notes                                        |
| :----------------------- | :------------: | :--------------: | :------------: | :------------------------------------------- |
| **FedAvg** (ResNet18GN)  |     36.27%     |    **67.57%**    |   34100.2 MB   | Uncompressed baseline                        |
| **FedProx** (ResNet18GN) |     49.71%     |      67.01%      |   34100.2 MB   | Strongest baseline under α=0.1               |
| **FedMAQ-Lite** (Tuned)  |   **52.83%**   |      63.28%      | **3967.4 MB**  | **Beats FedProx by +3.12pp at 8.6x savings** |

### FedMAQ (ResNet18GN) — Smoke Test Results ⚠️ DEPRECATED

> [!WARNING]
> These results used ResNet18GN (~11.17M params), which has been replaced by MobileNetV2GN (~2.24M params). They are retained for historical reference only and must be re-run with MobileNetV2GN.

| Configuration                       | α=0.1 (Severe) |      α=1.0 (Moderate)       | Comm Footprint  | Notes                                  |
| :---------------------------------- | :------------: | :-------------------------: | :-------------: | :------------------------------------- |
| FedMAQ (unregularized baseline)     |     38.36%     |           53.04%            |   20195.2 MB    | SimpleCNN-tuned params                 |
| + Stacked Reg (KD + FedProx, μ=0.1) |     41.21%     |         **65.80%**          |   20195.2 MB    | Best with EMA enabled                  |
| + Stacked Reg, **No EMA**           |   **47.43%**   | 60.69% (R40) / 64.47% (R50) | ~25602 MB (R50) | Best at α=0.1; late-round dip at α=1.0 |
| **FedProx** (target to beat)        |   **49.71%**   |           67.01%            |   34100.2 MB    | Gap: −2.28pp at α=0.1                  |
| **FedAvg** (target to beat)         |     36.27%     |         **67.57%**          |   34100.2 MB    | Gap: −3.10pp at α=1.0                  |

**ResNet18GN communication savings**: ~1.7x (vs. 8.6x for FedMAQ-Lite). See §5 for implications.

---

## 4. Best-Known Configurations

### FedMAQ-Lite (Tuned — SimpleCNN)

| Hyperparameter     | α=0.1 | α=1.0 |
| :----------------- | :---: | :---: |
| `formulation`      |   3   |   3   |
| `ema_student`      | true  | true  |
| `ema_decay`        |  0.7  |  0.1  |
| `soft_voting`      | true  | true  |
| `entropy_weight`   |  4.0  |  2.0  |
| `precision_weight` |  1.0  |  0.5  |
| `temperature`      |  1.0  |  1.0  |
| `grad_norm_ema`    | true  | true  |
| `grad_norm_beta`   |  0.7  |  0.7  |

### FedMAQ (Best-Known — ResNet18GN)

| Hyperparameter     |   α=0.1   |   α=1.0   | Notes                                        |
| :----------------- | :-------: | :-------: | :------------------------------------------- |
| `formulation`      |     3     |     3     |                                              |
| `ema_student`      | **false** | **false** | Capacity-EMA duality: EMA hurts large models |
| `client_kd_reg`    |   true    |   true    |                                              |
| `kd_reg_alpha`     |    0.5    |    0.3    |                                              |
| `kd_reg_temp`      |    1.0    |    2.0    |                                              |
| `kd_prox_mu`       |    0.1    |    0.1    |                                              |
| `soft_voting`      |   true    |   true    |                                              |
| `entropy_weight`   |    1.0    |    2.0    | Lowered from 4.0 to fix voter exclusion      |
| `precision_weight` |    1.0    |    0.5    | ⚠️ Transferred from SimpleCNN, not re-tuned  |
| `temperature`      |    1.0    |    1.0    | ⚠️ Transferred from SimpleCNN, not re-tuned  |
| `grad_norm_ema`    |   true    |   true    |                                              |
| `grad_norm_beta`   |    0.7    |    0.7    |                                              |

---

## 5. Critical Decisions — RESOLVED (2026-07-16)

All framing/methodology decisions were resolved in a grilling session on 2026-07-16. Full list + rationale: **[docs/DECISIONS.md](DECISIONS.md)**. Grid design detail: [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md).

---

## 6. Key Novel Findings (Smoke Tests)

These findings should be prominently featured in the thesis, pending formal validation:

1. **Capacity-EMA Duality**: Student EMA helps small models (SimpleCNN) but _hurts_ large models (ResNet18GN) under severe skew. Explained by interaction between model capacity and local regularization. (Experiment 9)

2. **Heterogeneity-EMA Inverse Relationship**: Optimal EMA decay is strongly heterogeneity-dependent (β=0.7 for α=0.1, β=0.1 for α=1.0). (Experiment 3)

3. **Stacked Dual-Space Regularization**: Combining logit-space KD and parameter-space L2 regularization outperforms either alone on high-capacity models. (Experiment 8)

4. **Formulation 3 Dominance**: Gradient-primary, data-modulated scaling is robustly optimal across heterogeneity regimes. (Experiment 2)

---

## 7. What Remains

See [HANDOFF.md §5](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md) for the current next-agent action list.

---

## 8. Reference Links

| Document                                                                                                                                      | Purpose                                                      |
| :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ |
| [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)                                           | Resolved decisions log (single source of truth)             |
| [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md)                                                         | Next-agent instructions and immediate action items          |
| [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)                         | Chronological experiment registry with per-experiment links |
| [docs/audits/fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md)                       | Full algorithm audit with line-level code references        |
| [docs/audits/fedmaq-audit-recos.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit-recos.md)           | Actionable audit recommendations with priority table        |
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                         | Canonical glossary (resolves naming drift between repos)    |
