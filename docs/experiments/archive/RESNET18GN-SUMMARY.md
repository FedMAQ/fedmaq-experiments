# ResNet18GN Era — Consolidated Accuracy Standings (Deprecated)

> [!WARNING]
> **Deprecated 2026-07-16.** All experiments below used ResNet18GN (~11.17M params) as the default CIFAR model, since replaced by MobileNetV2GN (~2.24M params) for edge realism (see [docs/DECISIONS.md](../../DECISIONS.md), Decision 1). Retained for historical reference only. Raw per-experiment data lives in the sibling `archive/<name>/results.md` files; this document is the cross-experiment rollup that previously lived in `STATUS.md` §3-4.

---

## FedMAQ-Lite (SimpleCNN) — Smoke Test Results

| Algorithm                | α=0.1 (Severe) | α=1.0 (Moderate) | Comm Footprint | Notes                                        |
| :------------------------ | :------------: | :---------------: | :-------------: | :-------------------------------------------- |
| **FedAvg** (ResNet18GN)  |     36.27%     |    **67.57%**    |   34100.2 MB   | Uncompressed baseline                        |
| **FedProx** (ResNet18GN) |     49.71%     |      67.01%      |   34100.2 MB   | Strongest baseline under α=0.1               |
| **FedMAQ-Lite** (Tuned)  |   **52.83%**   |      63.28%      | **3967.4 MB**  | **Beats FedProx by +3.12pp at 8.6x savings** |

Source: `archive/smoke-test-7-13/results.md`, `archive/soft-voting-sweep-7-14/results.md`.

## FedMAQ (ResNet18GN) — Smoke Test Results

| Configuration                       | α=0.1 (Severe) |      α=1.0 (Moderate)       | Comm Footprint  | Notes                                  |
| :------------------------------------ | :------------: | :---------------------------: | :---------------: | :---------------------------------------- |
| FedMAQ (unregularized baseline)     |     38.36%     |           53.04%            |   20195.2 MB    | SimpleCNN-tuned params                 |
| + Stacked Reg (KD + FedProx, μ=0.1) |     41.21%     |         **65.80%**          |   20195.2 MB    | Best with EMA enabled                  |
| + Stacked Reg, **No EMA**           |   **47.43%**   | 60.69% (R40) / 64.47% (R50) | ~25602 MB (R50) | Best at α=0.1; late-round dip at α=1.0 |
| **FedProx** (target to beat)        |   **49.71%**   |           67.01%            |   34100.2 MB    | Gap: −2.28pp at α=0.1                  |
| **FedAvg** (target to beat)         |     36.27%     |         **67.57%**          |   34100.2 MB    | Gap: −3.10pp at α=1.0                  |

**ResNet18GN communication savings**: ~1.7x (vs. 8.6x for FedMAQ-Lite) — see [docs/DECISIONS.md](../../DECISIONS.md) Decision 2 for why this comparison is confounded by model size, not just quantization.

Source: `archive/stacked-reg-sweep-7-15/results.md`, `archive/client-kd-reg-sweep-7-15/results.md`, `archive/fedmaq-normal-no-ema-50r/results.md`, `archive/baseline-comparison-resnet18/results.md`.

## Best-Known Configurations (ResNet18GN / SimpleCNN era)

### FedMAQ-Lite (Tuned — SimpleCNN)

| Hyperparameter     | α=0.1 | α=1.0 |
| :------------------ | :----: | :----: |
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
| :------------------ | :--------: | :--------: | :--------------------------------------------- |
| `formulation`      |     3     |     3     |                                               |
| `ema_student`      | **false** | **false** | Capacity-EMA duality: EMA hurts large models |
| `client_kd_reg`    |   true    |   true    |                                               |
| `kd_reg_alpha`     |    0.5    |    0.3    |                                               |
| `kd_reg_temp`      |    1.0    |    2.0    |                                               |
| `kd_prox_mu`       |    0.1    |    0.1    |                                               |
| `soft_voting`      |   true    |   true    |                                               |
| `entropy_weight`   |    1.0    |    2.0    | Lowered from 4.0 to fix voter exclusion      |
| `precision_weight` |    1.0    |    0.5    | ⚠️ Transferred from SimpleCNN, not re-tuned  |
| `temperature`      |    1.0    |    1.0    | ⚠️ Transferred from SimpleCNN, not re-tuned  |
| `grad_norm_ema`    |   true    |   true    |                                               |
| `grad_norm_beta`   |    0.7    |    0.7    |                                               |

## Key Novel Findings (from this era, pending MobileNetV2GN re-validation)

1. **Capacity-EMA Duality**: Student EMA helps small models (SimpleCNN) but _hurts_ large models (ResNet18GN) under severe skew. (Experiment 9)
2. **Heterogeneity-EMA Inverse Relationship**: Optimal EMA decay is strongly heterogeneity-dependent (β=0.7 for α=0.1, β=0.1 for α=1.0). (Experiment 3)
3. **Stacked Dual-Space Regularization**: Combining logit-space KD and parameter-space L2 regularization outperforms either alone on high-capacity models. (Experiment 8)
4. **Formulation 3 Dominance**: Gradient-primary, data-modulated scaling is robustly optimal across heterogeneity regimes. (Experiment 2)

See [docs/plans/formal-experiment-plan.md](../../plans/formal-experiment-plan.md) §2 for which of these mechanisms are flagged for re-validation on MobileNetV2GN.
