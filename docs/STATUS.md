# FedMAQ Project Status

Single source of truth for current project state. Updated after each experiment batch.

**Last updated**: 2026-07-15

---

## 1. Important Context

> [!IMPORTANT]
> **All experiments conducted so far are exploratory smoke tests** — short-round sweeps (40–50R) on single seeds to validate the algorithm direction and identify which hyperparameters matter. They are **not** the formal thesis results. The formal experiment grid (multi-seed, multi-α, 100+ rounds) has not been executed.

> [!WARNING]
> **ResNet18GN hyperparameters are not natively tuned.** The current ResNet18GN configs were transferred from SimpleCNN (FedMAQ-Lite) smoke tests. The `entropy_weight=4.0` transfer failure (Experiment 7, where regularized accuracy _dropped_ to 33.98% vs. 38.36% baseline) demonstrates that SimpleCNN-tuned values may be silently suboptimal for ResNet18GN. A full re-sweep on ResNet18GN is required before formal experiments.

---

## 2. Algorithm Variants

FedMAQ has been formally partitioned into two variants:

| Variant                         | Client Model | Params  | Status                               | Primary Use Case                                    |
| :------------------------------ | :----------- | :-----: | :----------------------------------- | :-------------------------------------------------- |
| **FedMAQ** (`fedmaq`)           | ResNet18GN   | ~11.17M | Active development — needs re-tuning | Iso-architecture baseline comparison                |
| **FedMAQ-Lite** (`fedmaq_lite`) | SimpleCNN    | ~2.16M  | Smoke tests complete — tuned         | Demonstrates even small models beat large baselines |

---

## 3. Best-Known Accuracy Standings

### FedMAQ-Lite (SimpleCNN) — Smoke Test Results

| Algorithm                | α=0.1 (Severe) | α=1.0 (Moderate) | Comm Footprint | Notes                                        |
| :----------------------- | :------------: | :--------------: | :------------: | :------------------------------------------- |
| **FedAvg** (ResNet18GN)  |     36.27%     |    **67.57%**    |   34100.2 MB   | Uncompressed baseline                        |
| **FedProx** (ResNet18GN) |     49.71%     |      67.01%      |   34100.2 MB   | Strongest baseline under α=0.1               |
| **FedMAQ-Lite** (Tuned)  |   **52.83%**   |      63.28%      | **3967.4 MB**  | **Beats FedProx by +3.12pp at 8.6x savings** |

### FedMAQ (ResNet18GN) — Smoke Test Results

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

## 5. Critical Open Decisions

> [!CAUTION]
>
> ### Decision 1: Thesis Contribution Framing
>
> The ResNet18GN communication savings are only **1.7x** (vs. 8.6x for FedMAQ-Lite). This weakens the "communication-efficient FL" headline claim under iso-architecture comparison. Two options:
>
> - **(a) Stick with communication efficiency**: May require a different compression approach (e.g., deeper quantization, structured pruning, or a fundamentally different KD-based method beyond parameter averaging).
> - **(b) Frame contribution as the mechanism**: Primary contribution becomes the dual-tier precision scaling + soft-voting + capacity-EMA duality _framework_, with communication savings as a secondary benefit.
>
> **Current leaning**: Option (b) — contribution is the mechanism, not the magnitude.
>
> **Action**: Must be resolved with the next agent before formal experiments begin.

### Decision 2: ResNet18GN Hyperparameter Re-Tuning Scope

Current ResNet18GN configs are partially transferred from SimpleCNN. The following need native re-sweeping:

- [ ] Soft-voting grid (`entropy_weight` × `precision_weight`) — known transfer failure
- [ ] EMA decay values (if EMA is re-enabled for any regime)
- [ ] Formulation study (whether Formulation 3 is still optimal for higher capacity)
- [ ] Client KD reg temperature and alpha (partially done, but with contaminated `entropy_weight`)

### Decision 3: Experiment Organization

The next agent should help plan a clean experiment structure. Current state is messy:

- Experiments were run iteratively, each building on the previous
- Script output paths point to `experiments/` but data lives in `multirun/` after manual transfer
- No unified execution plan for the formal experiment grid

---

## 6. Key Novel Findings (Smoke Tests)

These findings should be prominently featured in the thesis, pending formal validation:

1. **Capacity-EMA Duality**: Student EMA helps small models (SimpleCNN) but _hurts_ large models (ResNet18GN) under severe skew. Explained by interaction between model capacity and local regularization. (Experiment 9)

2. **Heterogeneity-EMA Inverse Relationship**: Optimal EMA decay is strongly heterogeneity-dependent (β=0.7 for α=0.1, β=0.1 for α=1.0). (Experiment 3)

3. **Stacked Dual-Space Regularization**: Combining logit-space KD and parameter-space L2 regularization outperforms either alone on high-capacity models. (Experiment 8)

4. **Formulation 3 Dominance**: Gradient-primary, data-modulated scaling is robustly optimal across heterogeneity regimes. (Experiment 2)

---

## 7. What Remains Before Formal Experiments

### Pre-Experiment (Next Agent)

- [ ] Resolve thesis contribution framing (§5, Decision 1)
- [ ] Re-sweep ResNet18GN hyperparameters natively
- [ ] Plan the formal experiment grid (α values, seeds, rounds, datasets, ablations)
- [ ] Validate FedMAQ formulation on ResNet18GN (may differ from SimpleCNN)
- [ ] Design clean experiment organization and execution pipeline
- [ ] Investigate late-round accuracy degradation on ResNet18GN (R45→R50 drop under α=0.1)
- [ ] Add gradient-norm-smoothing isolation ablation

### Formal Experiment Grid (Planned)

- 100+ rounds
- α ∈ {0.1, 0.3, 0.5, 1.0} (intermediate values for Pareto curve)
- 3+ seeds for statistical significance (mean ± std)
- Proper ablation table isolating each feature's contribution
- Possibly FEMNIST as a second dataset

---

## 8. Reference Links

| Document                                                                                                                                      | Purpose                                                     |
| :-------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md)                                                         | Next-agent instructions and immediate action items          |
| [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)                         | Chronological experiment registry with per-experiment links |
| [docs/audits/fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md)                       | Full algorithm audit with line-level code references        |
| [docs/audits/fedmaq-audit-recos.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit-recos.md)           | Actionable audit recommendations with priority table        |
| [docs/plans/fedmaq-audit-remediation.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/fedmaq-audit-remediation.md) | Phase-by-phase remediation plan                             |
| [docs/plans/client-regularization.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/client-regularization.md)       | Client-side KD regularization design & sweep plan           |
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                         | Canonical glossary (resolves naming drift between repos)    |
