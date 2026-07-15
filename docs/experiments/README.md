# FedMAQ Experiment Registry & Roadmap

This directory houses all experimental sweeps, ablation studies, and hyperparameter tuning results conducted for the FedMAQ thesis.

Every experiment is self-contained within its own directory and adheres to a strict organization standard:

- `results.md`: Detailed tabular data (accuracy, loss, communication footprint, simulated latency) and Hydra configuration paths.
- `comments.md`: In-depth empirical analysis, physical mechanisms, and Master's thesis narrative alignment.

---

## 1. Chronological Experiment Roadmap

Below is the historical sequence of experiments leading to the final optimized **FedMAQ-Lite** formulation:

| Order | Experiment Directory                                                                                                                           | Date (2026) | Description & Key Objective                                                                                                        | Major Outcome / Finding                                                                                                                                                 |
| :---: | :--------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :--------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | [`smoke-test-7-13/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/smoke-test-7-13/)                           |   July 13   | 40-round initial benchmark baseline sweep comparing original FedMAQ against 7 other algorithms (FedAvg, FedProx, FedPAQ, etc.).    | Baseline established. High non-IID skew ($\alpha=0.1$) identified as the primary area of focus due to a -10.95pp gap to FedProx.                                        |
| **2** | [`pilot-formulation-study-7-14/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/pilot-formulation-study-7-14/) |   July 14   | Evaluated 5 alternative formulations (0-4) for Tier-2 precision scaling.                                                           | **Formulation 3** (Gradient-Primary, Data-Modulated) identified as the winner. Revealed that noisy distillation was the primary bottleneck.                             |
| **3** | [`ema-decay-sweep-7-14/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/ema-decay-sweep-7-14/)                 |   July 14   | Swept Student Model EMA decay ($\beta \in [0.1, 0.9]$) at 50 rounds to stabilize late-round convergence.                           | Validated EMA: **$\beta=0.7$** is optimal for severe skew ($\alpha=0.1$, +13.75pp boost), while **$\beta=0.1$** is optimal for moderate skew ($\alpha=1.0$).            |
| **4** | [`soft-voting-sweep-7-14/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/soft-voting-sweep-7-14/)             |   July 14   | 32-run grid sweep over soft-voting parameters ($\gamma_{entropy}, \gamma_{precision} \in \{0.5, 1.0, 2.0, 4.0\}^2$).               | Confirmed complementary spatial filtering: **`ew=4.0, pw=1.0`** is best for $\alpha=0.1$, while **`ew=2.0, pw=0.5`** is best for $\alpha=1.0$.                          |
| **5** | [`temperature-ablation/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/temperature-ablation/)                 |   July 15   | Evaluated distillation softmax temperature $T \in \{1.0, 2.0\}$ under the best EMA and soft-voting configs.                        | Confirmed **$T=1.0$** as the optimal default for quantized edge distillation to prevent logit noise amplification.                                                      |
| **6** | [`baseline-comparison-resnet18/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/baseline-comparison-resnet18/) |   July 15   | Baseline ResNet18GN comparison sweeps (40 rounds) under default parameters.                                                        | FedMAQ achieved 38.36% (α=0.1), outperforming FedAvg/FedDistill. α=1.0 was cancelled due to ema_decay mismatch (0.7 used instead of 0.1).                               |
| **7** | [`client-kd-reg-sweep-7-15/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/client-kd-reg-sweep-7-15/)         |   July 15   | Swept client-side KD regularization parameters (alpha & temp) to mitigate client drift on ResNet18GN under both alpha values.      | Validated client regularization: kd_reg alpha=0.5, temp=2.0 under α=1.0 gets **63.65%** (+6.16pp over baseline); alpha=0.5, temp=1.0 under α=0.1 gets **33.98%**.       |
| **8** | [`stacked-reg-sweep-7-15/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/stacked-reg-sweep-7-15/)             |   July 15   | Swept stacked FedProx parameter-space penalty ($\mu \in \{0.0, 0.001, 0.01, 0.1, 1.0\}$) on top of best client-side KD parameters. | Stacked regularization is highly successful: **$\mu=0.1$** is optimal for both skews, yielding **41.21%** (α=0.1) and **65.80%** (α=1.0), stabilizing late convergence. |
| **9** | [`fedmaq-normal-no-ema-50r/`](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/fedmaq-normal-no-ema-50r/)         |   July 15   | Evaluated the performance of the full-sized FedMAQ (ResNet18GN) without student EMA for 50 rounds.                                 | Disabling EMA yields a massive **+6.22pp** boost under severe skew ($\alpha=0.1$, reaching **47.43%**), but causes a slight **-5.11pp** dip under moderate skew.        |

---

## 2. Accuracy Progression (FedMAQ-Lite vs. Baselines)

The table below traces the performance improvement of FedMAQ-Lite (SimpleCNN, ~2.16M params) across our optimization phases:

| Step / Phase                                              | $\alpha=0.1$ (Severe Skew) | Gap to FedProx (49.71%) | $\alpha=1.0$ (Moderate Skew) | Gap to FedAvg (67.57%) |
| :-------------------------------------------------------- | :------------------------: | :---------------------: | :--------------------------: | :--------------------: |
| **Original FedMAQ** (Smoke Test)                          |           38.76%           |        −10.95pp         |            58.07%            |        −9.50pp         |
| **+ Formulation 3** (Pilot)                               |          ~42.00%           |         −7.71pp         |           ~55.00%            |        −12.57pp        |
| **+ Robustness Bundle** (Soft-voting, EMA, Gradient Norm) |           52.51%           |         +2.80pp         |            61.07%            |        −6.50pp         |
| **+ EMA Tuning**                                          |           52.51%           |         +2.80pp         |            61.86%            |        −5.71pp         |
| **+ Soft-voting Tuning** (Tuned Lite)                     |         **52.83%**         |       **+3.12pp**       |          **63.28%**          |      **−4.29pp**       |

_Note: With full optimization, FedMAQ-Lite now **outperforms the FedProx baseline** by **+3.12pp** under severe skew while maintaining an **8.6x communication reduction**._

---

## 3. Best-Known FedMAQ-Lite Configurations

These parameters define the final, optimized **FedMAQ-Lite** recipe:

| Hyperparameter     | $\alpha=0.1$ (Severe Skew) | $\alpha=1.0$ (Moderate Skew) | Purpose / Mechanism                                                     |
| :----------------- | :------------------------: | :--------------------------: | :---------------------------------------------------------------------- |
| `formulation`      |           **3**            |            **3**             | Gradient-primary, data-modulated soft-quality target                    |
| `ema_student`      |           `true`           |            `true`            | Enables Exponential Moving Average student model                        |
| `ema_decay`        |          **0.7**           |           **0.1**            | Tracks temporal parameters (higher decay smooths high non-IID variance) |
| `soft_voting`      |           `true`           |            `true`            | Enables spatial multi-teacher logit confidence filtering                |
| `entropy_weight`   |          **4.0**           |           **2.0**            | Suppresses uncertain, high-entropy predictions                          |
| `precision_weight` |          **1.0**           |           **0.5**            | Discounts predictions from low-precision/quantized edge nodes           |
| `temperature`      |          **1.0**           |           **1.0**            | Standard distillation scaling (higher T amplifies quantization noise)   |
| `grad_norm_ema`    |           `true`           |            `true`            | Server-side per-client gradient norm smoothing                          |
| `grad_norm_beta`   |           `0.7`            |            `0.7`             | Smoothing momentum factor                                               |

---

## 4. ResNet18GN Client-Side Regularization Performance

To address client drift on the high-capacity ResNet18GN model, we evaluated client-side logit-space KD regularization. Below is a comparison of the target regularized configurations against the unregularized baselines and other models:

| Model / Configuration                    | Dirichlet $\alpha=0.1$ (Severe Skew) | Dirichlet $\alpha=1.0$ (Moderate Skew) | Comm Footprint ($\alpha=0.1$) | Params |
| :--------------------------------------- | :----------------------------------: | :------------------------------------: | :---------------------------: | :----: |
| **FedAvg** (ResNet18GN)                  |                36.27%                |                 67.57%                 |          34100.2 MB           | 11.17M |
| **FedProx** (ResNet18GN)                 |              **49.71%**              |                 67.01%                 |          34100.2 MB           | 11.17M |
| **FedDistill** (ResNet18GN)              |                32.94%                |                 66.86%                 |          34100.5 MB           | 11.17M |
| **FedMAQ** (ResNet18GN Baseline)         |                38.36%                |                 53.04%                 |          20195.2 MB           | 11.17M |
| **FedMAQ + Client KD Reg** (Target)      |    33.98% (`alpha_0.5_temp_1.0`)     |     63.65% (`alpha_0.5_temp_2.0`)      |          20195.2 MB           | 11.17M |
| **FedMAQ + Client KD Reg** (Best Sweep)  |  35.06% (`alpha_0.3_temp_2.0` peak)  |   **65.94%** (`alpha_0.3_temp_2.0`)    |          20195.2 MB           | 11.17M |
| **FedMAQ + Stacked Reg** (KD + Proximal) |     **41.21%** (Best: `mu=0.1`)      |      **65.80%** (Best: `mu=0.1`)       |          20195.2 MB           | 11.17M |
| **FedMAQ + Stacked Reg (No EMA)**        | **47.43%** (R40) / **45.55%** (R50)  |    60.69% (R40) / **64.47%** (R50)     |       25601.9 MB (R50)        | 11.17M |
| **FedMAQ-Lite** (Tuned SimpleCNN)        |              **52.83%**              |                 63.28%                 |         **3967.4 MB**         | 2.16M  |

## 5. Run execution & Context

- **Process-Isolated Execution**: All experiments use process-isolated runner scripts under the `scripts/` directory to prevent CUDA Out-of-Memory (OOM) leaks from sequential multi-runs.
- **Hardware Grounding**: Simulates edge client memory sizes matching **Raspberry Pi variants (2GB/4GB/8GB)** and **Jetson Edge Nodes (16GB)** capping quantization bit-widths (4/8/16/32-bit).
