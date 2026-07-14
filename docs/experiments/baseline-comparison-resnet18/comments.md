# ResNet18 Baseline Comparison: Empirical Analysis & Next Steps

Detailed analysis of the full-sized **FedMAQ** (ResNet18GN) baseline sweeps completed on July 15, 2026. Cross-referenced with [results.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/baseline-comparison-resnet18/results.md).

---

## 1. Empirical Findings

### 1.1 Severe Heterogeneity (α=0.1) — Outperforming Baseline Quantization/Distillation

Under severe statistical skew ($\alpha=0.1$):

- **FedMAQ** achieves **38.36%** test accuracy at Round 40.
- This outperformance beats the uncompressed baseline **FedAvg (36.27%)** by **+2.09pp** and the uncompressed baseline **FedDistill (32.94%)** by **+5.42pp**.
- However, it still lags behind **FedProx (49.71%)** by **−11.35pp**.

### 1.2 The Local Overfitting & Client Drift Bottleneck

- A key trend in the logs is the extreme gap between client training accuracy and global test accuracy. For instance, in Round 40, `client/avg_train_acc` reached **83.96%**, but the global `test/accuracy` was only **38.36%**.
- Because ResNet18GN is over 5x larger than SimpleCNN, local clients quickly overfit to their highly skewed local shards (sometimes only containing 10–50 samples). This causes client weights and representation spaces to diverge significantly, neutralizing the benefits of standard aggregation and making distillation harder to train.

---

## 2. Parameter Correction & Rerun Plan

### 2.1 The α=1.0 Sweep Cancellation

- The Dirichlet $\alpha=1.0$ sweep was run using `ema_decay=0.7`, which is the optimal value for severe skew ($\alpha=0.1$). However, prior tuning sweeps on SimpleCNN established that **`ema_decay=0.1`** is optimal for moderate statistical skew ($\alpha=1.0$).
- High EMA smoothing values ($\beta=0.7$) under moderate skew over-dampen the student updates when clients already have relatively clean and balanced local distributions, resulting in slower convergence (reaching only 53.03% at Round 34).
- **Next Step:** Rerun the $\alpha=1.0$ ResNet18GN sweep with `algorithm.ema_decay=0.1`.

---

## 3. Regularization Strategy

To address the local overfitting and client drift bottlenecks on larger models like ResNet18GN, we need client-side stabilization:

- **Client-Side KD Regularization:** We will implement local KL-divergence matching during client epochs (leveraging global student logits as local targets) as outlined in [client-regularization.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/client-regularization.md), referencing the survey by **Salman et al. (2025)**.
- **Parameter-Space Regularization:** We will also evaluate enabling parameter-space regularization (FedProx's proximal term) in combination with FedMAQ to damp extreme quantization/heterogeneity noise.
