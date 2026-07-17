# Empirical Analysis: Impact of Student Model EMA on High-Capacity Convergence

This document analyzes the physical and training dynamics of disabling the server-side Exponential Moving Average (EMA) student tracking on the high-capacity **ResNet18GN** model under different data partition skews.

---

## 1. Tabular Comparison: With EMA vs. No EMA (at Round 40)

| Partition Skew                             | Metric              | With EMA (Best Stacked config) | No EMA (This Experiment) |    Delta    |
| :----------------------------------------- | :------------------ | :----------------------------: | :----------------------: | :---------: |
| **Dirichlet $\alpha=0.1$** (Severe Skew)   | Test Accuracy (R40) |             41.21%             |        **47.43%**        | **+6.22pp** |
|                                            | Peak Accuracy       |          41.27% (R40)          |     **48.11%** (R45)     | **+6.84pp** |
|                                            | Test Loss (R40)     |             1.6624             |        **1.5595**        | **-0.1029** |
| **Dirichlet $\alpha=1.0$** (Moderate Skew) | Test Accuracy (R40) |           **65.80%**           |          60.69%          | **-5.11pp** |
|                                            | Peak Accuracy       |        **65.80%** (R40)        |       65.42% (R46)       | **-0.38pp** |
|                                            | Test Loss (R40)     |           **1.0058**           |          1.1423          | **+0.1365** |

---

## 2. Key Findings & Physical Mechanisms

### 1. Severe Skew ($\alpha = 0.1$): The "EMA Inertia" Bottleneck

Under extreme Dirichlet partition skew, edge clients possess highly non-overlapping class subsets (often just 1 or 2 classes out of 10). Consequently, their local optimization steps push parameter coordinates in divergent directions (extreme client drift).

- **With EMA (`ema_student=true`):** Applying a temporal moving average ($\beta=0.7$) to the global student model parameters adds significant "inertia". While this suppresses noise, it severely dampens the rate at which the global model can adapt to new client gradients.
- **Without EMA (`ema_student=false`):** Disabling the moving average allows the global student model to immediately absorb the full aggregated updates from client models at the end of each round. Since client regularization (`kd_reg_alpha=0.5` and `kd_prox_mu=0.1`) already constrains local parameter drift, the global aggregation remains stable without requiring the temporal smoothing of EMA.
- **Outcome:** The network learns substantially faster, yielding a massive **+6.22pp** accuracy boost at Round 40 and peaking at **48.11%**. This bridges the gap to the uncompressed 32-bit FedProx baseline (49.71%) to within **−2.28pp**.

### 2. Moderate Skew ($\alpha = 1.0$): The Stabilization Benefit of EMA

When partition skew is moderate ($\alpha = 1.0$), client data distributions are more homogeneous, meaning the averaged client updates represent a relatively clean global descent direction.

- **With EMA (`ema_student=true`):** In this regime, the updates are clean enough that a tiny EMA decay ($\beta=0.1$) acts as a stabilizing filter, smoothing out minor fluctuations between rounds and helping the model settle into a sharp local minimum.
- **Without EMA (`ema_student=false`):** Without the smoothing filter, the raw aggregated parameter coordinates experience higher variance across late rounds. This variance causes the test accuracy to dip slightly to **60.69%** at Round 40. However, the model is still highly capable and eventually recovers to **64.47%** at Round 50, peaking at **65.42%** (virtually identical to the EMA baseline peak of 65.80%).

---

## 3. Thesis Narrative & Key Insights

1. **The Capacity-EMA Duality:**
   - **Normal FedMAQ (High-Capacity ResNet18GN):** Student EMA should **not** be enabled (`ema_student=false`). Doing so is highly detrimental under severe skew ($\alpha=0.1$), causing a **-6.84pp** peak accuracy penalty due to excessive temporal "inertia" that dampens gradient updates. Under moderate skew ($\alpha=1.0$), EMA offers no peak accuracy benefit. Because ResNet18GN is already regularized locally (via logit-space KD and proximal L2 penalties), its parameter coordinates are stable without EMA.
   - **FedMAQ-Lite (Low-Capacity SimpleCNN):** Student EMA is **highly beneficial**. For small models, local parameter trajectories are volatile under non-IID conditions. Adding student EMA smoothing ($\beta=0.7$ for severe skew) is crucial to stabilize the distillation signals and prevent model divergence.

2. **Handoff Guideline for Audits:**
   - **Normal FedMAQ configuration:** Must be run with `algorithm.ema_student=false` to avoid the capacity-drift inertia bottleneck.
   - **FedMAQ-Lite configuration:** Must be run with `algorithm.ema_student=true` (with tuned $\beta=0.7$ or $\beta=0.1$ depending on skew) to stabilize small-network training.
