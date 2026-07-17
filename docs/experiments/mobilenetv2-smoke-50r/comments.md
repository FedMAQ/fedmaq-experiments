# Empirical Analysis: MobileNetV2GN Smoke Test Convergence & Stability

This document analyzes the physical training dynamics of the default **MobileNetV2GN** model under statistical heterogeneity, based on the 50-round smoke sweeps of FedMAQ and five baselines.

---

## 1. Key Findings & Physical Mechanisms

### A. Severe Skew ($\alpha = 0.1$): Quantization Stability & SVD Collapse

Under severe Non-IID skew, local updates on edge nodes diverge significantly due to non-overlapping label distributions.

- **FedProx Late-Round Instability**: In the early rounds, FedProx performs well, climbing to a peak test accuracy of **49.04%** by Round 45. However, in the final 5 rounds, it suffers from severe divergence, with test loss skyrocketing to **3.2988** and test accuracy collapsing to **24.79%**. This demonstrates that proximal L2 regularization at the canonical $\mu=0.01$ is insufficient to guarantee long-term convergence under extreme skew when utilizing depthwise-separable architectures like MobileNetV2.
- **Quantization Robustness (FedMAQ vs. DAdaQuant)**: Both FedMAQ and DAdaQuant exhibit excellent training stability under severe skew:
  - **FedMAQ** (with client KD regularization) peaks at **53.17%** (R37) and stabilizes to finish at **46.32%** (R50).
  - **DAdaQuant** peaks at **47.75%** (R43) and finishes at **46.76%** (R50).
  - This shows that adaptive-precision quantization methods are inherently robust under extreme skew, preventing the parameter-space divergence that plagued FedProx. FedMAQ's client-side KD regularization gives it a higher peak performance (**+5.42pp** over DAdaQuant's peak).
- **FedPAQ Degradation**: FedPAQ (fixed 8-bit quantization) lacks adaptive scaling, leading to a lower final accuracy of **36.69%** (R50).
- **SVD Capacity Collapse (FedKD)**: While FedKD achieves a highly compressed footprint of **36.2 MB**, SVD energy-based compression causes a severe degradation of representation capacity, with accuracy peaking at only **20.80%** and finishing at **17.09%**. **Root cause (audit F10, telemetry-confirmed):** `mean_rank_retained` is starved to **~3.7% of full rank** through the convergence window (rounds 2–~35), and accuracy rises only as rank recovers late. This is **two joint effects**, not just SVD lossiness: (1) the linear energy schedule (`tmin=0.1, tmax=0.95`) parks early rounds in a low-rank regime, and (2) energy→rank is non-monotonic with a ceiling of only ~16% rank even at 0.95 energy — so SVD truncation is punishingly aggressive for depthwise-separable weights. See [docs/audits/distillation-direction-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/distillation-direction-audit.md#f10).

### B. Moderate Skew ($\alpha = 1.0$): The Capacity-EMA Duality

Under moderate skew ($\alpha=1.0$), client updates are relatively aligned, making the global aggregation cleaner.

- **The Accuracy Gap**: FedMAQ (**60.93%**) and DAdaQuant (**64.65%**) lag behind the uncompressed baselines (FedAvg: **66.90%**, FedProx: **67.16%**) and the fixed 8-bit baseline (FedPAQ: **66.81%**).
- **Mechanism**: This gap confirms the **Capacity-EMA Duality** found in prior SimpleCNN/ResNet18GN explorations. Smaller/medium-sized architectures like MobileNetV2GN (~2.24M parameters) require the temporal smoothing of Student EMA (`ema_student=true`) under moderate skews to filter out quantization-induced variance and settle into sharper local minima. Disabling EMA (`ema_student=false`) in this sweep exposed this variance, preventing it from matching baseline performance under $\alpha=1.0$.

---

## 2. Recommendation for the Formal Grid

1. **EMA Configuration**: The formal experimental sweeps must document and sweep the Student EMA configuration, as it is a critical hyperparameter for MobileNetV2GN stability under moderate skews, whereas disabling it benefits extreme skews.
