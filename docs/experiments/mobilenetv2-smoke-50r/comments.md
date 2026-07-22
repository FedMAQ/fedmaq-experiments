# Empirical Analysis: MobileNetV2GN Smoke Test Convergence & Stability

This document analyzes the physical training dynamics of the default **MobileNetV2GN** model under statistical heterogeneity, based on the 50-round smoke sweeps of FedMAQ and five baselines.

---

## Key Findings & Physical Mechanisms

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

### C. KD-Family Baselines (F13 closure, 2026-07-17): FedDistill, CFD, FedAvg+KD

Three previously-unmeasured KD baselines completed their first MobileNetV2GN runs this pass, closing the F13 evidence gap.

- **FedDistill lands mid-pack, cleanly.** 39.03% (α=0.1) / 56.96% (α=1.0) final accuracy — behind the quantization baselines and FedMAQ, ahead of FedKD and collapsed CFD. No stability issues at either skew. Ready as a comparison baseline.
- **FedAvg+KD is heterogeneity-sensitive, not broken.** Weak under severe skew (17.28% at α=0.1, trailing every other baseline) but recovers to a sane 51.42% under moderate skew — a similar shape to FedMAQ's own α=0.1/α=1.0 asymmetry, consistent with KD regularization costing accuracy when data is easy and needing more rounds/tuning when it's hard.
- **CFD collapses to chance at both α — a genuine defect, not a tuning artifact.** Test accuracy is pinned at ~10.0% for all 50 rounds under both severe and moderate skew, with test loss _above_ chance-level cross-entropy (2.65–3.79 vs. ln(10)≈2.30). Per-round telemetry shows the failure originates in the client-side soft-vote aggregation (`pre_aggregate_fit` in `strategy_hooks/cfd.py`) — the `targets_acc` metric (soft-voted client predictions on the public set, _before_ server training) is chance-level from round 1 onward, even though individual clients' local training accuracy is healthy (50–65% throughout, per `client/avg_train_acc`). The server model is faithfully distilling noise for all 50 rounds; this is not an undertrained-server or bad-hyperparameter issue. This empirically confirms a static-review concern flagged earlier (memory obs #695, "CFD soft-label codec one-hot quantization bootstrap vulnerability") that had not previously been runtime-verified. **CFD's numbers here are not a valid baseline entry** — see `docs/audits/distillation-direction-audit.md` F15 for the full diagnosis and next steps.

---

## Recommendation for the Formal Grid

1. **EMA Configuration**: The formal experimental sweeps must document and sweep the Student EMA configuration, as it is a critical hyperparameter for MobileNetV2GN stability under moderate skews, whereas disabling it benefits extreme skews.
2. **CFD is blocked**: do not include CFD in the formal grid until F15 (soft-vote aggregation collapse) is fixed. FedDistill and FedAvg+KD are ready to enter the formal comparison set as-is.
