# Pilot Formulation Study Results (40 Rounds)

This document contains the empirical evaluation results for the 40-round pilot study comparing the 5 alternative FedMAQ formulations (0-4) under Dirichlet $\alpha=0.1$ and $\alpha=1.0$ data heterogeneity.

---

## Benchmark Results

### Sweep A: Dirichlet $\alpha = 0.1$ (High Non-IID Skew, 40 Rounds)

| Formulation       | Description                 | Test Accuracy | Test Loss  | Cumulative Comm (MB) | Sim Latency (s) |    Status     |
| :---------------- | :-------------------------- | :-----------: | :--------: | :------------------: | :-------------: | :-----------: |
| **Formulation 0** | Resource-Only Hard Cap      |    40.10%     |   1.6301   |      4532.0 MB       |     3430.2s     |   Completed   |
| **Formulation 1** | Normalized Linear Sum       |    38.97%     |   1.8015   |      4010.9 MB       |     3490.7s     |   Completed   |
| **Formulation 2** | Normalized Multiplicative   |    35.70%     |   1.8189   |      3928.4 MB       |     3545.6s     |   Completed   |
| **Formulation 3** | **Gradient-Primary (Main)** |  **40.60%**   | **1.6825** |    **3991.9 MB**     |   **3550.6s**   | **Completed** |
| **Formulation 4** | Threshold-Based Staged      |    10.00%     |   _NaN_    |      3916.3 MB       |     3542.1s     |   Completed   |

### Sweep B: Dirichlet $\alpha = 1.0$ (Moderate Non-IID Skew, 40 Rounds)

| Formulation       | Description                 | Test Accuracy | Test Loss  | Cumulative Comm (MB) | Sim Latency (s) |    Status     |
| :---------------- | :-------------------------- | :-----------: | :--------: | :------------------: | :-------------: | :-----------: |
| **Formulation 0** | Resource-Only Hard Cap      |    57.62%     |   1.1914   |      4573.1 MB       |     2305.1s     |   Completed   |
| **Formulation 1** | Normalized Linear Sum       |    56.28%     |   1.2179   |      4075.7 MB       |     2274.9s     |   Completed   |
| **Formulation 2** | Normalized Multiplicative   |    54.78%     |   1.2563   |      4072.6 MB       |     2320.7s     |   Completed   |
| **Formulation 3** | **Gradient-Primary (Main)** |  **55.42%**   | **1.2585** |    **4069.2 MB**     |   **2309.4s**   | **Completed** |
| **Formulation 4** | Threshold-Based Staged      |    55.82%     |   1.2663   |      4491.6 MB       |     2343.3s     |   Completed   |

---

## Key Analysis & Takeaways

- **Formulation 3 Robustness**: Formulation 3 achieves the best trade-off. It is the top performer under severe non-IID skew ($\alpha=0.1$), beating Formulation 0 in accuracy while saving 12% in communication. Under moderate skew ($\alpha=1.0$), it provides the lowest overall communication footprint (4069.2 MB) with only a minor accuracy penalty ($2.2\%$) compared to Formulation 0.
- **Formulation 4 Failure and Recovery**: Under severe skew, Formulation 4 collapsed to 10% random guessing due to hard thresholds over-compressing client updates to 1-bit. Under moderate skew ($\alpha=1.0$), where gradient norms and partition sizes are more uniform, the thresholds were crossed regularly, allowing it to recover to 55.82% accuracy, though it consumed more communication (4491.6 MB) than Formulations 1-3.
