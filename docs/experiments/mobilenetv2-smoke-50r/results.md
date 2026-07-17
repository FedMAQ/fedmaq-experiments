# Experiment Results: MobileNetV2GN Smoke Test Sweeps (50 Rounds)

This experiment evaluates the baseline, FedProx, and FedMAQ performance, along with additional compression baselines (**DAdaQuant**, **FedPAQ**, and **FedKD**), on the default **MobileNetV2GN** model (~2.24M params) trained for 50 rounds.

The sweeps compare performance under severe skew (Dirichlet $\alpha=0.1$) and moderate skew (Dirichlet $\alpha=1.0$) data partitions. FedMAQ was run using default parameters with student EMA disabled (`algorithm.ema_student=false`) and client KD regularization enabled (`algorithm.client_kd_reg=true`, `kd_prox_mu=0.0`).

---

## 1. Experimental Configurations

- **Model:** MobileNetV2GN (~2.24M parameters)
- **Dataset:** CIFAR-10
- **Total Rounds:** 50 Rounds
- **Run Configurations:**
  - **FedAvg**: Default configuration (32-bit parameters)
  - **FedProx**: canonical configuration (32-bit parameters, $\mu=0.01$)
  - **FedMAQ**: `ema_student=false`, `client_kd_reg=true`, `kd_prox_mu=0.0` (no EMA, client KD regularization only)
  - **DAdaQuant**: Default configuration (adaptive quantization)
  - **FedPAQ**: Default configuration (fixed 8-bit quantization)
  - **FedKD**: Default configuration (SVD SVD-energy compression)

---

## 2. Centralized Test Performance

The tables below present the centralized test accuracy, loss, and cumulative uploaded communication footprint at both Round 40 and Round 50.

### Sweep A: Dirichlet $\alpha = 0.1$ (Severe Skew, 50 Rounds)

| Configuration | Test Accuracy (R40) | Test Accuracy (R50) | Peak Test Accuracy | Test Loss (R40) | Test Loss (R50) | Uploaded Comm (R40) | Uploaded Comm (R50) |
| :------------ | :-----------------: | :-----------------: | :----------------: | :-------------: | :-------------: | :-----------------: | :-----------------: |
| **FedAvg**    |       40.54%        |       41.37%        |    42.29% (R47)    |     2.2132      |     2.2025      |      6825.8 MB      |      8532.3 MB      |
| **FedProx**   |       47.96%        |       24.79%        |    49.04% (R45)    |     1.7058      |     3.2988      |      6825.8 MB      |      8532.3 MB      |
| **FedMAQ**    |     **47.06%**      |     **46.32%**      |  **53.17%** (R37)  |   **1.4594**    |   **1.4414**    |    **4142.4 MB**    |    **5190.1 MB**    |
| **DAdaQuant** |       42.06%        |       46.76%        |    47.75% (R43)    |     1.7857      |     1.5844      |      3746.2 MB      |      4698.6 MB      |
| **FedPAQ**    |       33.92%        |       36.69%        |    40.33% (R44)    |     1.8747      |     1.6095      |      4266.4 MB      |      5333.0 MB      |
| **FedKD**     |       24.04%        |       26.41%        |    30.10% (R45)    |     2.0847      |     2.3732      |      589.5 MB       |      737.6 MB       |

> **FedKD note (F10 re-confirmation, 2026-07-17):** row above supersedes the retired pre-fix figure (20.80% peak) — that number was measured on the old SimpleCNN student and is not comparable to this run (`min_rank_frac=0.25`, width-0.5 MobileNetV2GN student per DECISIONS #22). The starvation *mechanism* is confirmed fixed on the current architecture (`mean_rank_retained` holds at 0.278 vs. a same-model A/B minitest's 3.7% unfixed baseline — see the audit), but the run stays volatile (±10pp swings) and trails every other baseline by 15–27pp. See `docs/audits/distillation-direction-audit.md` F10 for the full verdict.

### Sweep B: Dirichlet $\alpha = 1.0$ (Moderate Skew, 50 Rounds)

| Configuration | Test Accuracy (R40) | Test Accuracy (R50) | Peak Test Accuracy | Test Loss (R40) | Test Loss (R50) | Uploaded Comm (R40) | Uploaded Comm (R50) |
| :------------ | :-----------------: | :-----------------: | :----------------: | :-------------: | :-------------: | :-----------------: | :-----------------: |
| **FedAvg**    |     **64.38%**      |     **66.90%**      |  **66.90%** (R50)  |     1.2283      |     1.2062      |      6825.8 MB      |      8532.3 MB      |
| **FedProx**   |     **65.56%**      |     **67.16%**      |  **67.18%** (R48)  |     1.1853      |     1.2006      |      6825.8 MB      |      8532.3 MB      |
| **FedMAQ**    |       59.17%        |       60.93%        |    60.93% (R50)    |   **1.1899**    |   **1.1617**    |    **4217.3 MB**    |    **5272.4 MB**    |
| **DAdaQuant** |       63.16%        |       64.65%        |    65.13% (R49)    |     1.2566      |     1.2237      |      3626.5 MB      |      4533.1 MB      |
| **FedPAQ**    |       65.49%        |       66.81%        |    67.01% (R44)    |     1.2280      |     1.2166      |      4266.4 MB      |      5333.0 MB      |
| **FedKD**     |       34.70%        |       36.29%        |    38.31% (R49)    |     1.7591      |     1.7551      |      589.5 MB       |      739.9 MB       |

> **FedKD note (F10 re-confirmation, 2026-07-17):** post-fix rerun (see Sweep A note above) — noticeably more stable than the α=0.1 run, with a near-monotonic climb, but still trails all other baselines.
