# Temperature Ablation Results (FedMAQ-Lite)

Empirical results comparing temperature values $T \in \{1.0, 2.0\}$ under the best soft-voting configurations for the FedMAQ-Lite variant (SimpleCNN, ~2.16M params).
Completed July 15, 2026.

All runs use:

- **Variant**: `fedmaq_lite` (SimpleCNN)
- **Formulation**: 3 (Gradient-Primary)
- **Grad Norm Smoothing**: `grad_norm_ema=true` (`β=0.7`)
- **Regime-Best EMA**: `ema_decay=0.7` for α=0.1, `ema_decay=0.1` for α=1.0
- **Regime-Best SV**:
  - α=0.1: `entropy_weight=4.0`, `precision_weight=1.0`
  - α=1.0: `entropy_weight=2.0`, `precision_weight=0.5`

Logs: [multirun/2026-07-15/temperature-ablation/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/multirun/2026-07-15/temperature-ablation/)

---

## Centralized Test Performance Comparison (at Round 40)

An apples-to-apples comparison with the Phase 2 grid sweep (which ran for 40 rounds):

|    α    | Temperature ($T$) | Final Acc (R40) | Peak Acc (R40) | Final Loss (R40) | Comm (MB) |
| :-----: | :---------------: | :-------------: | :------------: | :--------------: | :-------: |
| **0.1** | **1.0 (Best SV)** |   **52.83%**    |   **53.71%**   |    **1.3629**    |  3967.4   |
|   0.1   |        2.0        |     48.74%      |     49.67%     |      1.5686      |  4016.9   |
|         |                   |                 |                |                  |           |
| **1.0** | **1.0 (Best SV)** |   **63.28%**    |     63.28%     |    **1.0815**    |  4072.1   |
|   1.0   |        2.0        |     62.92%      |   **63.77%**   |      1.3201      |  4056.4   |

_Note: $T=1.0$ is the default used in all preceding sweeps. $T=2.0$ runs were executed in `multirun/2026-07-15/temperature-ablation/0` (α=0.1) and `1` (α=1.0)._

---

## Centralized Test Performance Comparison (at Round 50)

Both $T=2.0$ runs and Phase 1 ablation runs were allowed to run for 50 rounds:

|    α    |   Temperature ($T$)    | Final Acc (R50) | Peak Acc (R50) | Final Loss (R50) |
| :-----: | :--------------------: | :-------------: | :------------: | :--------------: |
| **0.1** | **1.0 (Run 0, SV on)** |   **50.20%**    |   **54.06%**   |    **1.4927**    |
|   0.1   |          2.0           |     43.63%      |     49.67%     |      1.9470      |
|         |                        |                 |                |                  |
| **1.0** | **1.0 (Run 2, SV on)** |     60.38%      |     61.28%     |    **1.1384**    |
|   1.0   |          2.0           |   **62.35%**    |   **63.77%**   |      1.3621      |

_Note: For α=1.0 at R50, the $T=2.0$ run converges to a higher final accuracy (62.35% vs 60.38% for Run 2), though the peak accuracy of the $T=1.0$ run was obtained earlier in the training trajectory. For α=0.1, $T=2.0$ lags significantly behind ($43.63\%$ vs $50.20\%$)._
