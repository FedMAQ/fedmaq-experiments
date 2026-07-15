# Soft-Voting Ablation & Hyperparameter Sweep Results

Empirical results for the two-phase soft-voting sweep run on July 14, 2026.
All runs use Formulation 3 (Gradient-Primary), `grad_norm_ema=true` (`β=0.7`), and the regime-best EMA setting from the prior EMA decay sweep: **`ema_decay=0.7` for α=0.1** and **`ema_decay=0.1` for α=1.0**.

Logs: [multirun/2026-07-14/19-13-43-soft-voting-sweep/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/multirun/2026-07-14/19-13-43-soft-voting-sweep/)

---

## Phase 1 — Soft-Voting Ablation (50 Rounds)

Isolates the contribution of soft-voting by toggling it on/off at best EMA settings. `entropy_weight=1.0`, `precision_weight=1.0` when enabled.

| Config |  α  | `soft_voting` | Final Acc (R50) |  Peak Acc  | Final Loss | Comm (MB) |
| :----- | :-: | :-----------: | :-------------: | :--------: | :--------: | :-------: |
| Run 0  | 0.1 |     ✅ on     |   **50.20%**    | **54.06%** |   1.4927   |  5005.5   |
| Run 1  | 0.1 |    ❌ off     |     47.72%      |   50.55%   |   1.6068   |  5008.1   |
| Run 2  | 1.0 |     ✅ on     |   **60.38%**    | **61.28%** |   1.1384   |  5096.2   |
| Run 3  | 1.0 |    ❌ off     |     56.42%      |   60.29%   |   1.2410   |  5089.3   |

**Soft-voting gain:** +2.48pp (α=0.1), +3.96pp (α=1.0) at R50.

---

## Phase 2 — Hyperparameter Grid Sweep (40 Rounds)

Swept `entropy_weight` × `precision_weight` ∈ {0.5, 1.0, 2.0, 4.0}² at best EMA settings per regime.

### α = 0.1 (Severe Non-IID) — Runs 4–19, ema_decay=0.7

| `entropy_weight` | `precision_weight` | Final Acc (R40) |  Peak Acc  | Final Loss | Comm (MB)  |
| :--------------: | :----------------: | :-------------: | :--------: | :--------: | :--------: |
|       0.5        |        0.5         |     50.43%      |   51.39%   |   1.5631   |   4011.1   |
|       0.5        |        1.0         |     48.68%      |   52.66%   |   1.6145   |   4004.4   |
|       0.5        |        2.0         |     44.11%      |   50.00%   |   1.6133   |   4004.4   |
|       0.5        |        4.0         |     46.27%      |   54.03%   |   1.5658   |   4014.5   |
|       1.0        |        0.5         |     46.45%      |   51.94%   |   1.5934   |   4001.1   |
|       1.0        |        1.0         |     51.38%      |   53.44%   |   1.5135   |   4036.6   |
|       1.0        |        2.0         |   **52.02%**    |   52.28%   |   1.4721   |   3989.8   |
|       1.0        |        4.0         |     50.26%      |   52.91%   |   1.5372   |   4009.3   |
|       2.0        |        0.5         |     50.21%      |   55.12%   |   1.4476   |   4009.1   |
|       2.0        |        1.0         |     52.08%      |   54.75%   |   1.4184   |   3994.4   |
|       2.0        |        2.0         |     49.34%      |   52.75%   |   1.4388   |   4006.0   |
|     **2.0**      |      **4.0**       |   **52.45%**    | **53.48%** | **1.4390** |   3983.4   |
|       4.0        |        0.5         |     50.58%      |   54.55%   |   1.4036   |   4019.9   |
|     **4.0**      |      **1.0**       |   **52.83%**    | **53.71%** | **1.3629** | **3967.4** |
|       4.0        |        2.0         |     49.03%      |   54.14%   |   1.4210   |   4004.2   |
|       4.0        |        4.0         |     46.31%      |   53.09%   |   1.4332   |   3977.7   |

**Best at α=0.1:** `entropy_weight=4.0, precision_weight=1.0` → 52.83% final, 53.71% peak

### α = 1.0 (Moderate Non-IID) — Runs 20–35, ema_decay=0.1

| `entropy_weight` | `precision_weight` | Final Acc (R40) |  Peak Acc  | Final Loss | Comm (MB)  |
| :--------------: | :----------------: | :-------------: | :--------: | :--------: | :--------: |
|       0.5        |        0.5         |     60.19%      |   60.98%   |   1.1255   |   4061.8   |
|       0.5        |        1.0         |     56.58%      |   61.82%   |   1.2312   |   4065.6   |
|       0.5        |        2.0         |     59.30%      |   60.26%   |   1.1722   |   4066.1   |
|       0.5        |        4.0         |     60.12%      |   60.56%   |   1.1440   |   4065.4   |
|       1.0        |        0.5         |     58.19%      |   61.30%   |   1.2131   |   4064.9   |
|       1.0        |        1.0         |     59.54%      |   62.19%   |   1.1656   |   4068.2   |
|     **1.0**      |      **2.0**       |   **61.59%**    | **61.59%** | **1.1212** | **4074.9** |
|       1.0        |        4.0         |     58.54%      |   61.52%   |   1.1778   |   4083.9   |
|     **2.0**      |      **0.5**       |   **63.28%**    | **63.28%** | **1.0815** | **4072.1** |
|       2.0        |        1.0         |     61.62%      |   62.48%   |   1.1261   |   4073.3   |
|       2.0        |        2.0         |     61.33%      |   62.41%   |   1.1175   |   4060.2   |
|       2.0        |        4.0         |     61.29%      |   62.86%   |   1.1736   |   4056.4   |
|       4.0        |        0.5         |     62.48%      |   63.26%   |   1.1629   |   4073.6   |
|       4.0        |        1.0         |     61.23%      |   62.74%   |   1.1845   |   4060.0   |
|       4.0        |        2.0         |     61.47%      |   63.22%   |   1.1504   |   4058.4   |
|       4.0        |        4.0         |     62.48%      |   62.49%   |   1.1311   |   4074.4   |

**Best at α=1.0:** `entropy_weight=2.0, precision_weight=0.5` → 63.28% final, 63.28% peak

---

## Summary: Optimal Soft-Voting Parameters by Regime

| Regime           | Best `entropy_weight` | Best `precision_weight` | Final Acc (R40) |  Peak Acc  |
| :--------------- | :-------------------: | :---------------------: | :-------------: | :--------: |
| α=0.1 (Severe)   |        **4.0**        |         **1.0**         |   **52.83%**    | **53.71%** |
| α=1.0 (Moderate) |        **2.0**        |         **0.5**         |   **63.28%**    | **63.28%** |

---

## Comparison Against Prior Baselines at R40

### α = 0.1 (Severe Non-IID)

| Algorithm / Config                    | Accuracy @ R40 | Top Accuracy |
| :------------------------------------ | :------------: | :----------: |
| **FedMAQ (Tuned SV: ew=4.0, pw=1.0)** |   **52.83%**   |  **53.71%**  |
| FedMAQ (EMA=0.7, default SV 1.0/1.0)  |     52.51%     |    53.90%    |
| FedProx                               |     49.71%     |    49.71%    |
| FedMAQ (No EMA, No SV tuning)         |     38.76%     |    47.50%    |
| FedAvg                                |     36.27%     |    46.18%    |

### α = 1.0 (Moderate Non-IID)

| Algorithm / Config                    | Accuracy @ R40 | Top Accuracy |
| :------------------------------------ | :------------: | :----------: |
| FedAvg                                |   **67.57%**   |    67.57%    |
| FedProx                               |     67.53%     |    67.62%    |
| FedDistill                            |     67.37%     |    67.37%    |
| FedPAQ                                |     66.91%     |    67.07%    |
| DAdaQuant                             |     65.91%     |    67.06%    |
| **FedMAQ (Tuned SV: ew=2.0, pw=0.5)** |   **63.28%**   |  **63.28%**  |
| FedMAQ (EMA=0.1, default SV 1.0/1.0)  |     61.86%     |    61.93%    |
| FedMAQ (No EMA)                       |     61.07%     |    62.52%    |
