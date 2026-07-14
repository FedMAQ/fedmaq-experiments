# EMA Decay Sweep Results (50 Rounds)

This document contains the empirical evaluation results for the 50-round sweep over EMA decay values (`0.1, 0.2, ..., 0.9`) under Dirichlet $\alpha=0.1$ and $\alpha=1.0$ data heterogeneity.

---

## 1. Sweep Results at R=50

### Sweep A: Dirichlet $\alpha = 0.1$ (High Non-IID Skew)

| `ema_decay` | Final Accuracy | Final Test Loss | Cumulative Comm. |
| :---------: | :------------: | :-------------: | :--------------: |
|     0.1     |     40.01%     |     1.6047      |    5020.1 MB     |
|     0.2     |     48.44%     |     1.5057      |    5020.9 MB     |
|     0.3     |     49.46%     |     1.4959      |    5004.2 MB     |
|     0.4     |     47.81%     |     1.4945      |    5035.6 MB     |
|     0.5     |     49.42%     |     1.5073      |    4977.0 MB     |
|     0.6     |     47.25%     |     1.4954      |    5023.7 MB     |
|   **0.7**   |   **52.51%**   |   **1.4513**    |  **5007.6 MB**   |
|     0.8     |     46.08%     |     1.5936      |    5018.1 MB     |
|     0.9     |     48.15%     |     1.5784      |    5015.8 MB     |

### Sweep B: Dirichlet $\alpha = 1.0$ (Moderate Non-IID Skew)

| `ema_decay` | Final Accuracy | Final Test Loss | Cumulative Comm. |
| :---------: | :------------: | :-------------: | :--------------: |
|     0.1     |     60.62%     |     1.1440      |    5093.7 MB     |
|     0.2     |     59.89%     |     1.1573      |    5077.0 MB     |
|     0.3     |     60.32%     |     1.1295      |    5090.6 MB     |
|     0.4     |     60.55%     |     1.1233      |    5080.6 MB     |
|   **0.5**   |   **60.68%**   |   **1.1307**    |  **5082.9 MB**   |
|     0.6     |     59.32%     |     1.1542      |    5091.2 MB     |
|     0.7     |     58.80%     |     1.1601      |    5080.5 MB     |
|     0.8     |     58.90%     |     1.1578      |    5082.3 MB     |
|     0.9     |     57.15%     |     1.2001      |    5079.8 MB     |

---

## 2. Baseline Comparison at R=40

To compare directly with baseline algorithms that were evaluated on a 40-round budget, we query the test accuracies of the EMA sweep configurations at exactly **Round 40**.

### Dirichlet $\alpha = 0.1$ (High Non-IID Skew)

| Algorithm / Config           | Accuracy at R40 | Top Accuracy |
| :--------------------------- | :-------------: | :----------: |
| **FedMAQ (Ours, EMA = 0.7)** |   **52.51%**    |  **53.90%**  |
| **FedProx**                  |     49.71%      |    49.71%    |
| **FedMAQ (Ours, No EMA)**    |     38.76%      |    47.50%    |
| **FedAvg**                   |     36.27%      |    46.18%    |
| **FedPAQ**                   |     37.47%      |    43.91%    |
| **DAdaQuant**                |     41.49%      |    43.29%    |
| **FedDistill**               |     32.94%      |    39.19%    |
| **FedKD**                    |     14.09%      |    20.75%    |
| **CFD**                      |     10.00%      |    16.15%    |

### Dirichlet $\alpha = 1.0$ (Moderate Non-IID Skew)

| Algorithm / Config           | Accuracy at R40 | Top Accuracy |
| :--------------------------- | :-------------: | :----------: |
| **FedAvg**                   |   **67.57%**    |    67.57%    |
| **FedProx**                  |     67.53%      |    67.62%    |
| **FedDistill**               |     67.37%      |    67.37%    |
| **FedPAQ**                   |     66.91%      |    67.07%    |
| **DAdaQuant**                |     65.91%      |    67.06%    |
| **FedMAQ (Ours, EMA = 0.1)** |   **61.86%**    |  **61.93%**  |
| **FedMAQ (Ours, EMA = 0.3)** |     61.42%      |    62.52%    |
| **FedMAQ (Ours, No EMA)**    |     61.07%      |    62.52%    |
| **FedMAQ (Ours, EMA = 0.7)** |     59.32%      |    59.97%    |
| **FedKD**                    |     29.17%      |    29.63%    |
| **CFD**                      |     30.23%      |    31.25%    |
