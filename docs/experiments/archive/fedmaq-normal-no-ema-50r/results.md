# Experiment Results: FedMAQ Normal (ResNet18GN) 50-Round Sweeps (No EMA)

This experiment evaluates the performance of the full-sized **FedMAQ** model (ResNet18GN, ~11.17M params) trained for 50 rounds **without student model Exponential Moving Average (EMA)** (`algorithm.ema_student=false`).

The sweeps were run using the best-known regularization and hyperparameter configurations under both severe skew (Dirichlet $\alpha=0.1$) and moderate skew (Dirichlet $\alpha=1.0$) data partitions.

---

## 1. Experimental Configurations

- **Model:** ResNet18GN (~11.17M parameters)
- **Dataset:** CIFAR-10
- **Total Rounds:** 50 Rounds
- **EMA student:** `false` (EMA student tracking disabled)
- **Tuned Hyperparameters (Ground configs from Stacked Sweep):**
  - **Dirichlet $\alpha=0.1$:** `kd_reg_alpha=0.5`, `kd_reg_temp=1.0`, `entropy_weight=1.0`, `precision_weight=1.0`, `kd_prox_mu=0.1`
  - **Dirichlet $\alpha=1.0$:** `kd_reg_alpha=0.3`, `kd_reg_temp=2.0`, `entropy_weight=2.0`, `precision_weight=0.5`, `kd_prox_mu=0.1`

---

## 2. Centralized Test Performance

The tables below present the centralized test accuracy, loss, and cumulative uploaded communication footprint at both Round 40 and Round 50.

### Sweep A: Dirichlet $\alpha = 0.1$ (Severe Skew, 50 Rounds)

| Configuration      | Test Accuracy (R40) | Test Accuracy (R50) | Peak Test Accuracy | Test Loss (R40) | Test Loss (R50) | Uploaded Comm (R40) | Uploaded Comm (R50) |
| :----------------- | :-----------------: | :-----------------: | :----------------: | :-------------: | :-------------: | :-----------------: | :-----------------: |
| **No student EMA** |     **47.43%**      |     **45.55%**      |  **48.11%** (R45)  |     1.5595      |     1.4696      |     21360.5 MB      |     26845.6 MB      |

### Sweep B: Dirichlet $\alpha = 1.0$ (Moderate Skew, 50 Rounds)

| Configuration      | Test Accuracy (R40) | Test Accuracy (R50) | Peak Test Accuracy | Test Loss (R40) | Test Loss (R50) | Uploaded Comm (R40) | Uploaded Comm (R50) |
| :----------------- | :-----------------: | :-----------------: | :----------------: | :-------------: | :-------------: | :-----------------: | :-----------------: |
| **No student EMA** |     **60.69%**      |     **64.47%**      |  **65.42%** (R46)  |     1.1423      |     1.0364      |     21750.6 MB      |     27260.4 MB      |
