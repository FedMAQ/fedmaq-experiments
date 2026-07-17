# Experiment Results: Client-Side KD Regularization Sweep (ResNet18GN)

This sweep evaluates **Client-Side Knowledge Distillation Regularization** (FedGKD-style) on the full-sized **FedMAQ** (ResNet18GN, ~11.17M params) under both Dirichlet $\alpha=0.1$ and $\alpha=1.0$ data partitions.

---

## 1. Centralized Test Performance (at Round 40)

Below is the accuracy, loss, and communication footprint of all configurations run in the sweep.

### Sweep A: Dirichlet $\alpha = 0.1$ (Severe Skew, 40 Rounds)

_SimpleCNN-tuned overrides used: `ema_decay=0.7`, `entropy_weight=4.0`, `precision_weight=1.0`_

| Configuration                                 | Test Accuracy (R40) | Peak Test Accuracy | Test Loss (R40) | Uploaded Comm (MB) | Comm Savings vs FedAvg |
| :-------------------------------------------- | :-----------------: | :----------------: | :-------------: | :----------------: | :--------------------: |
| `baseline_no_reg`                             |       29.58%        |       31.74%       |     1.8609      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.1_temp_1.0`                   |       28.39%        |       28.39%       |     1.7935      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.1_temp_2.0`                   |       31.72%        |       31.72%       |     1.8201      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.3_temp_1.0`                   |       30.81%        |       34.61%       |     1.7535      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.3_temp_2.0`                   |       29.32%        |     **35.06%**     |     1.7678      |     20195.2 MB     |      1.7x saving       |
| **`kd_reg_alpha_0.5_temp_1.0` (User Target)** |     **33.98%**      |     **33.98%**     |   **1.7217**    |   **20195.2 MB**   |    **1.7x saving**     |
| `kd_reg_alpha_0.5_temp_2.0`                   |       27.00%        |       27.00%       |     1.7848      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.7_temp_1.0`                   |       32.06%        |       32.06%       |     1.8344      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.7_temp_2.0`                   |       29.79%        |       29.79%       |     1.8307      |     20195.2 MB     |      1.7x saving       |

### Sweep B: Dirichlet $\alpha = 1.0$ (Moderate Skew, 40 Rounds)

_SimpleCNN-tuned overrides used: `ema_decay=0.1`, `entropy_weight=2.0`, `precision_weight=0.5`_

| Configuration                                 | Test Accuracy (R40) | Peak Test Accuracy | Test Loss (R40) | Uploaded Comm (MB) | Comm Savings vs FedAvg |
| :-------------------------------------------- | :-----------------: | :----------------: | :-------------: | :----------------: | :--------------------: |
| `baseline_no_reg`                             |       57.49%        |       60.85%       |     1.2836      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.1_temp_1.0`                   |       62.01%        |       63.13%       |     1.1300      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.1_temp_2.0`                   |       63.35%        |       64.65%       |     1.0913      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.3_temp_1.0`                   |       64.40%        |       64.40%       |     1.0078      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.3_temp_2.0`                   |     **65.94%**      |     **65.94%**     |   **0.9841**    |     20195.2 MB     |      1.7x saving       |
| **`kd_reg_alpha_0.5_temp_2.0` (User Target)** |     **63.65%**      |     **63.87%**     |   **1.0636**    |   **20195.2 MB**   |    **1.7x saving**     |
| `kd_reg_alpha_0.5_temp_1.0`                   |       58.35%        |       61.59%       |     1.2029      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.7_temp_1.0`                   |       60.46%        |       60.46%       |     1.1307      |     20195.2 MB     |      1.7x saving       |
| `kd_reg_alpha_0.7_temp_2.0`                   |       63.38%        |       63.38%       |     1.1022      |     20195.2 MB     |      1.7x saving       |

---

## 2. Comparative Analysis

The table below aligns the selected regularized configs with **FedMAQ-Lite (SimpleCNN)** and the primary **ResNet18GN baselines**.

| Algorithm / Variant                  |       Dirichlet $\alpha=0.1$ Acc       |    Dirichlet $\alpha=1.0$ Acc     | Comm saving vs. FedAvg | Params |
| :----------------------------------- | :------------------------------------: | :-------------------------------: | :--------------------: | :----: |
| **FedAvg** (ResNet18GN)              |                 36.27%                 |            **67.57%**             |          1.0x          | 11.17M |
| **FedProx** (ResNet18GN)             |               **49.71%**               |              67.01%               |          1.0x          | 11.17M |
| **FedDistill** (ResNet18GN)          |                 32.94%                 |              66.86%               |          1.0x          | 11.17M |
| **FedMAQ-Lite** (Tuned SimpleCNN)    |               **52.83%**               |              63.28%               |    **8.6x saving**     | 2.16M  |
| **FedMAQ** (ResNet18GN baseline)     |                 38.36%                 |              53.04%               |    **1.7x saving**     | 11.17M |
| **FedMAQ** + KD Reg (Target Configs) |   **33.98%** (`alpha_0.5_temp_1.0`)    | **63.65%** (`alpha_0.5_temp_2.0`) |    **1.7x saving**     | 11.17M |
| **FedMAQ** + KD Reg (Best Configs)   | **35.06%** (`alpha_0.3_temp_2.0` peak) | **65.94%** (`alpha_0.3_temp_2.0`) |    **1.7x saving**     | 11.17M |
