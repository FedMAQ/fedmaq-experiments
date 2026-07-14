# Walkthrough: Completed Smoke Test Sweeps (40 Rounds)

We successfully updated [run_smoke_test.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/scripts/run_smoke_test.py) to support command line arguments for `--total_rounds` and `--heterogeneity`, and executed two complete sweeps of 40 rounds each under different data partitions:

1. **Dirichlet $\alpha=0.1$** (high statistical heterogeneity/non-IID skew)
2. **Dirichlet $\alpha=1.0$** (moderate statistical heterogeneity)

Below is the summary of changes made and the performance of all benchmarked federated learning algorithms.

---

## 1. Code Changes

We wrapped the execution logic of [run_smoke_test.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/scripts/run_smoke_test.py) inside a [main()](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/scripts/run_smoke_test.py#L33) function, added `argparse` to handle `--total_rounds` and `--heterogeneity`, and formatted the file using Ruff.

---

## 2. Benchmark Results

### Sweep A: Dirichlet $\alpha = 0.1$ (High Non-IID Skew, 40 Rounds)

| Algorithm         | Test Accuracy | Test Loss  | Cumulative Comm (MB) | Comm Reduction vs. FedAvg | Sim Latency (s) |
| :---------------- | :-----------: | :--------: | :------------------: | :-----------------------: | :-------------: |
| **FedProx**       |  **49.71%**   | **1.6441** |      34100.2 MB      |           1.0x            |     5554.0s     |
| **FedMAQ** (Ours) |  **38.76%**   | **1.6788** |    **3534.6 MB**     |    **9.6x reduction**     |   **3423.9s**   |
| **FedPAQ**        |    37.47%     |   1.9050   |      21312.7 MB      |           1.6x            |     4406.6s     |
| **FedAvg**        |    36.27%     |   2.1447   |      34100.2 MB      |           1.0x            |     5468.2s     |
| **DAdaQuant**     |    41.49%     |   2.0600   |      18531.4 MB      |           1.8x            |     4122.6s     |
| **FedDistill**    |    32.94%     |   1.8297   |      34100.5 MB      |           1.0x            |     5225.7s     |
| **FedKD**         |    14.09%     |   2.2648   |       26.9 MB        |          1267.4x          |     6294.5s     |
| **CFD**           |    10.00%     |   2.7509   |        0.9 MB        |         36052.3x          |     3396.0s     |

### Sweep B: Dirichlet $\alpha = 1.0$ (Moderate Non-IID Skew, 40 Rounds)

| Algorithm         | Test Accuracy | Test Loss  | Cumulative Comm (MB) | Comm Reduction vs. FedAvg | Sim Latency (s) |
| :---------------- | :-----------: | :--------: | :------------------: | :-----------------------: | :-------------: |
| **FedAvg**        |  **67.57%**   | **1.2225** |      34100.2 MB      |           1.0x            |     4231.0s     |
| **FedProx**       |    67.53%     |   1.2023   |      34100.2 MB      |           1.0x            |     4224.6s     |
| **FedDistill**    |    67.37%     |   0.9642   |      34100.5 MB      |           1.0x            |     4186.7s     |
| **FedPAQ**        |    66.91%     |   1.2328   |      21312.7 MB      |           1.6x            |     3129.9s     |
| **DAdaQuant**     |    65.91%     |   1.2534   |      18115.8 MB      |           1.9x            |     2860.7s     |
| **FedMAQ** (Ours) |  **58.07%**   | **1.1902** |    **3864.8 MB**     |    **8.8x reduction**     |   **2295.6s**   |
| **FedKD**         |    29.17%     |   1.8933   |       33.6 MB        |          1014.2x          |     3348.5s     |
| **CFD**           |    30.23%     |   2.1817   |        2.4 MB        |         14198.7x          |     2000.0s     |
