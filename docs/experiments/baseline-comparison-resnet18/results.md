# Baseline Comparison Results (FedMAQ ResNet18GN)

Empirical results for the full-sized **FedMAQ** (ResNet18GN, ~11.17M params) baseline sweeps, comparing its performance under statistically skewed regimes.

Completed/Recorded: July 15, 2026.

All runs use:

- **Variant**: `fedmaq` (ResNet18GN)
- **Formulation**: 3 (Gradient-Primary)
- **Grad Norm Smoothing**: `grad_norm_ema=true` (`β=0.7`)
- **Default Hyperparameters**:
  - `ema_decay=0.7` (best for α=0.1)
  - `entropy_weight=1.0`, `precision_weight=1.0`
  - `temperature=1.0`

Logs: [multirun/2026-07-15/baseline-comparison-resnet18/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/multirun/2026-07-15/baseline-comparison-resnet18/)

---

## Centralized Test Performance Comparison (at Round 40)

Comparison with baseline results from the 40-round sequential smoke tests on ResNet18GN:

| Algorithm / Skew  | Dirichlet α=0.1 (Severe Skew) | Dirichlet α=1.0 (Moderate Skew) | Comm Footprint (α=0.1) |
| :---------------- | :---------------------------: | :-----------------------------: | :--------------------: |
| **FedMAQ (Ours)** |          **38.36%**           |       \*53.04% (R34)\*\*        |     **20195.2 MB**     |
| **FedAvg**        |            36.27%             |             67.57%              |       34100.2 MB       |
| **FedDistill**    |            32.94%             |             66.86%              |           -            |
| **FedProx**       |          **49.71%**           |             67.01%              |       34100.2 MB       |

_\*Note: The Dirichlet α=1.0 run was manually cancelled at Round 34 because it was run with the severe-skew default (`ema_decay=0.7`) instead of the moderate-skew optimal value (`ema_decay=0.1`). It will be rerun in the next session._

### Key Metrics for Dirichlet α=0.1 Run (Round 40):

- **Final Test Accuracy:** 38.36%
- **Final Test Loss:** 1.6914
- **Final Uploaded Footprint:** 20195.2 MB (representing ~1.7× communication savings compared to full-precision FedAvg)

### Key Metrics for Dirichlet α=1.0 Run (Cancelled at Round 34):

- **Accuracy at Round 34:** 53.03% (peak 53.04%)
- **Loss at Round 34:** 1.3185
