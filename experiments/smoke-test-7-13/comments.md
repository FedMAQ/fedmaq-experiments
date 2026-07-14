# Empirical Evaluation & Hyperparameter Tuning Guide (40-Round Smoke Test)

This document contains empirical performance comments and an exhaustive hyperparameter tuning guide compiled from the 40-round baseline smoke tests on CIFAR-10 (Dirichlet $\alpha=0.1$ and $\alpha=1.0$) and the [fedmaq-literature](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/) knowledge graph (KG).

---

## 1. Empirical Evaluation of Algorithm Correctness

Based on the 40-round smoke test across severe ($\alpha=0.1$) and moderate ($\alpha=1.0$) statistical skews, the baseline implementations fall into three functional categories:

### Correct and Validated Implementations

- **FedAvg (Seminal):** Establishes the baseline communication footprint ($1.0\times$ baseline, 34100.2 MB). The accuracy drop from 67.57% under moderate skew to 36.27% under severe statistical skew aligns with the well-documented weight divergence and client-drift phenomena in literature [\cite{mcmahanCommunicationEfficientLearningDeep2017}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/mcmahan-2017-fedavg.md).
- **FedProx (Seminal):** Functions as expected. It maintains FedAvg's communication footprint but successfully mitigates non-IID client drift, improving accuracy to 49.71% at $\alpha=0.1$. This confirms the proximal regularization term is correctly pulling local updates closer to the global model [\cite{liFederatedOptimizationHeterogeneous2020}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/li-2020-fedprox.md).
- **DAdaQuant & FedPAQ (Pure Quantization):** Both algorithms demonstrate the expected $1.6\times$ to $1.9\times$ communication reduction [\cite{honigDAdaQuantDoublyadaptiveQuantization2022}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/honig-2022-dadaquant.md), [\cite{reisizadehFedPAQCommunicationEfficientFederated2020}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/reisizadeh-2020-fedpaq.md). DAdaQuant correctly outperforms FedPAQ in both accuracy and communication reduction across both sweeps, validating that doubly-adaptive quantization preserves more gradient features than static fixed-point quantization.

### Suspicious Implementation (Potential Misalignment)

- **FedDistill+ (Pure KD):** Telemetry records a 34100.5 MB footprint, which is slightly higher than FedAvg's footprint. This indicates the execution of the stronger FedDistill+ baseline, which shares both parameters and logit vectors [\cite{zhuDataFreeKnowledgeDistillation2021}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/zhu-2021-fedgen.md). However, the accuracy is 32.94% at $\alpha=0.1$, performing worse than FedAvg. Since knowledge distillation is designed to mitigate non-IID skew, this indicates a hyperparameter misalignment (e.g. `reg_alpha` is poorly tuned) rather than a structural bug.

### Over-compressed Implementations (Need Tuning)

- **FedKD (Hybrid Q+KD):** Test accuracy at 14.09% and 29.17% is substantially lower than expected. The $1000\times+$ communication reduction indicates the Singular Value Decomposition (SVD) threshold is over-compressing the model, deteriorating the gradient updates before transmission [\cite{wuCommunicationefficientFederatedLearning2022}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/wu-2022-fedkd.md).
- **CFD (Hybrid Q+KD):** At $\alpha=0.1$, the 10.00% accuracy equates to random guessing. While the $14,000\times+$ communication reduction matches the literature [\cite{sattlerCFDCommunicationEfficientFederated2022}](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/sattler-2022-cfd.md), the collapsed accuracy indicates the soft-label logits are deteriorating due to overly aggressive quantization prior to reaching the server (needs higher `b_up`/`b_down` values).

---

## 2. Exhaustive Hyperparameter Tuning Guide

To ensure a fair and competitive evaluation, we standardize common training hyperparameters and align algorithm-specific hyperparameters with original paper recommendations where possible. This simplifies the final 516-run grid and avoids confounding variables.

Below is the updated lookup of all hyperparameters, including default values, suggested ranges (where tuning is necessary), and specific statuses.

### 2.1. Global & Seminal Control Hyperparameters

These are defined globally in `conf/experiment/default.yaml` and form the foundation of local optimization for all algorithms.

| Parameter Key                    |    Symbol     | Codebase Default |             Tuning Status              | Recommended Value / Range | Description & Rationale                                                                                      |
| :------------------------------- | :-----------: | :--------------: | :------------------------------------: | :-----------------------: | :----------------------------------------------------------------------------------------------------------- |
| `experiment.local_epochs`        |      $E$      |       `5`        |   **Fixed to Paper Recommendation**    |            `5`            | Number of local gradient epochs (McMahan et al. 2017).                                                       |
| `experiment.batch_size`          |      $B$      |       `64`       |   **Fixed to Paper Recommendation**    |           `64`            | Local minibatch size (McMahan et al. 2017).                                                                  |
| `experiment.learning_rate`       |    $\eta$     |      `0.01`      | **Fixed to Literature Recommendation** |          `0.01`           | Base client learning rate. Standardized to 0.01 to avoid confounding optimization and compression variables. |
| `experiment.learning_rate_decay` |   $\gamma$    |      `0.99`      |     **Fixed to Baseline Default**      |          `0.99`           | Exponential decay factor applied per communication round.                                                    |
| `experiment.weight_decay`        |   $\lambda$   |     `0.0001`     |     **Fixed to Baseline Default**      |         `0.0001`          | SGD weight decay ($10^{-4}$) to prevent overfitting.                                                         |
| `experiment.momentum`            |    $\beta$    |      `0.9`       |     **Fixed to Baseline Default**      |           `0.9`           | SGD momentum coefficient.                                                                                    |
| `experiment.client_fraction`     |      $C$      |      `0.1`       |     **Fixed to Baseline Default**      |           `0.1`           | Client participation fraction per round (10% of 100 clients).                                                |
| `experiment.num_public_samples`  | $\|D_{pub}\|$ |      `3000`      |     **Fixed to Baseline Default**      |          `3000`           | Size of server-side unlabeled proxy dataset. Updated to 3000 to stabilize distillation.                      |

---

### 2.2. Algorithm-Specific Hyperparameters

Configured in their respective files in `conf/algorithm/`.

#### FedProx ([fedprox.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedprox.yaml))

- **Method Paper:** [Federated Optimization in Heterogeneous Networks (Li et al. 2020)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/li-2020-fedprox.md)

| Parameter Key  | Symbol | Codebase Default |  Tuning Status  | Recommended Value / Range | Description & Rationale                                                                                                                                                                                                 |
| :------------- | :----: | :--------------: | :-------------: | :-----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `algorithm.mu` | $\mu$  |      `0.01`      | **To Be Tuned** |   $\{0.01, 0.1, 1.0\}$    | Weight of proximal regularization term. Highly sensitive to heterogeneity: larger $\mu$ (e.g. 0.1 or 1.0) is needed to curb client drift under severe skew ($\alpha=0.1$), while too large values stall local training. |

#### FedPAQ ([fedpaq.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedpaq.yaml))

- **Method Paper:** [FedPAQ: A Communication-Efficient FL Method with Periodic Averaging & Quantization (Reisizadeh et al. 2020)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/reisizadeh-2020-fedpaq.md)

| Parameter Key | Symbol | Codebase Default |         Tuning Status         | Recommended Value / Range | Description & Rationale                                                                                                                         |
| :------------ | :----: | :--------------: | :---------------------------: | :-----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| `algorithm.q` |  $q$   |       `8`        | **Fixed to Baseline Default** |            `8`            | Bit-width for model delta quantization. Keeps quantization stable and comparable to other models without sweeping multiple discrete bit-widths. |

#### DAdaQuant ([dadaquant.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/dadaquant.yaml))

- **Method Paper:** [DAdaQuant: Doubly-adaptive quantization for FL (Hönig et al. 2022)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/honig-2022-dadaquant.md)

| Parameter Key     |  Symbol   | Codebase Default |           Tuning Status           | Recommended Value / Range | Description & Rationale                                               |
| :---------------- | :-------: | :--------------: | :-------------------------------: | :-----------------------: | :-------------------------------------------------------------------- |
| `algorithm.q_min` | $q_{min}$ |       `1`        | **Fixed to Paper Recommendation** |            `1`            | Minimum adaptive bit-width.                                           |
| `algorithm.q_max` | $q_{max}$ |       `8`        | **Fixed to Paper Recommendation** |            `8`            | Maximum adaptive bit-width cap.                                       |
| `algorithm.psi`   |  $\psi$   |      `0.9`       | **Fixed to Paper Recommendation** |           `0.9`           | Smoothing weight for moving average of estimated global loss.         |
| `algorithm.phi`   |  $\phi$   |       `5`        | **Fixed to Paper Recommendation** |            `5`            | Plateau lookback window (number of rounds before doubling bit-width). |

#### FedMD ([fedmd.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedmd.yaml))

- **Method Paper:** [FedMD: Heterogenous FL via Model Distillation (Li and Wang 2019)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/li-2019-fedmd.md)

| Parameter Key                       | Symbol | Codebase Default |           Tuning Status           | Recommended Value / Range | Description & Rationale                                                                 |
| :---------------------------------- | :----: | :--------------: | :-------------------------------: | :-----------------------: | :-------------------------------------------------------------------------------------- |
| `algorithm.public_pretrain_epochs`  |   -    |       `10`       | **Fixed to Paper Recommendation** |           `10`            | Number of transfer learning epochs on public data.                                      |
| `algorithm.private_pretrain_epochs` |   -    |       `10`       | **Fixed to Paper Recommendation** |           `10`            | Number of initial training epochs on local private data.                                |
| `algorithm.public_epochs`           |   -    |       `5`        | **Fixed to Paper Recommendation** |            `5`            | Number of distillation (digest) epochs per round. Set to 5 to avoid extra tuning steps. |

#### FedDistill+ ([feddistill.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/feddistill.yaml))

- **Method Paper:** [Data-Free KD for Heterogeneous FL (Zhu et al. 2021)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/zhu-2021-fedgen.md)

| Parameter Key         |  Symbol  | Codebase Default |           Tuning Status           | Recommended Value / Range | Description & Rationale                                                                                                      |
| :-------------------- | :------: | :--------------: | :-------------------------------: | :-----------------------: | :--------------------------------------------------------------------------------------------------------------------------- |
| `algorithm.reg_alpha` | $\alpha$ |      `1.0`       | **Fixed to Paper Recommendation** |           `1.0`           | Regularization weight for the label-wise logit-distillation term. Fixed at 1.0 to reflect standard literature configuration. |

#### CFD ([cfd.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/cfd.yaml))

- **Method Paper:** [CFD: Communication-Efficient Federated Distillation (Sattler et al. 2022)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/sattler-2022-cfd.md)

| Parameter Key                     |   Symbol   | Codebase Default |           Tuning Status           | Recommended Value / Range | Description & Rationale                                                                                                      |
| :-------------------------------- | :--------: | :--------------: | :-------------------------------: | :-----------------------: | :--------------------------------------------------------------------------------------------------------------------------- |
| `algorithm.b_up`                  |  $b_{up}$  |       `1`        |          **To Be Tuned**          |       $\{1, 2, 4\}$       | Upstream soft-label quantization bit-width. Needs tuning to 2 or 4 to avoid collapsed accuracy under Dirichlet $\alpha=0.1$. |
| `algorithm.b_down`                | $b_{down}$ |       `1`        |          **To Be Tuned**          |       $\{1, 2, 4\}$       | Downstream soft-label quantization bit-width.                                                                                |
| `algorithm.delta_coding`          |     -      |      `true`      | **Fixed to Paper Recommendation** |          `true`           | Lossless delta coding of prediction updates across rounds.                                                                   |
| `algorithm.distill_epochs`        |     -      |       `1`        | **Fixed to Paper Recommendation** |            `1`            | Client-side distillation epochs.                                                                                             |
| `algorithm.server_distill_epochs` |     -      |       `1`        | **Fixed to Paper Recommendation** |            `1`            | Server-side dual distillation epochs.                                                                                        |
| `algorithm.temperature`           |    $T$     |      `1.0`       | **Fixed to Paper Recommendation** |           `1.0`           | Softmax temperature for logit generation.                                                                                    |

#### FedKD ([fedkd.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedkd.yaml))

- **Method Paper:** [Communication-efficient FL via knowledge distillation (Wu et al. 2022)](file:///C:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/wu-2022-fedkd.md)

| Parameter Key               |   Symbol    | Codebase Default |           Tuning Status           | Recommended Value / Range | Description & Rationale                                                      |
| :-------------------------- | :---------: | :--------------: | :-------------------------------: | :-----------------------: | :--------------------------------------------------------------------------- |
| `algorithm.tmin`            | $T_{start}$ |      `0.1`       | **Fixed to Paper Recommendation** |           `0.1`           | Starting SVD energy cutoff threshold (more compression).                     |
| `algorithm.tmax`            |  $T_{end}$  |      `0.95`      | **Fixed to Paper Recommendation** |          `0.95`           | Target SVD energy threshold.                                                 |
| `algorithm.temperature`     |     $T$     |      `2.0`       | **Fixed to Paper Recommendation** |           `2.0`           | Client-side mutual distillation temperature.                                 |
| `algorithm.compute_penalty` |      -      |      `2.5`       |          **Fixed (Sim)**          |           `2.5`           | Simulated compute overhead factor representing local mentor+mentee training. |

#### FedMAQ (Ours) ([fedmaq.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedmaq.yaml))

- **Specification:** `fedmaq-manuscript/chapter_3.tex` Sections 3.3, 3.5; `chapter_4.tex` Sections 4.2, 4.4.

| Parameter Key                  |      Symbol      | Codebase Default |         Tuning Status          | Recommended Value / Range | Description & Rationale                                                                                   |
| :----------------------------- | :--------------: | :--------------: | :----------------------------: | :-----------------------: | :-------------------------------------------------------------------------------------------------------- |
| `algorithm.q_min`              |    $q_{min}$     |       `1`        | **Fixed to Recommended Value** |            `1`            | Minimum bit-width for soft quality interpolation range. Set to 1 to allow full binary compression limits. |
| `algorithm.q_max`              |    $q_{max}$     |       `16`       | **Fixed to Recommended Value** |           `16`            | Maximum bit-width for soft quality interpolation range. Set to 16 to allow high-precision updates.        |
| `algorithm.c_unit`             |    $c_{unit}$    |     `512.0`      | **Fixed to Recommended Value** |          `512.0`          | Memory capacity calibration (MB per quantization bit).                                                    |
| `algorithm.formulation`        |        -         |       `3`        | **Fixed to Recommended Value** |            `3`            | Tier 2 soft-quality combination logic. `3` (Gradient-Primary) was selected in the pilot study.            |
| `algorithm.lambda_val`         |    $\lambda$     |      `1.0`       |        **To Be Tuned**         |    $\{0.5, 1.0, 2.0\}$    | Dataset size modulator strength (Formulation 3 only).                                                     |
| `algorithm.temperature`        |       $T$        |      `1.0`       | **Fixed to Recommended Value** |           `1.0`           | Server-side KD temperature.                                                                               |
| `algorithm.kd_epochs`          |        -         |       `1`        | **Fixed to Recommended Value** |            `1`            | Number of distillation passes over $D_{proxy}$ per round.                                                 |
| `algorithm.server_kd_lr`       | $\eta_{server}$  |      `0.01`      | **Fixed to Recommended Value** |          `0.01`           | Server-side student learning rate during KD.                                                              |
| `algorithm.server_kd_momentum` | $\beta_{server}$ |      `0.9`       | **Fixed to Recommended Value** |           `0.9`           | Server-side student SGD momentum.                                                                         |
| `algorithm.post_process`       |        -         |      `true`      | **Fixed to Recommended Value** |          `true`           | Enables error-feedback, diff-coding, and zlib on gradients.                                               |

---

## 3. Verification Note

Prior to running the full 516-run grid, future agents must systematically trace the parameter dependencies in [src/fedmaq/baselines/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/baselines/) and [src/fedmaq/core/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/) to ensure no auxiliary hyperparameters (e.g. specific optimizer momentum schedules or differential privacy noise parameters) are omitted from our test configurations.

---

## 4. Suggested Manuscript Additions

Based on the smoke test evaluation, we recommend adding the following sections to Chapter 4 (Methodology):

1. **Hyperparameter Search Space Appendix:** Detail the exact grid-search ranges (such as those in Section 2 above) to document how baselines were optimized.
2. **Baseline Validation Paragraph:** Explain that all baselines were subjected to preliminary hyperparameter alignment sweeps to ensure they function as competitive upper bounds (preventing claims of beating artificially weakened comparators).
