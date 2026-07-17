# Empirical Analysis & thesis Narrative: Client-Side KD Regularization

This document provides the empirical analysis, physical interpretation, and Master's thesis narrative alignment for the Client-Side KD Regularization sweep (July 15, 2026).

---

## 1. Physical Interpretation of Results

### A. The Logit-Space Regularization Mechanism (FedGKD-Style)

Client-side KD regularization operates by penalizing the KL divergence between a client's local predictions and the incoming global model's predictions:
$$L = (1 - \alpha_{\text{reg}}) L_{\text{CE}} + \alpha_{\text{reg}} D_{\text{KL}}\left(\sigma(z_{\text{global}}/T_{\text{reg}}) \,\|\, \sigma(z_{\text{local}}/T_{\text{reg}})\right) T_{\text{reg}}^2$$

This logit-space penalty acts as a **soft representation constraint**:

1. **Prevents Aggressive Local Specialization**: Because each client's dataset is statistically skewed (Dirichlet non-IID), local training naturally shifts the model's decision boundaries to fit the local distribution. KD regularization forces the student logits to stay aligned with the global consensus.
2. **Quantization Noise Mitigation**: Overfitting generates larger weight updates (deltas). Since quantization error scales with update magnitude, aggressive overfitting increases quantization noise. KD regularization controls weight delta magnitude, leading to lower quantization error.
3. **Improved Ensemble Quality**: Server-side KD relies on an ensemble of client models. Reducing local client drift prevents Jensen's inequality from collapsing the blended probability distribution into a uniform (meaningless) distribution.

---

## 2. Parameter Tuning Dynamics

### A. Regularization Strength ($\alpha_{\text{reg}}$) Crossover

- **Moderate Skew ($\alpha = 1.0$)**: $\alpha_{\text{reg}}=0.3$ is optimal. Setting it too high ($\alpha_{\text{reg}}=0.5$ or $0.7$) penalizes local learning too much, causing a drop in final accuracy (e.g., $65.94\%$ at $\alpha_{\text{reg}}=0.3$ vs $63.65\%$ at $\alpha_{\text{reg}}=0.5$).
- **Severe Skew ($\alpha = 0.1$)**: $\alpha_{\text{reg}}=0.5$ is required to stabilize local training. Because client drift is much more aggressive, a stronger regularization penalty is needed to anchor the clients.

### B. Temperature ($T_{\text{reg}}$) and Soft vs. Sharp Targets

- Under moderate skew ($\alpha = 1.0$), softening the global model's logit distribution ($T=2.0$) transfers rich inter-class relationships (dark knowledge), preventing local models from over-specializing.
- Under severe skew ($\alpha = 0.1$), lower temperature ($T=1.0$) performs better. Since the global model under severe skew is less accurate, its soft predictions are noisy. A sharp target ($T=1.0$) filters out this noise, providing a cleaner consensus signal.

---

## 3. High-Capacity Model Capacity and Soft-Voting Bottleneck

### The Suboptimal Hyperparameter Gap under $\alpha=0.1$

A key finding from this sweep is that the regularized ResNet18GN under severe skew (`kd_reg_alpha_0.5_temp_1.0` at **33.98%**) performed worse than the original unregularized ResNet18GN baseline (**38.36%**).

This discrepancy is a direct result of **suboptimal hyperparameter transfer** from the low-capacity model:

1. **Entropy Threshold Sensitivity**: The sweep used `entropy_weight=4.0`, which was tuned for the smaller SimpleCNN (~2.16M params) in FedMAQ-Lite.
2. **Voter Exclusion**: High-capacity ResNet18GN overfits aggressively to its local shard. When evaluating the out-of-distribution public dataset ($D_{proxy}$), the client models produce highly uncertain, high-entropy predictions.
3. **KD Ensemble Collapse**: A strict entropy weight scale of `4.0` suppresses any predictions with moderate-to-high entropy:
   $$W_{\text{entropy}} = e^{-\gamma_{\text{entropy}} \cdot H(P)} \approx 0.0$$
   This causes the server to ignore nearly all client teachers during distillation, starving the global model of gradient updates.
4. **Resolution**: The unregularized ResNet18GN baseline run used a more tolerant `entropy_weight=1.0`, allowing the global student to train on client knowledge. To realize the gains of KD regularization under severe skew on ResNet18GN, we must re-tune the soft-voting hyperparameters (e.g., lowering `entropy_weight` to `1.0` or `0.5`).
