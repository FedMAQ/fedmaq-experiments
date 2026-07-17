# Empirical Analysis & Thesis Narrative: Stacked Regularization Sweep

This document provides the theoretical basis, physical interpretation, and thesis narrative alignment for stacking logit-space KD regularization and parameter-space proximal regularization (July 15, 2026).

---

## 1. Physical Mechanism of Stacked Regularization

By combining FedGKD (logit-space) and FedProx (parameter-space) regularizations, the local client training loss function is formulated as:

$$\mathcal{L}_{\text{client}} = (1 - \alpha_{\text{reg}}) \mathcal{L}_{\text{CE}} + \alpha_{\text{reg}} D_{\text{KL}}\left(\sigma\left(\frac{z_{\text{global}}}{T_{\text{reg}}}\right) \,\middle\|\, \sigma\left(\frac{z_{\text{local}}}{T_{\text{reg}}}\right)\right) T_{\text{reg}}^2 + \frac{\mu_{\text{reg}}}{2} \sum \|w - w_{\text{global}}\|^2$$

This dual-regularization approach targets client drift at two distinct levels of abstraction:

1. **Logit-Space Constraint (Representation / Function Space):**
   - Restricts local models from shifting their functional mappings (outputs on the public dataset $D_{proxy}$) too far from the global consensus.
   - Preserves dark knowledge (inter-class correlations) and stabilizes distillation gradients on the server.
2. **Parameter-Space Constraint (Weights Space):**
   - Restricts local models from shifting their weight coordinates too far from the global starting point in the high-dimensional parameter space.
   - Mitigates the "Capacity-Drift" Paradox: in high-capacity models like ResNet18GN (~11.17M params), logit-space constraints alone are under-determined, allowing weights to drift significantly while maintaining similar logits on the training set. A direct L2 parameter constraint anchors the model weights, stabilizing server-side model averaging.

---

## 2. Parameter Tuning Rationale

This experiment sweeps `kd_prox_mu` $\in \{0.0, 0.001, 0.01, 0.1, 1.0\}$ under the best configurations obtained from the prior Client-Side KD Regularization Sweep:

- **Dirichlet $\alpha=0.1$ (Severe Skew):**
  - Uses `kd_reg_alpha=0.5` and `kd_reg_temp=1.0` (which achieved the best final accuracy of 33.98%).
  - Lowered `entropy_weight` from `4.0` (the SimpleCNN default) to `1.0` to prevent voter exclusion under high skew, enabling ResNet18GN teachers to contribute logits to the ensemble.
- **Dirichlet $\alpha=1.0$ (Moderate Skew):**
  - Uses `kd_reg_alpha=0.3` and `kd_reg_temp=2.0` (which achieved the best accuracy of 65.94%, within 1.63pp of uncompressed FedAvg).
  - Uses `entropy_weight=2.0`.
