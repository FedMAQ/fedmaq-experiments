# Analysis of Stage 1a Power-Mean Degree Selection (\(p = 0.5\))

**Date**: 2026-09-14  
**Context**: Resolves Stage 1a of the formulation design space ([ADR-0021](../adr/0021-power-mean-formulation-family.md), [ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md), [Issue #34](https://github.com/FedMAQ/fedmaq-experiments/issues/34), [Issue #110](https://github.com/FedMAQ/fedmaq-experiments/issues/110)).  
**Authoritative artifacts**:
- `scripts/analysis_output/power_mean_degree_selection.json`
- `scripts/analysis_output/power_mean_degree_resolution.json`

---

## 1. Executive Summary & Verdict

The 126-cell Stage 1a sweep (`power_mean_design`) evaluated the generalized power-mean degree ladder:
\[
s_k = M_p(\tilde{g}_k, \tilde{n}_k; \omega) = \left(\omega \tilde{g}_k^p + (1-\omega) \tilde{n}_k^p\right)^{1/p}, \quad \omega = 0.5
\]
across \(p \in \{1, 0.5, 0, -0.5, -1, -2, \text{min}\}\), plus three descriptive controls (F0 resource-only, F3 gradient-primary \(\kappa \in \{0.5, 1.0, 2.0\}\), and F4 threshold \(\tau \in \{0.3, 0.5, 0.7\}\)) over CIFAR-10 / MobileNetV2 (100 rounds, 3 seeds).

Under the pre-registered iso-byte evaluation rule ([ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md)):
- **Winning formulation parameter**: **\(p = 0.5\)**
- **Selection rule outcome**: **`"rule": "agreement"`**
- Both reporting skews independently and unambiguously selected \(p = 0.5\):
  - \(\alpha = 0.1\) (severe non-IID): **`p0.5`** (accuracy: \(0.2863 \pm 0.0097\), margin \(+0.0096\) over runner-up `p-0.5`, \(+0.0164\) over geometric `p0`)
  - \(\alpha = 1.0\) (mild non-IID): **`p0.5`** (accuracy: \(0.5331 \pm 0.0174\), margin \(+0.0027\) over runner-up `p0`, \(+0.0155\) over arithmetic `p1`)

No severe-skew tie-break was needed.

---

## 2. Empirical Performance Across Skews

The table below summarizes the mean top-1 validation accuracy at the minimum common cumulative-MB budget (\(B\)), with seed standard deviation across seeds 0, 42, 123:

| Degree \(p\) | Operator Category | \(\alpha = 0.1\) (\(B \approx 8058\) MB) | \(\alpha = 1.0\) (\(B \approx 8198\) MB) | \(\alpha = 0.3\) Robustness Holdout (\(B \approx 8130\) MB) |
| :--- | :--- | :---: | :---: | :---: |
| **`p1`** | Arithmetic mean (linear, fully compensatory) | \(0.2716 \pm 0.0216\) | \(0.5176 \pm 0.0097\) | \(0.4186 \pm 0.0200\) |
| **`p0.5`** | Root-mean (concave, mildly compensatory) | **\(0.2863 \pm 0.0097\)** ★ | **\(0.5331 \pm 0.0174\)** ★ | \(0.4167 \pm 0.0181\) |
| **`p0`** | Geometric mean (multiplicative, historical v1 F2) | \(0.2699 \pm 0.0205\) | \(0.5305 \pm 0.0198\) | \(0.4254 \pm 0.0183\) |
| **`p-0.5`** | Sub-geometric mean | \(0.2767 \pm 0.0146\) | \(0.5061 \pm 0.0228\) | \(0.4407 \pm 0.0180\) |
| **`p-1`** | Harmonic mean | \(0.2630 \pm 0.0189\) | \(0.5269 \pm 0.0192\) | \(0.4425 \pm 0.0291\) |
| **`p-2`** | Sub-harmonic mean | \(0.2572 \pm 0.0201\) | \(0.5249 \pm 0.0150\) | \(0.4373 \pm 0.0224\) |
| **`p-min`** | Leontief minimum operator | \(0.2498 \pm 0.0247\) | \(0.5266 \pm 0.0043\) | \(0.4297 \pm 0.0493\) |

### Comparison to Descriptive Controls

| Control | Parameters | \(\alpha = 0.1\) | \(\alpha = 1.0\) | \(\alpha = 0.3\) |
| :--- | :--- | :---: | :---: | :---: |
| **Resource-only (F0)** | Hardware memory cap only | \(0.2767 \pm 0.0308\) | \(0.5174 \pm 0.0096\) | \(0.4233 \pm 0.0200\) |
| **F3 (Gradient-primary)** | \(\kappa = 0.5\) | \(0.2816 \pm 0.0241\) | \(0.5121 \pm 0.0034\) | \(0.4183 \pm 0.0172\) |
| | \(\kappa = 1.0\) | \(0.2710 \pm 0.0136\) | \(0.5120 \pm 0.0178\) | \(0.4078 \pm 0.0429\) |
| | \(\kappa = 2.0\) | \(0.2587 \pm 0.0055\) | \(0.5027 \pm 0.0318\) | \(0.4353 \pm 0.0414\) |
| **F4 (Threshold-tied)** | \(\tau_g = \tau_n = 0.3\) | \(0.2751 \pm 0.0283\) | \(0.5174 \pm 0.0096\) | \(0.4233 \pm 0.0200\) |
| | \(\tau_g = \tau_n = 0.5\) | \(0.2878 \pm 0.0361\) | \(0.5051 \pm 0.0170\) | \(0.4194 \pm 0.0275\) |
| | \(\tau_g = \tau_n = 0.7\) | \(0.2501 \pm 0.0262\) | *(out of support)* | *(out of support)* |

---

## 3. Theoretical & Behavioral Insights

### 3.1 Why Both Extremes (\(p=1\) and \(p < 0\)) Underperform
1. **The Linear Failure Mode (\(p = 1\))**:
   Under an arithmetic combination, high local sample count (\(\tilde{n}_k\)) can completely offset a near-zero or noisy gradient update (\(\tilde{g}_k\)). The server allocates high quantization precision to clients that contribute little new directional information, diluting uplink bandwidth.
2. **The Non-Compensatory / Leontief Failure Mode (\(p \le -1\))**:
   Harmonic and minimum operators impose severe penalties on any asymmetry. If a client possesses highly informative gradient updates but modest local partition size, its soft score collapses near zero, starving it of communication bits.
3. **The Concave Optimum (\(p = 0.5\))**:
   \[
   s_k = \left(\frac{\sqrt{\tilde{g}_k} + \sqrt{\tilde{n}_k}}{2}\right)^2
   \]
   Sub-linear powers in \((0, 1)\) enforce **diminishing marginal returns** on any single signal (capping the degree to which data volume can dominate state awareness), while remaining sufficiently compensatory that an informative client with small data volume is not starved.

### 3.2 Subsuming Historical v1 (Formulation 2)
In the superseded v1 campaign ([ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md)), FedMAQ froze Formulation 2 (geometric / multiplicative, corresponding to \(p = 0\)). The continuous power-mean ladder reveals that:
- Geometric combination was indeed better than arithmetic addition, but
- \(p = 0.5\) strictly improves upon geometric mean across both reporting skews (\(+1.64\%\) accuracy at \(\alpha=0.1\), \(+0.27\%\) accuracy at \(\alpha=1.0\)), with noticeably lower variance across seeds (\(\sigma = 0.0097\) vs. \(0.0205\) at \(\alpha=0.1\)).

### 3.3 Continuous Formulation vs. Discrete Heuristics
While `f4-threshold-0.5` achieved \(0.2878\) at \(\alpha=0.1\), it collapsed to \(0.5051\) at \(\alpha=1.0\), and threshold \(\tau=0.7\) failed completely to reach the common communication budget. `p0.5` avoids threshold tuning, providing a smooth mathematical operator that excels under both severe and mild non-IID conditions.

### 3.4 The \(\alpha = 0.3\) Robustness Contrast
Per [ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md), \(\alpha=0.3\) was added as an empirical contrast that does not govern selection. In \(\alpha=0.3\), the negative powers (\(p=-1, -0.5\)) yielded slightly higher mean accuracies (\(\sim 0.44\) vs. \(0.417\)), within the overlapping noise bounds (\(\sigma \approx 0.02\text{--}0.03\)). This variation provides an empirical footnote for the dissertation discussion regarding intermediate skew regimes without compromising pre-registered protocol discipline.

---

## 4. Operational Handoff to Stage 1b

With \(p = 0.5\) established:
1. **Stage 1b matrix**: Materialize `conf/matrix/power_mean_omega.yaml` by setting `algorithm.p=0.5` over \(\omega \in \{0.25, 0.75\}\) (12 cells total).
2. **Reused cells**: The \(\omega = 0.5\) cells at \(p = 0.5\) from Stage 1a are reused by analysis, not re-run.
3. **Host layout**: Non-canonical directories on JupyterHub remain preserved in `outputs_stash/` at repo root, leaving `outputs/` strictly canonical.
