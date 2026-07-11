# Codebase-to-Manuscript Alignment Rules

All agents and developers modifying this codebase must adhere strictly to the following mathematical, architectural, and hyperparameter constraints from the LaTeX thesis manuscript (located in the `fedmaq-manuscript` sibling repository).

## 1. Hyperparameter Synchronization

Every benchmark experiment must use the hyperparameters specified in **Table 4.1** of the manuscript. These default parameters are captured in [conf/experiment/default.yaml](../../conf/experiment/default.yaml):

- $K = 100$ total clients (or $200$ for FEMNIST)
- $B = 64$ local batch size
- $E = 5$ local training epochs
- $\eta = 0.01$ base learning rate
- $\gamma = 0.99$ learning rate decay (exponential per-round decay)
- $\beta = 0.9$ SGD momentum
- $\lambda = 10^{-4}$ SGD weight decay
- $C = 0.1$ client fraction per round
- $R = 100$ communication rounds
- $\|D_{pub}\| = 1600$ server-side proxy dataset size

## 2. Quantization Soft Quality Target Formulas

Ensure that any precision-scaling evaluations strictly match the five mathematical formulations detailed in **Section 3.3** of the manuscript:

1. **Formulation 0**: Resource-Only Hard Cap ($q_hat = q_max$)
2. **Formulation 1**: Normalized Linear Weighted Sum ($\gamma_1 \tilde{g}_k + \gamma_2 \tilde{n}_k$)
3. **Formulation 2**: Normalized Multiplicative Scaling ($(\tilde{g}_k)^{\gamma_1} \cdot (\tilde{n}_k)^{\gamma_2}$)
4. **Formulation 3**: Gradient-Primary, Data-Modulated ($\tilde{g}_k \cdot \frac{1 + \lambda \tilde{n}_k}{1 + \lambda}$) -- _Default FedMAQ_
5. **Formulation 4**: Threshold-Based Staged Rule (piecewise discrete mapping to $q_max$, $q_mid$, or $q_min$ using thresholds $\tau_g, \tau_n$)

The outer memory constraint must always be active: $q_k^{(t)} = \min(Q_k^{max}, \hat{q}_k^{(t)})$ where $Q_k^{max} = \lfloor c_k / c_{unit} \rfloor$.

The final bit-width (both $\hat{q}_k^{(t)}$ and $Q_k^{max}$) must snap to the manuscript's permissible discrete set $\mathcal{Q} = \{1,2,3,4,5,6,7,8,16,32\}$, not an arbitrary continuous integer.

## 3. Decoupled Simulated Time & Overheads

The simulation time logged in `TelemetryFedAvg` must track:

1. **Round Time**: $T_{round}^{(t)} = \max_{i \in S^{(t)}} (T_{loc}^{i,t}) + T_{server}^{(t)}$
2. **Edge Compute Penalties**:
   - FedKD client compute speed scales by $1 / 2.5$ (due to joint teacher-student training).
   - FedMD client training delay includes public/private pre-training epochs during Round 1.
3. **Server Distillation Overhead**:
   - FedMAQ server distillation delay scales with proxy size, epochs, and number of teachers: $T_{server} = \frac{\|D_{pub}\| \cdot E_{kd} \cdot K_{active}}{ServerSpeed}$.

## 4. Locked Cross-Repo Architectural Decisions

| Topic              | Decision                                                                                                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thesis context     | `fedmaq-experiments/.claude/rules/` is canonical (decomposed from a prior `context.md`)                                                                                                                                                      |
| Experiments layout | uv monorepo, code under `src/fedmaq/core/` and `src/fedmaq/baselines/`                                                                                                                                                                       |
| Tooling            | Preferred stack in `tech-stack.md`; adopt extra libs (pandas, sklearn, etc.) when justified                                                                                                                                                  |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                                                                                                                                                              |
| Literature KG      | OKF bundle at `kg/` (see `fedmaq-literature/SPEC.md`); two layers — raw `markdown/` (citable) + curated OKF nodes. No vector store (grep + read); nodes authored directly, no approve gate (review via `git diff`); no cross-repo auto-edits |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                                                                                                                                                               |

## 5. Testing Rigor

Any mathematical or simulation configuration updates must be accompanied by comprehensive tests verifying correctness and determinism, particularly for:

- Stochastic rounding (DAdaQuant); symmetric uniform quantization (FedPAQ and FedMAQ, which share bit-width semantics)
- Formulation output values
- Decoupled telemetry calculations
- Deterministic data partitioning (Dirichlet skew and FEMNIST writer chunking)
