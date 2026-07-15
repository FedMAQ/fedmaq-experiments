# Client-Side Regularization for FedMAQ (ResNet18GN)

## Problem Statement

FedMAQ "full" (ResNet18GN, ~11.17M params) at α=0.1 achieved only **38.36%** accuracy at Round 40 using the same hyperparameters tuned for FedMAQ-Lite (SimpleCNN, ~2.16M params, **52.83%**). This is a **−14.47pp** gap between variants that share identical server-side logic, distillation pipeline, and quantization scheme.

### Root Cause Analysis

The performance gap is almost certainly **client drift amplified by model capacity**:

1. **High-capacity overfitting**: ResNet18GN (11.17M params) has **5.2x more parameters** than SimpleCNN (2.16M). Under α=0.1 severe non-IID, each client's local shard may contain only ~500 samples dominated by 1-2 classes. ResNet18GN has ample capacity to memorize these tiny, skewed partitions in 5 local epochs, producing highly specialized local models that diverge wildly from each other.

2. **Quantization amplifies drift**: After local training, client updates are quantized to 4-8 bits (depending on memory caps). The quantization error is proportional to the **magnitude** of the weight deltas. When ResNet18GN overfits aggressively, its deltas are much larger than SimpleCNN's, meaning quantization noise scales up proportionally — a compounding effect absent in FedMAQ-Lite.

3. **Distillation ensemble poisoning**: The server-side KD ensemble receives teachers that are now heavily drifted ResNet18GN models. Even with entropy-weighted soft-voting, the blended teacher signal may be near-uniform (Jensen's inequality on entropy of mixtures), degrading the KD refinement step that is critical for FedMAQ's convergence.

FedMAQ-Lite avoids this because SimpleCNN simply lacks the capacity to overfit as aggressively — it acts as an implicit regularizer.

## Proposed Regularization Strategies

Based on the literature survey in [client-regularization.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/client-regularization.md) (Salman et al. 2025) and the existing codebase architecture, we propose implementing **two complementary strategies** that attack the problem from different angles:

### Strategy 1: Client-Side KD Regularization (FedGKD-style) — _Recommended Primary_

> [!IMPORTANT]
> This is the highest-priority strategy. It operates in **logit/representation space**, which is naturally symmetric with FedMAQ's existing server-side KD pipeline.

**Mechanism**: During local training, each client minimizes a blended loss that combines the standard CE task loss with a KL divergence term penalizing the local model's predictions from diverging too far from the incoming global model's predictions:

$$\mathcal{L}_{\text{client}} = (1 - \alpha_{\text{reg}}) \cdot \mathcal{L}_{\text{CE}}(y, y_{\text{true}}) + \alpha_{\text{reg}} \cdot D_{\text{KL}}\left(\sigma\left(\frac{z_{\text{global}}}{T_{\text{reg}}}\right) \,\middle\|\, \sigma\left(\frac{z_{\text{local}}}{T_{\text{reg}}}\right)\right)$$

**Why this is the right fit for FedMAQ:**

- Constrains the _output distribution_ of the local model, directly reducing the teacher disagreement that poisons server-side KD
- Does NOT add communication cost — the global model is already available on-device
- Naturally complements the server-side soft-voting: client-side KD regularization reduces _input noise_ to the ensemble, while server-side soft-voting reduces the _impact_ of remaining noise
- More principled than FedProx's weight-space L2 penalty for KD-based FL: constraining logit distributions preserves the model's ability to learn new local features while preventing representational drift

### Strategy 2: FedProx-style Proximal Regularization — _Re-evaluated for Stacking_

> [!TIP]
> Based on the July 15 sweep results, high-capacity ResNet18GN under severe skew ($\alpha=0.1$) suffers from parameter-space drift that logit-space KD regularization alone cannot fully anchor. Stacking a FedProx-style weight-space L2 penalty ($(\mu/2) \|w - w_{\text{global}}\|^2$) on top of KD regularization is recommended to prevent optimization valley divergence (averaging collapse) while KD regularization stabilizes representation quality.

---

## Proposed Changes

### Component 1: Client-Side KD Loss Hook

#### [NEW] [kd_loss_hook.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/client_hooks/kd_loss_hook.py)

New `ClientKDLossHook` class in a dedicated module. This follows the same `LossHook` pattern as the existing `FedProxLossHook` in [client.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/client.py):

- **`on_train_begin(model)`**: Saves a frozen copy of the global model's state (like FedProx saves global params, but here we save the full model for inference)
- **`compute_loss(model, outputs, targets, criterion)`**: Runs the frozen global model forward on the same batch, computes KL divergence between local and global softmax outputs, and returns the blended loss
- Parameters: `kd_reg_alpha` (blending weight, default 0.5), `kd_reg_temp` (softmax temperature for KD, default 2.0)

---

#### [MODIFY] [client.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/client.py)

Update `get_loss_hook()` factory to dispatch `ClientKDLossHook` when `algorithm.client_kd_reg == true` for `fedmaq`/`fedmaq_lite`:

```python
def get_loss_hook(alg_name: str, alg_cfg: dict[str, Any]) -> LossHook:
    if alg_name == "fedprox":
        return FedProxLossHook(mu=float(alg_cfg.get("mu", 0.01)))
    if alg_name in {"fedmaq", "fedmaq_lite"}:
        if alg_cfg.get("client_kd_reg", False):
            return ClientKDLossHook(
                alpha=float(alg_cfg.get("kd_reg_alpha", 0.5)),
                temperature=float(alg_cfg.get("kd_reg_temp", 2.0)),
            )
    return LossHook()
```

---

### Component 2: FedMAQ Config Updates

#### [MODIFY] [fedmaq.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedmaq.yaml)

Add client-side regularization parameters (disabled by default so existing runs are unaffected):

```yaml
# --- Client-Side Regularization ---
client_kd_reg: false # Enable client-side KD regularization (FedGKD-style)
kd_reg_alpha: 0.5 # Blending weight: (1-alpha)*CE + alpha*KD
kd_reg_temp: 2.0 # Softmax temperature for client-side KD
```

#### [MODIFY] [fedmaq_lite.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedmaq_lite.yaml)

Same additions as fedmaq.yaml for config consistency (disabled by default; this sweep focuses on FedMAQ only).

---

### Component 3: No Changes Needed

The following components require **no modifications**:

- **`StandardFit` / `FedMAQFit`**: Already calls `client.loss_hook.compute_loss(...)` and `client.loss_hook.on_train_begin(...)` in the training loop — the hook pattern handles everything transparently
- **Strategy hooks**: Server-side logic is completely decoupled from client-side loss computation
- **Model dispatch**: `get_client_model()` already correctly dispatches ResNet18GN for `fedmaq` and SimpleCNN for `fedmaq_lite`
- **Quantization/compression**: Operates on deltas after training, unaffected by the loss function used

---

## Resolved Design Decisions

| Decision              | Resolution                                                                                                    |
| :-------------------- | :------------------------------------------------------------------------------------------------------------ |
| KD reg vs FedProx     | **KD reg only** — logit-space regularization is more principled for KD-based FL and keeps the narrative clean |
| kd_reg_temp sweep     | **Include in sweep**: `kd_reg_temp ∈ {1.0, 2.0}`                                                              |
| FedMAQ-Lite           | **Deferred** — existing results already beat FedProx; focus on ResNet18GN                                     |
| ema_decay α=1.0 fix   | **Folded into this sweep** — use `ema_decay=0.1` for all α=1.0 runs                                           |
| Stacking KD + FedProx | **Not implemented** — revisit only if KD reg alone is insufficient                                            |

## Sweep Grid (18 runs)

| Arm                          | `client_kd_reg` |    `kd_reg_alpha`    | `kd_reg_temp` | Configs |
| :--------------------------- | :-------------: | :------------------: | :-----------: | :-----: |
| Baseline (no reg)            |     `false`     |          —           |       —       |    1    |
| KD reg sweep                 |     `true`      | {0.1, 0.3, 0.5, 0.7} |  {1.0, 2.0}   |    8    |
| **Total per alpha**          |                 |                      |               |  **9**  |
| **Grand total** (× 2 alphas) |                 |                      |               | **18**  |

Per-alpha overrides:

- **α=0.1**: `ema_decay=0.7`, `entropy_weight=4.0`, `precision_weight=1.0`
- **α=1.0**: `ema_decay=0.1`, `entropy_weight=2.0`, `precision_weight=0.5`

---

## Verification Plan

### Automated Tests

- Run existing unit tests: `uv run pytest tests/ -x` to verify no regressions from the new hook
- Verify the new `ClientKDLossHook` works correctly in isolation with a quick CI-config run: `uv run python scripts/run.py experiment=ci algorithm=fedmaq algorithm.client_kd_reg=true`

### Empirical Validation

- Run the 18-config regularization sweep at 40 rounds
- Compare FedMAQ (ResNet18GN) accuracy with each regularization setting against the 38.36% baseline (α=0.1)
- Record communication footprint to confirm regularization doesn't change bytes transmitted (it shouldn't — only client-side loss changes)

### Success Criteria

- FedMAQ (ResNet18GN) at α=0.1 with best regularization should **close the gap** toward FedMAQ-Lite's 52.83% or at minimum beat FedProx (49.71%)
- No regression in α=1.0 performance
- Communication footprint remains unchanged
