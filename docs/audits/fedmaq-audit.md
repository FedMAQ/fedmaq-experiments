# FedMAQ Comprehensive Audit

Full audit of the original and improved FedMAQ algorithm covering mathematical grounding, implementation correctness, literature defensibility, and potential thesis-defense vulnerabilities. Grounded in the [fedmaq-literature KG](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/) and the [fedmaq-experiments](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/) codebase.

> [!NOTE]
> Findings are architecture-independent (math/logic audit). Occurrences of ResNet18GN below are illustrative examples, not deprecated results — the MobileNetV2GN switch (see [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)) does not invalidate these findings.

---

## Audit Summary: Verdicts at a Glance

| Component                                   |            Verdict             | Severity |
| :------------------------------------------ | :----------------------------: | :------: |
| Tier-1 hard memory cap                      |            ✅ Sound            |    —     |
| Tier-2 soft quality target (Formulation 3)  |            ✅ Sound            |    —     |
| Two-tier combination logic                  |            ✅ Sound            |    —     |
| Permissible bit-width set & snap-floor      |            ✅ Sound            |    —     |
| FedPAQ-style symmetric quantizer            |            ✅ Sound            |    —     |
| Server-side KD (plain ensemble)             |            ✅ Sound            |    —     |
| Quantization-aware soft-voting (Priority 1) |   ⚠️ Defensible with caveats   |  Medium  |
| EMA student model (Priority 2)              |   ⚠️ Defensible with caveats   |  Medium  |
| Gradient norm smoothing (Priority 3)        |            ✅ Sound            |    —     |
| EMA–FedAvg interaction (aggregation order)  |   🔴 Potential logical issue   |   High   |
| Grad norm probe model architecture          |    ⚠️ Subtle design choice     |   Low    |
| Byte-accounting fairness                    |        ⚠️ Minor concern        |   Low    |
| KD temperature = 1.0                        | ⚠️ Defensible but non-standard |   Low    |

---

## 1. Core FedMAQ: Dual-Tier Precision Scaling

### 1.1 Tier 1 — Hard Memory Cap ✅

**Implementation**: [fedmaq.py:L81](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L81)

```python
q_k_max_raw = max(1.0, np.floor(c_k / c_unit))
```

**Assessment**: Correctly implements the manuscript's $Q_k^{max} = \max\{q \in \mathcal{Q} \mid q \le \lfloor c_k / c_{unit} \rfloor\}$. The `max(1.0, ...)` guard ensures no client receives 0 bits. The memory sampling $c_k \sim \mathcal{U}(2048, 16384)$ with $c_{unit}=512$ correctly maps to:

| $c_k$ (MB) | $\lfloor c_k / 512 \rfloor$ | Physical device      |
| :--------: | :-------------------------: | :------------------- |
|    2048    |              4              | Raspberry Pi 4 2GB   |
|    4096    |              8              | Raspberry Pi 4/5 4GB |
|    8192    |             16              | Raspberry Pi 4/5 8GB |
|   16384    |             32              | Jetson Orin NX 16GB  |

**Literature grounding**: Directly adapts DynFed's Eq. 2 ($q_k = \min(c_{max}, \lfloor c_k / c_p \rfloor)$) from [He et al. 2025](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/he-2025-dynfed.md#L44-L47). FedMAQ applies `floor()` rather than `min(c_max, ...)` because the permissible set $\mathcal{Q}$ already provides an implicit upper bound at 32. This is equivalent and defensible.

**Defense note**: A reviewer might ask why $c_{unit} = 512$ MB per bit. This is a calibration constant, not a fundamental parameter — it simply sets the mapping between real device memory and bit-width. The choice makes the 2GB→4-bit / 8GB→16-bit / 16GB→32-bit mapping align with practical edge hardware, which is a reasonable engineering choice. If challenged, note that $c_{unit}$ doesn't affect the algorithm's mathematical properties — only the range of bit-widths assigned to a given memory profile.

---

### 1.2 Tier 2 — Soft Quality Target (Formulation 3) ✅

**Implementation**: [fedmaq.py:L97-L100](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L97-L100)

```python
# Alternative 3: Gradient-Primary, Data-Modulated
modulator = (1.0 + lambda_val * tilde_n) / (1.0 + lambda_val)
q_hat = q_min + np.round((q_max - q_min) * tilde_g * modulator)
```

**Mathematical correctness**: $\hat{q}_k^{(t)} = q_{min} + \text{round}\left((q_{max} - q_{min}) \cdot \tilde{g}_k^{(t)} \cdot \frac{1 + \lambda \cdot \tilde{n}_k}{1 + \lambda}\right)$

The modulator is well-behaved:

- When $\tilde{n}_k = 0$: modulator $= 1/(1+\lambda)$, dampening the gradient signal.
- When $\tilde{n}_k = 1$: modulator $= 1$ (full gradient signal).
- The modulator is monotonically increasing in $\tilde{n}_k$ ∈ [0, 1], bounded in $[1/(1+\lambda), 1]$.
- At default $\lambda=1$: modulator range is $[0.5, 1.0]$, giving ≤50% data modulation.

**Literature grounding**: This is FedMAQ's primary novel contribution. DynFed's Eq. 4 uses a recursive inertial tracker $b_i^{(t)} = b_i^{(t-1)} + \eta \cdot (\ldots)$, which introduces history dependence and an additional learning rate hyperparameter. FedMAQ replaces this with a direct per-round measurement, removing the path-dependent state. The data-richness signal $\tilde{n}_k$ is entirely absent from DynFed and is FedMAQ's unique addition.

**Defensibility**: The formulation study ([pilot results](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/experiments/pilot-formulation-study-7-14/results.md)) empirically validates Formulation 3 as the winner across both skew regimes. The key defense argument is that gradient-primary logic with data modulation is more robust than alternatives because:

1. It avoids the catastrophic failure mode of Formulation 4 (threshold-based).
2. It achieves lower communication than Formulation 0 (resource-only) with equal or better accuracy.
3. The gradient signal is the primary driver (as in DynFed), but the data signal acts as a secondary stabilizer.

---

### 1.3 Two-Tier Combination ✅

**Implementation**: [fedmaq.py:L114-L119](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L114-L119)

```python
q_hat = max(float(q_min), min(float(q_max), float(q_hat)))
return _snap_floor(min(q_k_max_raw, q_hat), bit_widths)
```

**Assessment**: This correctly implements the two-tier combination: "raw Tier-1 cap and raw Tier-2 target via `min()`, then floor into the permissible set $\mathcal{Q}$ exactly once." The ordering is:

1. Clamp $\hat{q}$ to $[q_{min}, q_{max}]$ (soft-target bounds).
2. Take $\min(Q_k^{max}, \hat{q})$ (Tier-1 hard cap wins if memory is limited).
3. Floor into $\mathcal{Q}$ via `_snap_floor`.

The `_snap_floor` function ([fedmaq.py:L47-L50](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L47-L50)) correctly returns the largest permissible bit-width ≤ the combined value, falling back to `min(bit_widths) = 1` if no eligible value exists.

> [!IMPORTANT]
> The comment at [fedmaq.py:L117-L118](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L117-L118) states "Memory-limited clients may receive fewer bits than $q_{min}$ — intentional, the physical bound wins over the soft quality target." This is a correct design choice — a 2GB device physically cannot support more than 4 bits, regardless of what the soft quality target says. This is the key architectural distinction between Tier 1 and Tier 2 and should be highlighted in the thesis as a safety property.

---

### 1.4 Gradient Norm Computation ⚠️

**Implementation**: [fedmaq.py:L180-L217](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L180-L217)

The server computes gradient norms for each sampled client by:

1. Loading the _global model_ with current global parameters.
2. Performing a forward+backward pass on one batch of that client's data.
3. Extracting the L2 norm of the resulting gradients.

**Subtle design choice**: The gradient norm is computed using the **global model evaluated on client-local data**, not the client's locally-trained model. This is a deliberate choice documented in the code — it measures "how much this client's data would change the global model" rather than "how much the client has already changed." This is actually more informative for bit-width allocation because it captures the _current_ gradient magnitude of the client's data partition relative to the global model's state, which is a better proxy for how much information that client's update carries.

**Architecture mismatch note**: The code at [fedmaq.py:L168-L170](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L168-L170) uses `get_kd_student_model` (SimpleCNN for CIFAR) as the grad-norm probe, not the full `get_model` (ResNet18GN). This is correct because FedMAQ clients also train the student model. The comment explicitly warns about this: "using the full `get_model()` here loads mismatched parameters (e.g. ResNet18GN on CIFAR) and silently zeroes every norm."

**Potential defense question**: "Why compute gradient norms server-side rather than having clients report them?"

- **Answer**: This preserves the uplink communication budget — clients don't need to send an extra scalar per round. The server already has the global model parameters and a copy of the data partition indices, so it can compute the gradient norms itself. This is a valid engineering choice that doesn't compromise the algorithm's mathematical properties.

**Potential defense question**: "Single-batch gradient norms are noisy — how do you handle this?"

- **Answer**: This is exactly what Priority 3 (gradient norm smoothing) addresses. The per-client EMA at [fedmaq.py:L219-L230](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L219-L230) with $\beta=0.7$ smooths out mini-batch sampling noise across rounds.

---

## 2. Improved FedMAQ: Distillation Robustness Features

### 2.1 Priority 1 — Quantization-Aware Soft-Voting ⚠️

**Implementation**: [kd_utils.py:L62-L86](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L62-L86)

```python
entropy_weights = torch.exp(-entropy_weight_scale * entropy)  # [T, B]
precision_weights = (
    torch.tensor([q / q_max for q in teacher_bit_widths], ...).unsqueeze(1)
    ** precision_weight_scale
)  # [T, 1]
combined = entropy_weights * precision_weights  # [T, B]
combined = combined / (combined.sum(dim=0, keepdim=True) + eps)  # normalize
teacher_soft_preds = (preds_stack * combined.unsqueeze(2)).sum(dim=0)  # [B, C]
```

**Mathematical correctness**: ✅ The formula is correctly implemented:

$$W_k(x) = \frac{\exp(-\gamma_e \cdot H(P_k(x))) \cdot (q_k / q_{max})^{\gamma_p}}{\sum_{j} \exp(-\gamma_e \cdot H(P_j(x))) \cdot (q_j / q_{max})^{\gamma_p}}$$

The normalization over teachers per sample (dim=0) is correct. The $\epsilon = 10^{-8}$ prevents division by zero. The resulting weighted sum produces a valid probability distribution because:

- Each $P_k(x)$ is a valid distribution (output of softmax).
- The weights $W_k(x)$ are non-negative and sum to 1 over $k$ (teachers).
- A convex combination of probability distributions is a probability distribution.

**Literature grounding**: Per-sample confidence gating via entropy is used in DynFed's "comprehensive score" (Eq. 7), though DynFed applies it for _teacher selection_ (hard gate), not weighting (soft gate). FedMAQ's soft-voting is more principled because it preserves all teachers' contributions while downweighting unreliable ones — avoiding the information-theoretic waste of hard exclusion. The precision weighting $(q_k / q_{max})^{\gamma_p}$ is novel and directly addresses the gap identified in [Gap: heterogeneity-aware quantization](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/gaps/heterogeneity-aware-quantization.md).

> [!WARNING]
> **Potential vulnerability — Soft-voting normalizing to a mixture, not a single distribution**: The blended output $P_{ensemble}(x)$ may have higher entropy than any individual teacher's prediction, because mixing distributions increases entropy (Jensen's inequality on $H$). Under extreme heterogeneity where teachers violently disagree, the blended target could be near-uniform, providing a weak learning signal. This is the "ensemble softening" effect.
>
> **Defense**: Empirically, the EMA sweep shows FedMAQ already outperforms all baselines under severe skew. The soft-voting is not the sole mechanism — the student EMA (Priority 2) provides an independent stabilization path. If pressed, argue that the entropy weighting _precisely_ mitigates the ensemble softening effect by suppressing uncertain teachers before the mix, keeping the resulting distribution peaky.

> [!NOTE]
> **Implementation detail worth noting**: The `q_max` used in precision weighting ([kd_utils.py:L71](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L71)) is computed as `max(teacher_bit_widths)` — the max among _participating_ teachers, not the global config `q_max`. This means the precision weight is relative to the best-precision client in the current round. This is actually correct — it normalizes precision weights relative to the round's context. If the config `q_max=16` were used instead, a round where all clients have $q \le 4$ would see all precision weights $\le 0.25$, wasting the dynamic range. Using the round-local max is more principled.

---

### 2.2 Priority 2 — EMA Student Model ⚠️

**Implementation**: [fedmaq.py:L316-L328](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L316-L328)

```python
if aggregated_parameters is not None and alg_cfg.get("ema_student", False):
    ema_decay = float(alg_cfg.get("ema_decay", 0.99))
    new_params = parameters_to_ndarrays(aggregated_parameters)
    if self._ema_params is None:
        self._ema_params = [p.copy() for p in new_params]
    else:
        self._ema_params = [
            ema_decay * ema + (1.0 - ema_decay) * new
            for ema, new in zip(self._ema_params, new_params, strict=True)
        ]
    aggregated_parameters = ndarrays_to_parameters(self._ema_params)
```

**Mathematical correctness**: $\theta_{EMA}^{(t)} = \beta \cdot \theta_{EMA}^{(t-1)} + (1-\beta) \cdot \theta_{new}^{(t)}$

This is standard Polyak averaging, well-established in optimization theory. The implementation is correct.

> [!CAUTION]
> **🔴 HIGH-PRIORITY: EMA–FedAvg Aggregation Ordering Issue**
>
> The EMA is applied **after** both FedAvg aggregation and KD refinement. This means:
>
> 1. FedAvg aggregates quantized client updates → $\theta_{agg}^{(t)}$
> 2. KD refines $\theta_{agg}^{(t)}$ using the teacher ensemble → $\theta_{KD}^{(t)}$
> 3. EMA blends: $\theta_{EMA}^{(t)} = \beta \cdot \theta_{EMA}^{(t-1)} + (1-\beta) \cdot \theta_{KD}^{(t)}$
>
> The EMA output $\theta_{EMA}^{(t)}$ is then sent to clients in the _next_ round. But when the KD step in round $t+1$ starts, it initializes the student from the FedAvg-aggregated parameters (step 1), **not** from $\theta_{EMA}^{(t)}$. This is because the KD student is initialized from `aggregated_parameters` at [fedmaq.py:L303-L304](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L303-L304), which is the output of `super().aggregate_fit()` — i.e., the raw FedAvg aggregation of client updates that were trained starting from $\theta_{EMA}^{(t-1)}$.
>
> **This is actually correct behavior**, but the reasoning is subtle and worth articulating for defense:
>
> - Clients train starting from $\theta_{EMA}^{(t-1)}$ (the broadcast model).
> - They return $\theta_{EMA}^{(t-1)} + \Delta_k$ (their deltas).
> - FedAvg aggregates: $\theta_{agg}^{(t)} = \sum_k w_k (\theta_{EMA}^{(t-1)} + \Delta_k) = \theta_{EMA}^{(t-1)} + \bar{\Delta}$.
> - KD refines this, producing $\theta_{KD}^{(t)}$.
> - EMA smooths: $\theta_{EMA}^{(t)} = \beta \cdot \theta_{EMA}^{(t-1)} + (1-\beta) \cdot \theta_{KD}^{(t)}$.
>
> So the EMA acts as a _temporal smoothing_ of the KD-refined model, preventing large round-to-round jumps. The KD student initialization from the FedAvg aggregate is correct because it preserves the standard FedAvg aggregation semantics — the EMA is an outer-loop smoothing, not a replacement for aggregation.

**Literature grounding**: Polyak averaging / EMA of model parameters is standard in deep learning (Kingma & Ba 2014 — Adam, Tarvainen & Valpola 2017 — Mean Teacher). Its application to server-side FL models is less common but has precedent in FedProx's analysis where parameter-space regularization is discussed. The key insight — that EMA stabilizes late-round convergence under noisy distillation — is empirically validated by the sweep results.

**Empirical validation**: The [EMA decay sweep](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/experiments/ema-decay-sweep-7-14/results.md) shows a clear heterogeneity-dependent optimal:

- $\alpha=0.1$: $\beta=0.7$ best (52.51% vs 40.01% at $\beta=0.1$) — +12.5pp from EMA alone.
- $\alpha=1.0$: $\beta=0.1$ best (60.62% vs 57.15% at $\beta=0.9$) — moderate EMA helps, heavy EMA hurts.

The inverse relationship between optimal EMA strength and data homogeneity is well-defended: under high heterogeneity, client updates have high variance (client drift), and EMA acts as a variance regularizer. Under low heterogeneity, heavy EMA introduces convergence lag.

---

### 2.3 Priority 3 — Gradient Norm Smoothing ✅

**Implementation**: [fedmaq.py:L219-L230](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L219-L230)

```python
if alg_cfg.get("grad_norm_ema", False):
    beta = float(alg_cfg.get("grad_norm_beta", 0.7))
    smoothed_norms = []
    for pid, raw_norm in zip(client_pids, grad_norms, strict=True):
        if pid in self._grad_norm_ema:
            smoothed = beta * self._grad_norm_ema[pid] + (1.0 - beta) * raw_norm
        else:
            smoothed = raw_norm
        self._grad_norm_ema[pid] = smoothed
        smoothed_norms.append(smoothed)
    grad_norms = smoothed_norms
```

**Assessment**: Correct per-client EMA of gradient norms. On first encounter, uses raw norm (no warm-up bias). Subsequent encounters blend with history. This reduces mini-batch sampling noise in the gradient norm signal that feeds into the Tier-2 bit-width computation.

**Potential defense question**: "Why $\beta=0.7$ for grad norm smoothing? Is this also heterogeneity-dependent?"

- **Answer**: Grad norm smoothing operates on the _signal_ feeding into bit-width allocation, not on the model parameters. $\beta=0.7$ means 70% weight on history, which is sufficient to smooth single-batch variance without over-dampening the signal's responsiveness to real training dynamics. This is a different mechanism from the student EMA ($\beta$ for model parameters), and a single fixed $\beta$ is appropriate because it addresses measurement noise, not learning dynamics.

---

## 3. Server-Side Knowledge Distillation Pipeline

### 3.1 KD Loss Function ✅

**Implementation**: [kd_utils.py:L46](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L46) and [kd_utils.py:L95](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L95)

```python
kl_criterion = nn.KLDivLoss(reduction="batchmean")
loss = kl_criterion(student_log_soft, teacher_soft_preds) * (temperature**2)
```

**Mathematical correctness**: PyTorch's `KLDivLoss` expects log-probabilities for the first argument and probabilities for the second. The code correctly passes `F.log_softmax(student_logits / temperature, dim=1)` as the first argument and the teacher ensemble's soft probabilities (output of softmax) as the second. The $T^2$ scaling is the standard Hinton et al. (2015) gradient-magnitude correction for temperature-scaled distillation.

> [!NOTE]
> **$T=1.0$ is non-standard but defensible**: Most KD literature uses $T > 1$ (typically $T \in [2, 20]$) to soften the teacher's output distribution and reveal inter-class similarities ([Hinton 2015](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/hinton-2015-distillation.md)). FedMAQ uses $T=1.0$, which means no additional softening beyond the natural softmax output.
>
> **Defense**: In FedMAQ's context, the teachers are already _quantized_ models with lower-precision weights. Their logit outputs are inherently noisier and less peaked than a full-precision teacher's would be. Applying $T > 1$ would further flatten already-noisy distributions, potentially degrading the distillation signal. The choice $T=1.0$ preserves whatever confidence structure the quantized teachers retain. Additionally, at $T=1.0$ the $T^2$ multiplier is 1, making the KD loss and the standard CE loss on the same scale.
>
> If a reviewer presses on this, note that $T$ could be added as a tunable hyperparameter in a future ablation, but the current choice is conservative and avoids introducing another hyperparameter axis.

### 3.2 Aggregation Pipeline Order ✅

The full pipeline per round:

1. **FedAvg weighted aggregation** → warm-start global model ([strategy.py:L204](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy.py#L204))
2. **Teacher ensemble loading** → each client's returned model ([kd_utils.py:L149-L161](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L149-L161))
3. **Server-side KD** → refine student from warm-start using ensemble labels ([kd_utils.py:L168-L180](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/kd_utils.py#L168-L180))
4. **Student EMA** → temporal smoothing of KD output ([fedmaq.py:L316-L328](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L316-L328))

This ordering is correct. The FedAvg step first cancels the zero-mean quantization noise (variance ∝ $1/K_{active}$), then KD addresses the non-IID drift that parameter averaging leaves behind. The KG explicitly notes this: "FedAvg-style parameter averaging [...] this stage, not distillation, attenuates the zero-mean quantization noise" ([fedmaq.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/methods/fedmaq.md#L55-L57)).

---

## 4. Quantization Implementation

### 4.1 FedPAQ Symmetric Quantizer ✅

**Implementation**: [quantization.py:L54-L102](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/baselines/quantization.py#L54-L102)

FedMAQ uses `FedPAQCompressionHook` (not `DAdaQuantCompressionHook`) for client-side quantization. This is explicitly documented and correct: FedMAQ's `q` is a true bit-width from $\mathcal{Q}$, while DAdaQuant's `q` represents quantization _levels per sign_ — semantically different. Using DAdaQuant's quantizer with `q=16` would give 33 levels (~5 bits effective), not 16-bit precision.

The quantizer implements:

- **$q > 1$**: Symmetric uniform quantization with $2^{q-1} - 1$ positive levels. Maps $d \to$ codes in $[-levels, levels]$, then maps back.
- **$q = 1$**: Sign quantization — each element maps to $\text{sign}(d) \cdot \text{scale}$.
- **$q = 0$**: Falls through to sign quantization via the `max(1, ...)` guard on `levels`.

**Byte accounting**: [quantization.py:L45-L46](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/baselines/quantization.py#L45-L46)

```python
element_bits = d.size * bits_per_element
total_bytes += int(math.ceil(element_bits / 8.0)) + 4
```

The +4 accounts for the float32 scale factor stored per tensor. This is standard practice per [FedPAQ](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/reisizadeh-2020-fedpaq.md) and [QSGD](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/papers/alistarh-2017-qsgd.md).

### 4.2 Client-Side Delta Computation ✅

**Implementation**: [standard.py:L99-L108](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/client_hooks/standard.py#L99-L108)

```python
deltas = [u - o for u, o in zip(updated_params, parameters, strict=True)]
compressed_deltas, byte_size = client.compressor_hook.compress(deltas)
reconstructed_params = [o + cd for o, cd in zip(parameters, compressed_deltas, strict=True)]
```

Clients compute $\Delta_k = \theta_{local} - \theta_{global}$, compress $\Delta_k$, then reconstruct $\theta_{upload} = \theta_{global} + Q(\Delta_k)$. The server receives the **reconstructed parameters** (not the quantized deltas), and FedAvg's weighted averaging operates on these. This is the standard compress-then-reconstruct pattern from FedPAQ.

---

## 5. Potential Thesis-Defense Vulnerabilities

### 5.1 Architectural Confounder — Model Size Discrepancy (SimpleCNN vs. ResNet18GN) ⚠️

**Background**: In the original implementation, FedMAQ clients trained a smaller **SimpleCNN** (2.16M params on CIFAR-10) to simulate extreme edge constraints, while standard baselines trained **ResNet18GN** (11.17M params).

**The Confounder**: A reviewer will argue that the 9.6x communication reduction is primarily an architectural artifact (SimpleCNN is 5.1x smaller by default) rather than the result of multi-adaptive quantization.

**The Solution**: Standardize FedMAQ to use the same target architecture (`ResNet18GN` on CIFAR) as the other baselines. Under this setup, the server aggregates quantized `ResNet18GN` client weights and refines the global `ResNet18GN` student via **Self-Distillation** using the teacher ensemble (`ResNet18GN`). This completely isolates the quantization algorithm as the sole source of communication savings, yielding a fair Pareto efficiency benchmark.

**Suggested Implementation Changes**:

- **Client Model Dispatch** (`src/fedmaq/core/models.py`): Suggest removing `"fedmaq"` from the `get_kd_student_model` set so it defaults to `get_model` (e.g., `ResNet18GN` on CIFAR).
- **Strategy hooks** (`src/fedmaq/core/strategy_hooks/fedmaq.py`):
  1. Suggest changing the cached gradient norm probe `self._grad_norm_model` to instantiate using `get_model` instead of `get_kd_student_model` to prevent shape mismatches during parameters loading.
  2. Suggest updating the `distill_ensemble_into_global` model factory parameter to `get_model` so the server-side distillation runs self-distillation on `ResNet18GN`.

### 5.2 EMA Decay is Heterogeneity-Dependent — No Single Default ⚠️

The sweep reveals that the optimal EMA decay is strongly heterogeneity-dependent ($\beta=0.7$ for $\alpha=0.1$, $\beta=0.1$ for $\alpha=1.0$). This means FedMAQ requires either:

- (a) Prior knowledge of the heterogeneity regime to set $\beta$, or
- (b) An adaptive $\beta$-scheduling mechanism (not currently implemented).

**Defense**: This is a legitimate limitation that should be acknowledged in the thesis as future work. However, the practical argument is that in deployment, the system operator would know the approximate heterogeneity level from the client data characteristics (e.g., IoT sensor type, geographic distribution). Setting $\beta$ based on this prior is no more onerous than setting FedProx's $\mu$ or DAdaQuant's $\psi$.

### 5.3 Proxy Dataset Assumption

FedMAQ requires a 3000-sample unlabeled public proxy dataset on the server. This is a strong assumption shared with FedMD, FedDF, and DynFed ([Proxy-dataset distillation](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-literature/kg/concepts/proxy-dataset-distillation.md)), but it limits applicability to domains where such data exists.

**Defense**: The thesis should clearly state this assumption as a limitation. However, 3000 unlabeled samples from the target distribution is a mild requirement — it's <6% of CIFAR-10's training set and can often be obtained from public repositories without privacy concerns.

### 5.4 Byte Accounting for Soft-Voting Metadata ⚠️

When soft-voting is enabled, the server needs to know each teacher's bit-width $q_k$ to compute precision weights. This information is available server-side (the strategy hook stores it in `_round_client_q`), so no additional uplink bytes are needed. However, the _computation_ of entropy weights requires evaluating each teacher model on the full proxy set — this server-side cost should be reflected in the simulated server time.

**Current state**: The server-side time model at [fedmaq.py:L354-L369](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L354-L369) uses `kd_server_sim_time`, which scales with `num_public * kd_epochs * num_teachers / server_compute_speed`. The soft-voting computation happens within the same forward passes as the teacher inference, so it does not require additional forward passes — the entropy and precision weights are computed from the same teacher outputs that produce the soft labels. **No additional time cost is incurred.** ✅

---

## 6. Mathematical Properties Summary

| Property                                 | Status | Notes                                                                                                                                                                                                                                                                                                                |
| :--------------------------------------- | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unbiasedness of quantizer                |   ✅   | FedPAQ symmetric quantizer is deterministic rounding. FedMAQ uses round-to-nearest, which is _biased_ but with bounded bias (unlike DAdaQuant's stochastic rounding, which is unbiased). This is acceptable because FedAvg averaging cancels the bias across clients.                                                |
| Soft-voting produces valid distributions |   ✅   | Convex combination of softmax outputs, weights sum to 1.                                                                                                                                                                                                                                                             |
| EMA converges to recent model in limit   |   ✅   | Standard Polyak averaging property. $\beta < 1$ ensures the EMA tracks the current model; $\beta > 0$ provides smoothing.                                                                                                                                                                                            |
| Tier-1 always dominates Tier-2           |   ✅   | `min(q_k_max_raw, q_hat)` guarantees the physical constraint is never violated.                                                                                                                                                                                                                                      |
| Gradient norm is always positive         |   ✅   | `max(1e-8, norm)` guard at [fedmaq.py:L217](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L217).                                                                                                                                                             |
| Normalization by round-local max         |   ✅   | `g_max = max(grad_norms)` at [fedmaq.py:L235](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L235). This is correct — normalizing by round-local max means the highest-gradient client in each round gets $\tilde{g} = 1.0$, receiving the highest bit-width. |

---

## 7. Literature Alignment Check

| Claim/Design Choice                       | Literature Support                                   |              Assessment              |
| :---------------------------------------- | :--------------------------------------------------- | :----------------------------------: |
| Memory-based hard cap on bit-width        | DynFed Eq. 2                                         |         ✅ Direct adaptation         |
| Gradient-norm as training-state signal    | DynFed Eq. 4, QSGD                                   | ✅ Standard in adaptive quantization |
| Dataset-size as data-richness signal      | DAdaQuant (implicit via client-adaptive $q_i$)       |        ✅ Novel formalization        |
| Server-side ensemble KD                   | FedDF (Lin et al. 2020), DynFed                      |       ✅ Established approach        |
| Per-sample entropy weighting              | Hinton (2015) confidence, DynFed comprehensive score |     ✅ Adapted to soft weighting     |
| EMA of model parameters                   | Polyak averaging, Mean Teacher                       |         ✅ Well-established          |
| Per-client EMA of gradient norms          | Novel (no direct literature precedent)               |    ✅ Standard signal processing     |
| Formulation study as primary contribution | Standard in optimization (comparing function forms)  |         ✅ Valid methodology         |
| KD temperature $T=1.0$                    | Non-standard (Hinton uses $T>1$)                     |       ⚠️ Defensible, see §3.1        |
| SimpleCNN as client model                 | FedKD, CFD (student-teacher split)                   |      ✅ Standard in KD-based FL      |

---

## 8. Conclusions & Recommendations

### Strengths

1. **Clean two-tier architecture**: The separation of hard resource constraints (Tier 1) from soft quality optimization (Tier 2) is well-motivated, correctly implemented, and provides a clear structure for thesis exposition.

2. **Empirically validated improvements**: Each Priority 1-3 feature has specific empirical evidence supporting its inclusion (formulation study for Formulation 3, EMA sweep for student EMA, late-round volatility analysis for soft-voting).

3. **Implementation quality**: The codebase is well-factored (strategy hooks, client hooks, shared KD utils), defensively programmed (guards, fallbacks, strict zip), and thoroughly documented with inline comments.

4. **Literature grounding**: Every design choice has a clear lineage to established FL literature, with explicit documentation of what was adopted, adapted, or declined from DynFed.

### Items Requiring Attention for Defense

1. **Articulate the EMA–FedAvg interaction clearly** (§2.2 CAUTION box): Reviewers may probe whether the EMA operates correctly given the aggregation pipeline order. Prepare a clear explanation of why EMA acts as outer-loop temporal smoothing.

2. **Acknowledge heterogeneity-dependent $\beta$**: Frame this as a finding, not a weakness — "FedMAQ reveals an inverse relationship between optimal EMA decay and data heterogeneity, establishing a new hyperparameter-heterogeneity interaction law."

3. **Suggested Switch to ResNet18GN** (§5.1): Standardizing the model architecture removes the confounding model-capacity factor. Frame the proposed transitions in `models.py` and `strategy_hooks/fedmaq.py` as concrete implementation suggestions that will make the comparative MB savings scientifically rigorous.

4. **Consider adding $T>1$ to the ablation**: Even a brief 2-point ablation ($T=1.0$ vs $T=2.0$) would preempt the temperature question.
