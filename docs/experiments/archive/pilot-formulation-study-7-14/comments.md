# Pilot Formulation Study: Improvement Alignment Analysis

This analysis is grounded in the raw per-round training logs from [multirun/2026-07-14/11-45-55-pilot/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/multirun/2026-07-14/11-45-55-pilot), the [results.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/experiments/pilot-formulation-study-7-14/results.md), and the [smoke-test baselines](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/experiments/smoke-test-7-13/results.md).

---

## 1. What the Training Data Actually Shows

### Formulation 3 (Gradient-Primary) — The Selected Winner

#### α=0.1 (Severe Non-IID)

```
Final: 40.60% | Peak: 45.69% (round 26) | Loss: 1.682
Early(1-10): 28.51% avg → Mid(11-25): 35.99% → Late(26-40): 36.75%
Late volatility: σ = 0.0436 (HIGH — swings of ±4%)
Late loss trend: -0.0824 (still improving)
Comm/round: 98.7–107.8 MB (CV=0.019, very stable)
```

#### α=1.0 (Moderate Non-IID)

```
Final: 55.42% | Peak: 58.65% (round 26) | Loss: 1.258
Early(1-10): 49.10% avg → Mid(11-25): 54.63% → Late(26-40): 56.50%
Late volatility: σ = 0.0153 (low)
Late loss trend: +0.006 (plateau)
Comm/round: 104.6–110.0 MB (CV=0.012, extremely stable)
```

### Key Empirical Observations

| Observation                                                      | Evidence                                                                                                              | Implication                                                                                            |
| :--------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **Accuracy gap to uncompressed baselines is large**              | F3@α=0.1: 40.6% vs FedProx 49.7% (−9.1pp). F3@α=1.0: 55.4% vs FedAvg 67.6% (−12.2pp)                                  | The distillation ensemble quality is the primary bottleneck, not the communication path                |
| **Late-round accuracy oscillates significantly under α=0.1**     | F3 last 10 rounds: 0.40, 0.39, 0.40, 0.31, 0.35, 0.39, 0.39, 0.30, 0.35, 0.41 (σ=0.044)                               | Knowledge distillation is noisy — the ensemble is polluted by low-quality teachers                     |
| **Communication is already very stable**                         | All F3 rounds produce ~103–107 MB (CV < 2%). The dynamic normalization (`g_k/g_max`) prevents late-stage decay        | Communication scheduling/tuning is a lower priority — the current normalization works                  |
| **Gradient norms remain healthy and diverse**                    | F3 late rounds: `tilde_g` ranges from 0.50 to 1.00 across clients, with most clients assigned q=5–8                   | The soft quality signal is working; the issue is downstream in distillation quality                    |
| **F4 collapsed via gradient-norm annihilation**                  | After round ~8, all F4 clients show `g_k=0.0000`, `tilde_g=1.0000` — the model died and gradients vanished            | Threshold-based rules amplify catastrophic failure; continuous formulations (F3) are resilient         |
| **F2 (Multiplicative) shows rising late-stage loss under α=1.0** | Late loss trend: +0.063 (rising) despite decent accuracy (54.78%)                                                     | Multiplicative coupling can create instability; F3's additive modulator is more robust                 |
| **Peak accuracy often occurs mid-training, then degrades**       | F3@α=0.1 peaks at round 26 (45.69%) but ends at 40.60% (−5pp). F0@α=0.1 peaks at round 30 (43.46%) but ends at 40.10% | Late-round distillation is erasing earlier gains — the ensemble is harming convergence in later rounds |

---

## 2. Alignment Assessment of Each Proposed Concept

### Concept 1: Quantization-Aware Soft-Voting — ✅ Strongly Aligned

This is the **single most impactful** improvement suggested. The training data directly validates the need for it.

**Evidence:**

- The −9.1pp gap (α=0.1) and −12.2pp gap (α=1.0) to uncompressed baselines are almost entirely attributable to noisy distillation. FedMAQ's communication reduction is already 8.8–9.6x — the compression itself is not the problem; the _quality_ of compressed signals entering the ensemble is.
- Under α=0.1, F3's client bit-width assignments range from q=4 to q=8. A 4-bit client's logit predictions are inherently noisier, yet they currently receive **equal weight** in the ensemble average at `kd_utils.py:L60`.
- The high late-round accuracy volatility (σ=0.044) is a direct symptom: when a round's 10 selected clients happen to include more low-precision or non-expert teachers, the distillation quality drops, causing 5–10pp swings.
- The peak-then-degrade pattern (peaks at round 26, regresses to round 40) suggests the ensemble is actively overwriting good knowledge with noisy knowledge in late rounds.

**Expected impact:** Moderate-to-high. Weighting by precision and entropy would filter out the worst teacher contributions per-sample, smoothing the accuracy trajectory and likely closing 3–5pp of the accuracy gap.

---

### Concept 2: CFD-style Downlink Compression — ⚠️ Weakly Aligned

**Evidence:**

- Communication is already 8.8–9.6x reduced vs FedAvg. The per-round communication cost of ~104 MB (F3) is dominated by the **uplink** (client→server quantized deltas), not the downlink model broadcast.
- The raw model download size for the SimpleCNN/TinyCNN student is relatively small — this isn't a ResNet50-class model. Compressing a small model yields diminishing returns.
- No run in the pilot study shows communication as the bottleneck; the latency bottleneck is client-side training time (80–91s per round vs 15s server KD time).

**Verdict:** This is a valid thesis contribution for "doubly communication-efficient" framing, but **will not improve accuracy** and addresses a non-critical bottleneck. Defer until the accuracy gap is narrowed.

---

### Concept 3: Memory-Quantized/Sparse Error Feedback — ⚠️ Weakly Aligned (for accuracy), Thesis-Relevant

**Evidence:**

- The EF mechanism is working correctly — communication per round is extremely stable (CV < 2%), confirming the residual compensation is well-behaved.
- No run shows memory pressure as a limiting factor in the pilot study (these are simulated clients).
- Compressing the EF residual may _slightly hurt_ accuracy by adding a second layer of quantization noise, which is the opposite of what we need given the existing accuracy gap.

**Verdict:** Valid for thesis novelty ("resource-constrained EF") but unlikely to help accuracy and may slightly hurt it. Implement as an **ablation experiment** rather than a default-on feature.

---

### Concept 4: Convergence-State Tuning (λ Scheduling) — ⚠️ Partially Aligned, But Low Priority

**Evidence:**

- The training data shows that F3's communication per round is already very stable (CV < 2%) throughout all 40 rounds. The `g_k/g_max` normalization is doing its job — `tilde_g` values remain well-distributed (0.50–1.00) even in late rounds.
- Under α=1.0, F3 reaches a clear loss/accuracy plateau by round ~25. Under α=0.1, loss is still slowly improving at round 40 (loss trend: −0.08). In neither case is λ scheduling the bottleneck.
- The data-modulator `(1 + λ·ñ_k)/(1 + λ)` in F3 already has a mild effect (≤15% modulation) — scheduling λ to decay would further diminish the data signal, but the data signal isn't the problem.

**Verdict:** This is a fine theoretical refinement but the training data shows **no pathological behavior** that λ scheduling would fix. The accuracy gap comes from distillation noise, not from the quantization bit-width assignment formula.

---

## 3. Additional Recommendations From Data

Based on the empirical evidence, two improvements not in the original HANDOFF document are directly supported by the training data:

### Recommendation A: Ensemble Momentum / EMA Student

This directly addresses the peak-then-degrade pattern visible in the training data.

**Problem observed:** F3@α=0.1 peaks at 45.69% (round 26) but finishes at 40.60%. The student model is being overwritten every round by a fresh distillation pass that can be worse than the previous state.

**Proposal:** Maintain an Exponential Moving Average (EMA) of the student model parameters across rounds:

```python
# After each round's KD step:
ema_params = alpha * ema_params + (1 - alpha) * new_student_params
```

Use the EMA model for evaluation and as the starting point for next round's distillation student. This smooths out round-to-round volatility from noisy teacher ensembles.

**Difficulty:** Easy. Add ~15 lines to `FedMAQHook.aggregate_fit` to maintain and apply EMA parameters. No cross-module changes needed.

**Expected impact:** Could recover 2–4pp by preventing good knowledge from being overwritten by noisy late-round distillation.

---

### Recommendation B: Gradient Norm Smoothing (Per-Client EMA)

**Problem observed:** The per-round gradient norm `g_k` is computed from a **single batch** of client data (see `fedmaq.py:L195–207`). Under α=0.1, where client data partitions can be as small as 5–17 samples, a single stochastic batch yields a very noisy estimate of the true gradient magnitude.

This means the same client can get q=5 one round and q=8 the next, purely from mini-batch sampling noise — not from any meaningful change in training state.

**Proposal:** Maintain a per-client EMA of gradient norms across rounds on the server:

```python
ema_g_k[pid] = beta * ema_g_k[pid] + (1 - beta) * g_k_current
```

Use `ema_g_k` instead of raw `g_k` for the Tier 2 soft quality computation. This stabilizes the bit-width assignments without changing the formulation.

**Difficulty:** Easy. Add a `dict[int, float]` to `FedMAQHook.__init__` and update it in `configure_fit`. ~10 lines of code.

**Expected impact:** Reduces round-to-round quantization jitter, which in turn stabilizes the distillation ensemble quality. Indirectly helps with the accuracy volatility problem.

---

## 4. Revised Priority Ranking

Based on the training data evidence:

| Priority | Improvement                          | Why                                                                                                                      |
| :------: | :----------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
|  **1**   | **Soft-Voting** (Concept 1)          | Directly targets the #1 bottleneck: noisy distillation ensemble. Data shows 5–10pp swings from teacher quality variation |
|  **2**   | **EMA Student** (Rec. A)             | Prevents peak-then-degrade pattern. F3 loses 5pp between peak and final accuracy under α=0.1                             |
|  **3**   | **Gradient Norm Smoothing** (Rec. B) | Stabilizes bit-width assignments to reduce upstream noise that feeds into distillation                                   |
|  **4**   | **λ Scheduling** (Concept 4)         | Low-risk refinement, but data shows no pathological behavior it would fix                                                |
|  **5**   | **Memory-Quantized EF** (Concept 3)  | Thesis novelty value, but may slightly hurt accuracy                                                                     |
|  **6**   | **Downlink Compression** (Concept 2) | Valid framing contribution but addresses a non-bottleneck                                                                |

NOTE: Priorities 1–3 all target the same root cause: **distillation ensemble quality under severe heterogeneity**. They are complementary and could be implemented together in a single "distillation robustness" improvement batch.
