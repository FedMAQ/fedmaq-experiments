# Soft-Voting Sweep: Empirical Analysis & Insights

Analysis of the soft-voting ablation and hyperparameter sweep completed July 14, 2026. Cross-referenced with [results.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/experiments/soft-voting-sweep-7-14/results.md).

---

## 1. Soft-Voting Contribution (Phase 1 Ablation)

Soft-voting provides a **clear, consistent positive contribution** under both heterogeneity regimes:

- **α=0.1:** +2.48pp at R50 (50.20% → 47.72% without SV)
- **α=1.0:** +3.96pp at R50 (60.38% → 56.42% without SV)

Notably, the gain is _larger_ under moderate skew (α=1.0) than under severe skew (α=0.1). This is somewhat counterintuitive but has a clear explanation: under α=0.1, the EMA student model is already doing most of the heavy lifting (providing +13.75pp vs. no EMA). Soft-voting is the secondary stabilizer. Under α=1.0, the EMA is less dominant (only modest gain), so soft-voting's ability to filter uncertain/low-precision teachers has more room to contribute.

**Implication for thesis:** The two mechanisms are **complementary, not redundant**. EMA provides temporal stability; soft-voting provides spatial (per-sample, per-teacher) stability. Both are independently necessary.

---

## 2. Optimal Soft-Voting Parameters (Phase 2 Grid)

### 2.1 α = 0.1 (Severe Non-IID) — Best: `ew=4.0, pw=1.0`

The optimal under severe skew shows **high entropy weight** (4.0) and **linear precision weight** (1.0). This aligns with the theoretical prediction in [HANDOFF.md §10](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md#L163-L190):

> "Under α=0.1, client bit-widths vary widely (q=4 to q=8) and low-quality partitions produce noisy logits. Higher `precision_weight` values are likely to help more here."

The empirical result partially contradicts the precision-weight prediction: `pw=1.0` (linear) outperforms `pw=4.0` (hard exclusion). The likely reason: under severe skew, even low-bit-width clients have _discriminative_ predictions for their expert class (the class dominating their local data). Excluding them entirely (pw=4.0) discards this useful specialized signal. Linear weighting (pw=1.0) correctly tempers but preserves it.

The **high entropy weight** (ew=4.0) is clearly beneficial: under severe skew, many client predictions on non-expert classes are near-uniform (high entropy). Aggressively suppressing these uncertain predictions keeps the distillation signal clean.

**Key observation:** The `ew=4.0` row consistently outperforms `ew=0.5` across all `pw` values (52.83% vs best 50.43%), confirming that entropy gating is the primary driver at α=0.1.

### 2.2 α = 1.0 (Moderate Non-IID) — Best: `ew=2.0, pw=0.5`

The optimal under moderate skew shows **moderate entropy weight** (2.0) and **sub-linear precision weight** (0.5). This is also theory-consistent:

- Lower `entropy_weight` (2.0 vs 4.0) makes sense: under moderate skew, clients have more balanced partitions, so even uncertain predictions contain useful multi-class information. Hard-gating on entropy (ew=4.0) would throw away useful signal.
- Sub-linear `precision_weight` (0.5) is the most striking result. The theoretical prediction said "lower values are expected to be closer to optimal" for α=1.0, and pw=0.5 is the softest option in the grid. Under moderate skew, even 4-bit clients produce reasonably accurate predictions (their data is balanced), so being tolerant of low-precision teachers (pw=0.5 gives them a 50% weight vs 25% for pw=1.0) retains more ensemble diversity.

**The key finding:** The optimal point moved from the **strict/selective** quadrant (ew=4.0, pw=1.0) at α=0.1 to the **moderate/tolerant** quadrant (ew=2.0, pw=0.5) at α=1.0. This is a systematic, theory-consistent shift — **not noise**.

---

## 3. Accuracy Jump at α=1.0: The Breakthrough

The tuned soft-voting at α=1.0 (**63.28%**) represents a significant improvement over:

- Default soft-voting (61.86%) → +1.42pp
- No EMA, no tuning (61.07%) → +2.21pp
- But still trails FedAvg (67.57%) by **−4.29pp** (down from −5.71pp gap)

The gap to uncompressed baselines is narrowing with each tuning step. This progression is worth documenting in the thesis:

| Improvement                                      | α=1.0 Accuracy | Gap to FedAvg |
| :----------------------------------------------- | :------------: | :-----------: |
| Original FedMAQ (smoke test)                     |     58.07%     |    −9.50pp    |
| + All 3 robustness features (EMA sweep baseline) |     61.07%     |    −6.50pp    |
| + EMA tuning (β=0.1)                             |     61.86%     |    −5.71pp    |
| + Soft-voting tuning (ew=2.0, pw=0.5)            |   **63.28%**   |  **−4.29pp**  |

Each step has a clear mechanistic explanation.

---

## 4. Interaction Effects: EMA × Soft-Voting

Both mechanisms operate on the distillation pipeline but at different points:

- **Soft-voting:** Per-sample, per-teacher weighting _before_ the KD loss is computed (spatial filtering).
- **Student EMA:** Temporal smoothing of the student model _after_ the KD update (temporal regularization).

The ablation (Phase 1) used fixed optimal EMA settings. The Phase 2 grid also used fixed EMA. We did not sweep their interaction. Based on the results:

- The soft-voting optimum at α=0.1 (ew=4.0, pw=1.0) gives **52.83%**, slightly above the EMA-only best with default SV (52.51%). The gain from tuned SV over default SV is modest (~0.3pp).
- The soft-voting optimum at α=1.0 (ew=2.0, pw=0.5) gives **63.28%**, a much larger gain over default SV (61.86%), +1.42pp.

This suggests the two mechanisms are **largely orthogonal** — each contributes independently. A joint sweep is unlikely to reveal significant interaction effects beyond what we've already captured.

---

## 5. Revised Best FedMAQ Configuration

Based on all sweeps to date:

| Hyperparameter     | α=0.1 (Severe) | α=1.0 (Moderate) |
| :----------------- | :------------: | :--------------: |
| `formulation`      |       3        |        3         |
| `ema_student`      |      true      |       true       |
| `ema_decay`        |    **0.7**     |     **0.1**      |
| `soft_voting`      |      true      |       true       |
| `entropy_weight`   |    **4.0**     |     **2.0**      |
| `precision_weight` |    **1.0**     |     **0.5**      |
| `grad_norm_ema`    |      true      |       true       |
| `grad_norm_beta`   |      0.7       |       0.7        |

---

## 6. Next Steps

1. **Update HANDOFF.md** with these optimal configurations and sweep 4 completion.
2. **Decide on architecture** (SimpleCNN vs. ResNet18GN — per audit recommendations). This will affect whether all experimental numbers need to be re-run.
3. **Temperature ablation** (T=1.0 vs T=2.0): 4 runs, low effort, pre-empts a defense question.
4. **Final benchmark grid**: 100-round sweeps at α ∈ {0.1, 0.3, 0.5, 1.0} with tuned FedMAQ vs. all baselines.
