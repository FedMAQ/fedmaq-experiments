# FedMAQ Audit — Actionable Recommendations

**Last updated**: 2026-07-16

Cross-referencing every point in [fedmaq-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/fedmaq-audit.md) and [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md), plus additional findings from the codebase.

> [!NOTE]
> Findings are architecture-independent (math/logic audit). Occurrences of ResNet18GN below are illustrative examples, not deprecated results — the MobileNetV2GN switch (see [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)) does not invalidate these findings.

---

## 1. Architecture Confounder — Switch to Iso-Architecture ✅ DONE

> **Verdict**: ✅ **Applied.** Originally recommended switching FedMAQ clients to ResNet18GN; the project instead adopted **MobileNetV2GN** as the iso-architecture for all algorithms (edge-realism rationale, see [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md) Decision 1). The underlying diagnosis below is still correct and explains why the change was made — only the target architecture differs from what's shown here.

The audit correctly identified that FedMAQ clients originally trained **SimpleCNN** (~2.16M params) while baselines trained **ResNet18GN** (~11.17M params), confounding comm-reduction numbers with model-size reduction, not purely quantization. Current code (`get_client_model` in [models.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/models.py#L380-L388)) now routes `fedmaq` through `get_model()` (default `mobilenetv2gn`), same as all baselines — only `fedkd` and `fedmaq_lite` still use the smaller KD-student path. This makes FedMAQ clients train **the same MobileNetV2GN** as FedAvg/FedProx/etc., turning server-side KD into **self-distillation** (MobileNetV2GN → MobileNetV2GN).

### Impact on Thesis Narrative

The self-distillation framing is a **strength**, not a weakness:

- It proves the communication savings come purely from multi-adaptive quantization, not model-capacity reduction.
- Self-distillation (same architecture teacher→student) has strong literature precedent (Born-Again Networks, Zhang et al. 2019).
- The thesis narrative shifts from "FedMAQ compresses with a smaller model" to "FedMAQ achieves X× communication reduction at iso-architecture", which is far more defensible.

---

## 2. EMA–FedAvg Aggregation Ordering (Audit §2.2 — 🔴 High)

> **Verdict**: ✅ **No code change needed — but prepare a defense narrative.**

The audit's own analysis concludes the ordering is _actually correct_. The EMA acts as **outer-loop temporal smoothing** of the KD-refined model, not a replacement for aggregation. The key chain is:

1. Clients train from $\theta_{EMA}^{(t-1)}$ (the broadcast model)
2. FedAvg aggregates their returns → $\theta_{agg}^{(t)}$
3. KD refines → $\theta_{KD}^{(t)}$
4. EMA smooths → $\theta_{EMA}^{(t)} = \beta \cdot \theta_{EMA}^{(t-1)} + (1-\beta) \cdot \theta_{KD}^{(t)}$

### Recommendation

Add a **formal comment block** at [fedmaq.py:L316](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L316) explaining this. Prepare a 2-paragraph defense for the thesis Chapter 4 discussion section. The key argument: "EMA is applied to the _output_ of the full aggregation+KD pipeline, not _within_ it. This is equivalent to Polyak averaging of the server model trajectory, a well-established technique (Tarvainen & Valpola, 2017)."

---

## 3. EMA Decay — Heterogeneity-Dependent β (Audit §5.2 / HANDOFF §9)

> **Verdict**: ⚠️ **Acknowledge as a finding, implement a regime-aware default.**

The EMA sweep shows $\beta=0.7$ is optimal at $\alpha=0.1$ and $\beta=0.1$ at $\alpha=1.0$. The audit recommends framing this as a _discovery_, not a weakness.

### Recommendation

**Option A (Minimal — recommended for thesis timeline):**

- Set `ema_decay` in `fedmaq.yaml` to `0.5` as a compromise default.
- In the final grid sweep, fix $\beta \in \{0.1, 0.5, 0.7\}$ per heterogeneity regime using the best values from the sweep.
- Frame the inverse $\beta$-$\alpha$ relationship as a **finding**: "FedMAQ reveals a heterogeneity-EMA decay tradeoff law."

**Option B (Stretch goal — future work):**

- Implement simple adaptive scheduling: $\beta_t = \beta_0 \cdot \text{clip}(\sigma_{\Delta} / \sigma_0, 0.1, 1.0)$ where $\sigma_{\Delta}$ is the standard deviation of client weight deltas. When client drift is high (severe non-IID), $\beta$ stays high; when drift is low (homogeneous), $\beta$ decreases automatically.
- This removes the hyperparameter dependence on α entirely but adds implementation complexity.

---

## 4. KD Temperature $T = 1.0$ (Audit §3.1)

> **Verdict**: ⚠️ **Run a 2-point ablation ($T \in \{1.0, 2.0\}$), do not change the default.**

The audit's defense is convincing: quantized teacher outputs are already noisy and less peaked, so further softening via $T > 1$ may degrade signal quality. However, a reviewer will ask about it.

### Recommendation

Add a 2-run ablation ($T = 1.0$ vs $T = 2.0$) at the best EMA setting for each α regime. This takes ~4 runs (2 temperatures × 2 α values). If $T = 1.0$ wins or ties, cite this as empirical validation. If $T = 2.0$ wins, adopt it. Either way, the ablation costs minimal compute and preempts a defense question.

---

## 5. Soft-Voting Sweep (HANDOFF §10–11, item 4)

> **Verdict**: ✅ **Continue as planned — the 4×4 grid is well-designed.**

The $\gamma_{entropy} \times \gamma_{precision}$ grid is a principled log-scale search. The HANDOFF correctly identifies distinct physical regimes (Inclusive → Strict).

### Additional Recommendation

After the sweep completes, check for **interaction effects** between soft-voting and EMA:

- Does the optimal $(\gamma_e, \gamma_p)$ shift when EMA decay changes?
- If so, consider running a small joint sweep of the top-3 soft-voting configs × top-2 EMA values per α regime.
- If the interaction is weak (stable optimal across EMA values), report this as evidence that the two mechanisms are **orthogonal**, which strengthens the thesis.

---

## 6. Fixed vs. Memory-Proportional Compute (HANDOFF §7)

> **Verdict**: ✅ **Keep Option A (fixed compute speed) for the thesis.**

The thesis manuscript already commits to "memory remains the sole Tier 1 resource signal." Switching to Option B now would:

1. Require rewriting Chapter 1 and 4 sections.
2. Introduce a confounding variable that dilutes the core contribution (multi-adaptive _quantization_, not multi-adaptive _scheduling_).
3. Add another hyperparameter axis (compute speed scaling function).

### Recommendation

Keep fixed compute. Acknowledge variable compute as a **future work** direction: "Extending FedMAQ's Tier 1 to jointly optimize bit-width and local epochs based on both memory and compute heterogeneity is a natural extension."

---

## 7. Byte Accounting (Audit §5.4)

> **Verdict**: ✅ **No action needed — already correct.**

The audit confirms that:

- Precision weights are available server-side (no extra uplink cost).
- Entropy computation happens within the same teacher forward passes (no extra server time).
- The +4 bytes per tensor for the float32 scale factor is correctly accounted.

No changes required.

---

## 8. Gradient Norm Smoothing β = 0.7 (Audit §2.3)

> **Verdict**: ✅ **Keep as is.**

The audit correctly argues that grad norm smoothing addresses _measurement noise_, not _learning dynamics_, so a single fixed $\beta = 0.7$ is appropriate. This is a fundamentally different mechanism from the student EMA. No change needed.

---

## 9. Proxy Dataset Size (Audit §5.3 / HANDOFF §4)

> **Verdict**: ✅ **Already addressed — 3000 samples is defensible.**

The update from 1600 → 3000 is done. 3000 is <6% of CIFAR-10 and aligns with FedDF and DynFed precedent.

### Minor Recommendation

Add a single-line note in the thesis acknowledging the assumption: "FedMAQ assumes access to 3,000 unlabeled public samples, consistent with the proxy-dataset distillation paradigm (FedMD, FedDF, DynFed)."

---

## 10. Additional Issues Found During Code Review

### 10.1 `ema_decay` Default in Config vs. Code Mismatch

> **Severity**: ⚠️ Medium

[fedmaq.yaml:L45](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/algorithm/fedmaq.yaml#L45) sets `ema_decay: 0.99` but the EMA sweep found that $\beta = 0.99$ was never tested (sweep range was 0.1–0.9) and the best values are $0.7$ and $0.1$. The default of `0.99` would produce very heavy smoothing, likely introducing severe convergence lag.

**Recommendation**: Update `fedmaq.yaml` default to `ema_decay: 0.5` (neutral compromise) or `ema_decay: 0.7` (best for the harder/more interesting severe-skew regime).

### 10.2 `get_kd_student_model` Import Cleanup — ✅ Verify Applied

Recommendation 1's switch implies `fedmaq.py` no longer needs `get_kd_student_model` for its main client/grad-norm-probe/KD-factory paths. Spot-check [fedmaq.py:L26](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L26) for a dead import if not already cleaned up.

### 10.3 Evaluation Uses `get_client_model` — Correct After Switch

[simulation.py:L179](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/simulation.py#L179) uses `get_client_model(alg_name, ...)` for the `evaluate_fn`. After the switch, `get_client_model("fedmaq", "cifar10", 10)` correctly returns `MobileNetV2GN`, so the evaluation model matches the aggregated parameters. **No change needed here** — it auto-propagates.

### 10.4 `_grad_norm_ema` Unbounded Growth

[fedmaq.py:L136](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy_hooks/fedmaq.py#L136): `self._grad_norm_ema: dict[int, float] = {}` grows monotonically — every client ever sampled gets a persistent entry. With $K = 100$ clients and a typical participation rate of 10%, this dict will have at most 100 entries, so it's not a problem at thesis scale. But for production use with $K \gg 1000$, consider an LRU eviction policy.

**Recommendation**: No action for thesis. Add a brief "scalability note" comment in the code.

---

## Priority Summary

|  #   | Item                  | Action                                    |   Effort    |   Impact    |
| :--: | :-------------------- | :---------------------------------------- | :---------: | :---------: |
|  1   | Iso-arch switch       | ✅ Done (MobileNetV2GN, not ResNet18GN)   |    High     | 🔴 Critical |
|  2   | EMA–FedAvg ordering   | **Document** defense narrative            |     Low     | 🔴 Critical |
|  3   | EMA decay default     | **Config fix** (`0.99` → `0.5` or `0.7`)  |   Trivial   |  ⚠️ Medium  |
|  4   | Temperature ablation  | **Run 4 experiments** ($T \times \alpha$) |     Low     |  ⚠️ Medium  |
|  5   | Soft-voting sweep     | **Continue** as planned                   | In progress | ✅ On track |
|  6   | Fixed compute speed   | **No change** — frame as future work      |    None     | ✅ Decided  |
|  7   | Byte accounting       | **No change**                             |    None     | ✅ Verified |
|  8   | Grad norm β = 0.7     | **No change**                             |    None     | ✅ Verified |
|  9   | Proxy dataset         | **No change** — add thesis footnote       |   Trivial   |   ✅ Done   |
| 10.1 | Config default fix    | **Config edit**                           |   Trivial   |  ⚠️ Medium  |
| 10.2 | Dead import cleanup   | **Code edit** (alongside #1)              |   Trivial   |     Low     |
| 10.3 | Eval auto-propagation | **No change**                             |    None     | ✅ Verified |
| 10.4 | EMA dict growth       | **Comment** only                          |   Trivial   |     Low     |

---

## Recommended Execution Order

> [!NOTE]
> Superseded by [HANDOFF.md §5](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md) and [docs/plans/formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md) as the canonical next-steps list (2026-07-16 grilling). The list below is retained for historical context on how items 1-6 were originally sequenced; items 1-6 are done. See HANDOFF §5 for current priorities on MobileNetV2GN.

1. ~~Fix the config default (`ema_decay: 0.99` → `0.5`)~~ — done.
2. ~~Implement iso-architecture switch (Recommendation 1)~~ — done, MobileNetV2GN.
3. ~~Re-run the EMA decay sweep~~ — superseded; capacity-EMA duality re-opened as a question for MobileNetV2GN (see formal-experiment-plan.md §2).
4. ~~Wait for soft-voting sweep~~ — done (ResNet18GN era); MobileNetV2GN re-sweep tracked in formal-experiment-plan.md §2.
5. ~~Run the temperature ablation~~ — folded into the formal ablation table (DECISIONS.md Decision 12).
6. ~~Write the EMA–FedAvg defense paragraph~~ — still owed, still applies (architecture-independent).
7. ~~Launch the final 100-round grid on ResNet18GN~~ — superseded; grid now runs on MobileNetV2GN per DECISIONS.md Decision 9-10.
