# Analysis of Stage 1a Power-Mean Degree Selection (\(p = 0.5\))

**Date**: 2026-09-14 (originally); revised 2026-09-16 with the \(n=5\) seed extension.  
**Status**: **Final (\(n = 5\)).** The seed extension registered in
[ADR-0024](../adr/0024-stage-1a-seed-extension-and-stage-1b-omega-arm.md) has run and
supersedes the \(n = 3\) verdict below (ADR-0024 D1: the \(n=5\) result is authoritative
regardless of whether it confirms or overturns \(n=3\)). The selection rule still fires
`selected_p = 0.5`, and the extension does not change that outcome. It does change the
evidentiary picture underneath it, in one case substantially (§3.4): finalizing this
document closes Stage 1a's evidence gathering, not the question of how strong that evidence
is — §4 remains a live list of threats, not a formality.  
**Context**: Resolves Stage 1a of the formulation design space ([ADR-0021](../adr/0021-power-mean-formulation-family.md), [ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md), [Issue #34](https://github.com/FedMAQ/fedmaq-experiments/issues/34), [Issue #110](https://github.com/FedMAQ/fedmaq-experiments/issues/110)).  
**Authoritative artifacts**:
- `scripts/analysis_output/power_mean_degree_selection.json` (\(n=5\) for the seven degree-\(p\) arms; \(n=3\) for the eight descriptive controls, which the extension did not touch)
- `scripts/analysis_output/power_mean_degree_resolution.json`

---

## 1. Executive Summary & Verdict

The Stage 1a sweep (`power_mean_design`) evaluated the generalized power-mean degree ladder:
\[
s_k = M_p(\tilde{g}_k, \tilde{n}_k; \omega) = \left(\omega \tilde{g}_k^p + (1-\omega) \tilde{n}_k^p\right)^{1/p}, \quad \omega = 0.5
\]
across \(p \in \{1, 0.5, 0, -0.5, -1, -2, \text{min}\}\), plus three descriptive controls (F0 resource-only, F3 gradient-primary \(\kappa \in \{0.5, 1.0, 2.0\}\), and F4 threshold \(\tau \in \{0.3, 0.5, 0.7\}\)) over CIFAR-10 / MobileNetV2 (100 rounds; 5 seeds \(\{0, 7, 21, 42, 123\}\) for the seven degree-\(p\) arms, 3 seeds \(\{0, 42, 123\}\) for the descriptive controls).

Under the pre-registered iso-byte evaluation rule ([ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md)):

- **Selected formulation parameter**: **\(p = 0.5\)**
- **Selection rule outcome**: **`"rule": "agreement"`** — the \(\alpha = 0.1\) winner and the
  \(\alpha = 1.0\) winner coincide, so no severe-skew tie-break was invoked.

The rule fired as pre-registered and the outcome stands procedurally, unchanged from \(n=3\)
to \(n=5\). What the data does **not** support, at either \(n\), is a claim that \(p = 0.5\)
was separated from its rivals by the evidence. Both winning margins remain inside seed noise,
and the two selecting cells fail for two different, non-interchangeable reasons:

| Cell | Mean margin over runner-up | Paired \(t\) (\(n=5\)) | Sign-consistent across seeds? |
| :--- | :---: | :---: | :--- |
| \(\alpha = 0.1\) vs `p-1` | \(+0.0038\) | \(0.20\) | No — 2 of 5 seeds favor `p-1` |
| \(\alpha = 1.0\) vs `p0` | \(+0.0016\) | \(1.00\) (exact) | Yes, but 4 of 5 seeds are byte-identical ties |

**These are not the same kind of weak.** \(\alpha = 1.0\) is *degenerate*: as shown in §3.3,
the paired \(t\)-statistic for `p0.5` vs `p0` is forced to exactly \(1.0\) by the arithmetic
of having four exactly-tied seed pairs and one seed (42) carrying the entire margin — a
structural property of this cell, not evidence of an effect of any size. \(\alpha = 0.1\) is
the opposite failure mode: it is *not* degenerate (all five paired differences are distinct,
see §3.3 vs. this cell), but the differences do not agree in sign — seed 21 favors `p-1` by
\(0.061\) while seed 123 favors `p0.5` by \(0.056\) — so the comparison is real but
inconclusive, not real and weak. Neither cell supports a claim that `p0.5` was validated by
independent evidence at both skews; they fail independently, for independent reasons, and the
sum of two non-findings is not a finding.

At \(\alpha = 0.3\) — the non-selecting robustness contrast ([ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md)) — `p0.5` now ranks
**5th of seven** by mean (it ranked last at \(n=3\); `p0` and `p1` have since dropped below
it). One comparison in this cell, `p0.5` vs the cell winner `p-1`, remains sign-consistent
across all five seeds and is still the strongest signal anywhere in the sweep, though it
weakened from \(t=3.83\) at \(n=3\) to \(t=2.54\) at \(n=5\) (§3.4). That cell does not and
cannot change `selected_p`, but it remains the most defensible empirical claim this sweep
supports, and it is reported here rather than minimized.

**Read this document as: the pre-registered procedure selected \(p = 0.5\) at both \(n=3\)
and \(n=5\), on margins that remain unresolved at \(n=5\) — for structurally different
reasons in the two selecting cells.** The extension changed the diagnosis, not the verdict.

---

## 2. Empirical Performance Across Skews

Mean top-1 validation accuracy at the minimum common cumulative-MB budget (\(B\)), with
dispersion across seeds \(\{0, 7, 21, 42, 123\}\) for the seven degree-\(p\) arms
(\(n=5\)). Dispersion at \(n=5\) still carries substantial sampling noise; the \(\pm\)
figures below are descriptive, not confidence intervals.

| Degree \(p\) | Operator Category | \(\alpha = 0.1\) (\(B \approx 8058\) MB) | \(\alpha = 1.0\) (\(B \approx 8198\) MB) | \(\alpha = 0.3\) Robustness Contrast (\(B \approx 8130\) MB) |
| :--- | :--- | :---: | :---: | :---: |
| **`p1`** | Arithmetic mean (linear, fully compensatory) | \(0.2688 \pm 0.0226\) | \(0.5234 \pm 0.0114\) | \(0.4286 \pm 0.0198\) |
| **`p0.5`** | Root-mean (concave, mildly compensatory) | **\(0.2768 \pm 0.0250\)** ★ | **\(0.5328 \pm 0.0131\)** ★ | \(0.4311 \pm 0.0236\) (5th) |
| **`p0`** | Geometric mean (multiplicative, historical v1 F2) | \(0.2646 \pm 0.0176\) | \(0.5312 \pm 0.0147\) | \(0.4305 \pm 0.0173\) |
| **`p-0.5`** | Sub-geometric mean | \(0.2670 \pm 0.0184\) | \(0.5165 \pm 0.0220\) | \(0.4388 \pm 0.0183\) |
| **`p-1`** | Harmonic mean | \(0.2730 \pm 0.0198\) | \(0.5266 \pm 0.0136\) | **\(0.4480 \pm 0.0223\)** ★ |
| **`p-2`** | Sub-harmonic mean | \(0.2704 \pm 0.0255\) | \(0.5248 \pm 0.0182\) | \(0.4341 \pm 0.0175\) |
| **`p-min`** | Leontief minimum operator | \(0.2650 \pm 0.0297\) | \(0.5262 \pm 0.0082\) | \(0.4316 \pm 0.0365\) |

★ marks the per-cell mean winner. The \(\alpha = 0.3\) winner is `p-1`; per ADR-0023 that cell
does not feed `resolve_power_mean_degree`. **Ranking change from \(n=3\):** at \(\alpha=0.3\),
`p0.5` ranked last (7th of 7) at \(n=3\); at \(n=5\) it ranks 5th of 7, having overtaken `p0`
and `p1`. This is a real reordering, not rounding — see §3.4.

### Comparison to Descriptive Controls

| Control | Parameters | \(\alpha = 0.1\) | \(\alpha = 1.0\) | \(\alpha = 0.3\) |
| :--- | :--- | :---: | :---: | :---: |
| **Resource-only (F0)** | Hardware memory cap only | \(0.2767 \pm 0.0308\) | \(0.5174 \pm 0.0096\) | \(0.4233 \pm 0.0200\) |
| **F3 (Gradient-primary)** | \(\kappa = 0.5\) | \(0.2816 \pm 0.0241\) | \(0.5121 \pm 0.0034\) | \(0.4183 \pm 0.0172\) |
| | \(\kappa = 1.0\) | \(0.2710 \pm 0.0136\) | \(0.5120 \pm 0.0178\) | \(0.4078 \pm 0.0429\) |
| | \(\kappa = 2.0\) | \(0.2587 \pm 0.0055\) | \(0.5027 \pm 0.0318\) | \(0.4353 \pm 0.0414\) |
| **F4 (Threshold-tied)** | \(\tau_g = \tau_n = 0.3\) | \(0.2751 \pm 0.0283\) † | \(0.5174 \pm 0.0096\) † | \(0.4233 \pm 0.0200\) † |
| | \(\tau_g = \tau_n = 0.5\) | \(0.2878 \pm 0.0361\) | \(0.5051 \pm 0.0170\) | \(0.4194 \pm 0.0275\) |
| | \(\tau_g = \tau_n = 0.7\) | \(0.2501 \pm 0.0262\) | *(out of support)* | *(out of support)* |

† **F4 \(\tau = 0.3\) is not an independent control.** Its per-seed accuracies are
bit-identical to F0 resource-only in 8 of 9 cells (all three seeds at \(\alpha = 1.0\), all
three at \(\alpha = 0.3\), and seeds 0 and 42 at \(\alpha = 0.1\)) — identical to the full
printed float, not merely close. At \(\tau = 0.3\) the threshold never binds, so the
formulation degenerates to the resource-only policy and the row duplicates F0. It should be
read as a degeneracy check, not as a third point on a threshold sweep. This is a behavioral
property of the formulation at that threshold, not a defect in the runner.

**Reporting asymmetry.** The degree rows above carry \(n = 5\) while every control row
remains at \(n = 3\); ADR-0024's extension added seeds to the seven power-mean arms only, not
to the descriptive controls. The F4 \(\equiv\) F0 finding is and remains an \(n = 3\) result.

---

## 3. Behavioral Observations

The mechanisms below are offered as interpretation of the observed ordering. At the observed
effect sizes they are consistent with the data rather than demonstrated by it.

### 3.1 The Shape of the Ladder

1. **The linear mode (\(p = 1\))**: under an arithmetic combination, a high local sample
   count \(\tilde{n}_k\) can fully offset a near-zero or noisy gradient update
   \(\tilde{g}_k\), so the server can allocate precision to clients contributing little new
   directional information.
2. **The non-compensatory mode (\(p \le -1\))**: harmonic and minimum operators penalize any
   asymmetry sharply. A client with informative gradients but a modest partition sees its
   soft score collapse toward zero.
3. **The concave interior (\(p = 0.5\))**:
   \[
   s_k = \left(\frac{\sqrt{\tilde{g}_k} + \sqrt{\tilde{n}_k}}{2}\right)^2
   \]
   Sub-linear powers in \((0, 1)\) impose diminishing marginal returns on either signal
   alone, while staying compensatory enough that an informative small-data client is not
   starved.

This account predicts an interior optimum, and the \(\alpha = 0.1\) and \(\alpha = 1.0\)
means are ordered consistently with it. It does **not** predict the \(\alpha = 0.3\)
ordering, where the non-compensatory operators win (§3.4).

**Why degeneracy should track heterogeneity.** \(M_p(x_1, \ldots, x_k)\) is bounded between
\(\min(x_i)\) and \(\max(x_i)\) for every real \(p\), and the whole family collapses to that
common value as the \(x_i\) converge. Client-level \((\tilde{g}_k, \tilde{n}_k)\) dispersion
is set by the Dirichlet \(\alpha\): \(\alpha = 1.0\) produces near-homogeneous client signals,
so every \(p\) in the ladder should compute nearly the same score per client and the resulting
bit-width plans should frequently coincide — exactly the exact-tie pattern documented in
§3.3. \(\alpha = 0.1\) produces high dispersion, so the ladder's degree should actually matter
and plans should diverge — consistent with \(\alpha = 0.1\) showing zero exact ties across
any of the five seeds (§3.3). This is a property of \(M_p\), not an assumption about this
codebase, and the fact that the observed tie/no-tie pattern matches it exactly is evidence the
degree ladder is implemented and applied correctly — a bug that ignored \(p\) would tie
everywhere regardless of \(\alpha\), and a bug that used the wrong client signal would not tie
in the specific seed-and-arm pattern the boundedness argument predicts. It is evidence the
pipeline computes what it is supposed to compute, not evidence for or against any particular
\(p\): degeneracy governs whether a comparison is *interpretable*, not whether it is
*significant*, and the significance question is separate (§4.1).

### 3.2 Relation to Historical v1 (Formulation 2)

The superseded v1 campaign ([ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md)) froze Formulation 2 (geometric / multiplicative,
\(p = 0\)). On the continuous ladder, \(p = 0.5\) outscores \(p = 0\) in mean accuracy at
both reporting skews, at \(n=5\):

- \(\alpha = 0.1\): \(+0.0122\) **accuracy, i.e. \(+1.22\) percentage points** (about
  \(+4.6\%\) relative to \(p=0\)'s \(0.2646\)); paired \(t = 1.08\); not sign-consistent
  (3 of 5 seeds favor `p0.5`, 2 favor `p0`).
- \(\alpha = 1.0\): \(+0.0016\), i.e. \(+0.16\) percentage points (about \(+0.3\%\)
  relative); paired \(t = 1.00\) **exactly**, and the entire margin originates at seed 42 —
  the other four seeds are exact byte-identical ties (§3.3).

At \(n=3\) this comparison read \(+0.0164\) / \(t=0.96\) at \(\alpha=0.1\) and \(+0.0027\) /
\(t=1.00\) at \(\alpha=1.0\); both margins shrank at \(n=5\) as the two new seeds (7, 21) came
in — at \(\alpha=0.1\), seed 21 is in fact the one seed favoring `p0` by the largest margin in
the cell. Neither margin is distinguishable from noise at \(n = 5\). The defensible statement
is unchanged from \(n=3\): \(p = 0.5\) is **not worse** than the historical geometric choice
at either reporting skew, and the ladder subsumes it as a special case.

### 3.3 Numerical Degeneracy at \(\alpha = 1.0\)

The mild-skew cell carries much less information than its row in §2 suggests, because many
arms return bit-identical accuracies, and the \(n=5\) extension sharpens rather than
dissolves this: the two new seeds (7, 21) also land on exact ties in the comparisons that
matter most.

- **`p0.5` vs `p0` — the comparison that actually drives the "agreement" verdict**: seeds 0,
  7, 21, and 123 are exact byte-identical ties; only seed 42 differs, by \(+0.00798\). With 4
  of 5 paired differences forced to zero, the paired \(t\)-statistic is **algebraically fixed
  at exactly \(1.0\)** regardless of the fifth seed's magnitude: if \(d\) is the one nonzero
  difference, the sample mean is \(d/5\), the sample standard deviation is \(d/\sqrt{5}\), the
  standard error is \(d/5\), and \(t = \text{mean}/\text{se} = 1\) identically. This is not "a
  weak effect" — the statistic carries **zero information about effect size**, by
  construction, independent of whatever value seed 42 happens to take. The mean margin,
  \(0.00798/5 = 0.0015953\), is exactly the `margin_accuracy` field in
  `power_mean_degree_resolution.json`.
- **`p0.5` vs `p1`**: an identical pattern — seeds 7, 21, 42, 123 tie exactly; only seed 0
  differs. Same algebraic \(t \equiv 1.0\).
- **`p0.5` vs `p-0.5`**: 3 of 5 seeds tie exactly (7, 21, 123); seeds 0 and 42 both favor
  `p0.5`, giving \(t = 1.13\) — still low-information, though not forced to exactly 1 since
  two seeds now vary.

At \(\alpha = 1.0\) the client distribution is close enough to uniform that the allocation
policy frequently produces the same bit-width plan regardless of \(p\), and identical plans
give identical trajectories under the repeatability gates — consistent with the boundedness
argument in §3.1. The cell's "agreement" label for `selected_p = 0.5` rests entirely on one
seed (42) resolving a four-way tie; that is the complete evidentiary content of this cell's
contribution to the resolution.

One comparison in this cell is *not* degenerate: `p0.5` \(-\) `p-1` has only one exact tie
(seed 21) and is sign-consistent across all five seeds, giving mean \(+0.0062\) at
\(t = 3.07\) (\(n=3\) reported \(t=5.57\) on the three-seed subset). `p-1` is not the
resolution-relevant rival in this cell — `p0` is — so this comparison plays no role in the
"agreement" verdict; it is reported because it is the one genuinely informative comparison
this degenerate cell contains, and because its \(n=3\rightarrow n=5\) weakening is itself a
useful data point about how unstable small-\(n\) statistics are in this design.

### 3.4 The \(\alpha = 0.3\) Contrast Is the Strongest Signal in the Sweep

Per [ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md), \(\alpha = 0.3\) was added as an empirical contrast that does not govern
selection, and `power_mean_degree_resolution.json` accordingly carries no \(\alpha = 0.3\)
field. That protocol position is unchanged. The \(n=5\) extension changes this cell's
internal picture more than any other, and the change runs in both directions at once:

- **`p0.5`'s rank improved**: it ranked last (7th of 7, losing every seed) at \(n=3\). At
  \(n=5\) it ranks **5th of 7** by mean, ahead of `p0` and `p1` — the two new seeds (7, 21)
  favor `p0.5` over both of those arms. This reranking is real, not a rounding artifact; see
  the per-comparison diffs below.
- **Only one comparison survived as sign-consistent**: at \(n=3\), the three comparisons that
  mattered (`p-1`, `p-0.5`, `p0` vs `p0.5`) were all reported as sign-consistent, with
  \(p_{-1}-p_{0.5}\) at \(t=3.83\) and \(p_{-0.5}-p_{0.5}\) at \(t=3.08\). At \(n=5\), only
  `p_{-1} - p_{0.5}` remains sign-consistent across all five seeds (all five paired
  differences favor `p-1`: \(+0.027, +0.005, +0.002, +0.037, +0.014\)), giving
  \(t = 2.54\) — weaker than at \(n=3\), but still the sharpest signal in the entire sweep,
  and the only comparison anywhere in this document with all five seeds agreeing in sign on a
  nontrivial margin. `p_{-0.5} - p_{0.5}` lost sign-consistency at \(n=5\): seed 7 now favors
  `p0.5` by \(-0.038\), the largest single seed-difference in the comparison, flipping what
  was a clean 3-seed signal into \(t=0.60\). \(p_{0} - p_{0.5}\) also lost sign-consistency and
  its mean margin collapsed to \(-0.0006\) — confirming the \(n=3\) flag that this
  comparison's \(t=5.01\) was a near-zero-variance artifact, the same phenomenon §3.3
  discounts for the \(\alpha=1.0\) `p0.5`-`p1` comparison.

The earlier ($n=3$) draft described `p-1` vs `p0.5` and `p-0.5` vs `p0.5` together as the
sweep's clearest signal. At \(n=5\), only the first of those two holds up; the second did
not survive the additional seeds. Comparing dispersion bars is the wrong test here in either
case, because the comparison is paired — the same seeds run both arms.

Read plainly at \(n=5\): the compensatory interior of the ladder is favored, weakly and
inconclusively, at the two reporting skews that govern selection, while the non-compensatory
end (`p-1` specifically) is favored, moderately but not quite at conventional significance
(\(t=2.54\) against a two-tailed \(df=4\) critical value of \(2.78\)), at the intermediate
skew. Whether that is a real regime dependence or continuing seed noise remains open even at
\(n=5\) — this document does not resolve it, since ADR-0023 excludes \(\alpha=0.3\) from the
selection rule and no further seed extension is registered for this cell. It is the sweep's
most reportable finding for the discussion chapter precisely because it is the one comparison
that has partially survived a doubling of \(n\), not because it has reached significance.

---

## 4. Threats to This Selection

1. **Underpowering — worse at \(n=5\) than the \(n=3\) estimate projected.** At \(n=3\), the
   \(\alpha=0.1\) selection margin was \(+0.0096\) with paired \(\sigma \approx 0.024\), and
   this document projected roughly 37 seeds to reach conventional significance. At \(n=5\),
   the margin fell to \(+0.0038\) while the paired \(\sigma\) grew to \(\approx 0.042\) — both
   changes point the same way. A naive extrapolation from the \(n=5\) sample (fixing
   \(\sigma \approx 0.042\), solving for \(n\) at \(t \approx 2\)) puts the seed count needed
   at roughly 500 seeds (mid-hundreds), over an order of magnitude worse than the \(n=3\) projection. That
   figure should be treated as directional, not load-bearing: it is itself computed from a
   5-point standard-deviation estimate and would move again with more seeds. The more robust
   statement is qualitative and does not depend on extrapolation: the two \(\alpha=0.1\)
   comparison seeds added between \(n=3\) and \(n=5\) (7, 21) disagreed with each other in
   sign and by a large margin, which is the signature of a comparison whose population effect
   may not have a stable sign at all — more seeds narrow a stable effect's confidence
   interval, but they do not manufacture stability that was not there. The extension to
   \(n = 5\) bought a better point estimate and, per ADR-0024 D1, a decisive answer on whether
   `selected_p` changes; it did not buy significance, and this document does not expect a
   larger \(n\) to buy it cheaply either.
2. **Seed leverage, in two different shapes.** At \(\alpha = 1.0\) the selecting comparison
   (`p0.5` vs `p0`) is still single-seed: 4 of 5 seeds are exact ties, and seed 42 alone
   decides the cell (§3.3). At \(\alpha = 0.1\) leverage no longer sits on one seed — it now
   sits on the *disagreement* between two seeds (21 and 123) that individually swing the mean
   by roughly \(\pm 0.06\), an order of magnitude larger than the cell's own margin. Neither
   shape supports treating the selecting margin as a stable estimate.
3. **Winner's curse into Stage 1b — now resolved as a disclosure obligation.**
   `selected_p` was chosen as the maximum over seven arms, so the \((p = 0.5, \omega = 0.5)\)
   cells are upward-biased estimates of their own performance. ADR-0021 D2 has Stage 1b reuse
   those exact cells while adding \(\omega \in \{0.25, 0.75\}\), and `scripts/analysis.py:1987`
   breaks ties neutral-first (\(\omega = 0.5\) preferred), so the reused arm enters Stage 1b
   with an advantage that has nothing to do with \(\omega\). [ADR-0024](../adr/0024-stage-1a-seed-extension-and-stage-1b-omega-arm.md)
   D2 named the fork — disclose the bias in prose, or redraw all three \(\omega\) arms at
   seeds outside \(\{0, 7, 21, 42, 123\}\) — and deferred the choice until D1 resolved.
   D1 has resolved (this document, at \(n=5\)) without changing \(\hat{p}\).
   [ADR-0026](../adr/0026-stage-1b-omega-winners-curse-disclosed-not-redrawn.md) resolves D2
   against a fresh-seed redraw, on budget grounds, and instead binds every future presentation
   of the Stage 1b \(\omega\) verdict to disclose the bias and the tie rule above, so that an
   \(\hat{\omega} = 0.5\) outcome is read as the structurally expected result of a comparison
   tilted toward it, not as evidence for \(\omega = 0.5\).
4. **Interpretation outrunning evidence.** The mechanisms in §3.1 are plausible and are
   consistent with the two reporting skews, but §3.4 shows the ladder does not order the same
   way at every heterogeneity level. They are not established by this sweep.

---

## 5. Operational Handoff to Stage 1b

[ADR-0024](../adr/0024-stage-1a-seed-extension-and-stage-1b-omega-arm.md) D1, the \(n=5\) extension this document reports, has resolved: `selected_p`
does not change. [ADR-0026](../adr/0026-stage-1b-omega-winners-curse-disclosed-not-redrawn.md)
has resolved ADR-0024 D2, the \(\omega\) winner's-curse question (§4.3): the Stage-1a
\(\omega = 0.5\) cells are reused, not redrawn at fresh seeds, and the reuse's bias is
disclosed rather than removed. Both decisions this document's finalization deferred are
therefore settled. **Stage 1b dispatch itself remains separately gated**: the
`algorithm.p=` write-in into `conf/matrix/power_mean_omega.yaml` and the accompanying
`matrix_contracts.power_mean_omega.sha256` re-registration in
`conf/protocol/replacement-v1.yaml:50` are authorized in principle by ADR-0024 D1 and
ADR-0026, but the write-in itself, and Stage 1b dispatch, each require their own explicit
authorization per `AGENTS.md`'s frozen-config clause — ADR-0026 does not perform either.

1. **Matrix materialization**: `conf/matrix/power_mean_omega.yaml` still carries the
   `algorithm.p=???` placeholder. Writing in the resolved \(\hat{p}\) invalidates
   `matrix_contracts.power_mean_omega.sha256` in `conf/protocol/replacement-v1.yaml:50`;
   the contract must be re-registered in the same commit as the write-in, pending explicit
   authorization to make that change.
2. **The \(\omega = 0.5\) arm**: Stage 1b reuses the Stage-1a cells (ADR-0026 D1); the
   disclosure obligation this carries (ADR-0026 D2) must accompany every future
   presentation of the Stage 1b \(\omega\) verdict.
3. **Host layout**: non-canonical directories on JupyterHub remain preserved in
   `outputs_stash/` at repo root, leaving `outputs/` strictly canonical.

---

## 6. Revision Note

This document replaces a 2026-09-14 draft that reported the same `selected_p = 0.5` with
substantially stronger language — "unambiguously," "both skews independently," "excels under
both severe and mild non-IID conditions" — and dismissed the \(\alpha = 0.3\) contrast as
noise. The selection outcome is unchanged, because it is the output of a pre-registered rule
and that rule fired correctly. The characterization of the evidence behind it is corrected
throughout. The algebra in §3.1 and the accuracy tables in §2 were verified against
`power_mean_degree_selection.json` and carried over unchanged; the percentage-point error in
§3.2 was corrected.

**Second revision (this pass, 2026-09-16).** Per ADR-0024 D1, the power-mean degree arms
were extended from \(n=3\) to \(n=5\) seeds (`{0, 7, 21, 42, 123}`); `selected_p = 0.5` did
not change. This revision replaces every paired-comparison statistic in §3 with the recomputed
\(n=5\) values (independently verified against `power_mean_degree_selection.json`, not carried
over), adds §3.1's theoretical grounding for why the boundedness of the power-mean family
predicts exact-tie degeneracy at \(\alpha=1.0\) and genuine dispersion at \(\alpha=0.1\), and
corrects a conflation in the previous draft's framing of the \(\alpha=0.1\) result: the
\(n=5\) extension shows that comparison is sign-inconsistent (\(t\approx0.20\), 2 of 5 seeds
favor the runner-up), so its non-degeneracy is evidence the pipeline is computing distinct,
non-artifactual outcomes, not evidence that \(p=0.5\) wins at \(\alpha=0.1\). §4 and §5 were
updated to reflect that ADR-0024 D1 is now resolved while D2 (the \(\omega\) winner's-curse
choice) remains open and is the operative blocker before Stage 1b dispatch.
