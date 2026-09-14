# Analysis of Stage 1a Power-Mean Degree Selection (\(p = 0.5\))

**Date**: 2026-09-14  
**Status**: **Provisional.** The selection rule has fired and `selected_p = 0.5` is the
procedural outcome, but the margins that produced it are not separable from seed noise at
\(n = 3\). A registered seed extension to \(n = 5\) supersedes this document's verdict
([ADR-0024](../adr/0024-stage-1a-seed-extension-and-stage-1b-omega-arm.md)).  
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

- **Selected formulation parameter**: **\(p = 0.5\)**
- **Selection rule outcome**: **`"rule": "agreement"`** — the \(\alpha = 0.1\) winner and the
  \(\alpha = 1.0\) winner coincide, so no severe-skew tie-break was invoked.

The rule fired as pre-registered and the outcome stands procedurally. What the data does
**not** support is a claim that \(p = 0.5\) was separated from its rivals by the evidence.
Both winning margins are well inside seed noise, and \(p = 0.5\) is not the per-seed winner
in most cells:

| Cell | Mean margin over runner-up | Paired \(t\) (\(n=3\)) | Per-seed rank of `p0.5` |
| :--- | :---: | :---: | :--- |
| \(\alpha = 0.1\) vs `p-0.5` | \(+0.0096\) | \(0.69\) | 2nd, 3rd, 1st |
| \(\alpha = 1.0\) vs `p0` | \(+0.0027\) | \(1.00\) | 1st, 3rd, 3rd |

Neither margin approaches significance. The \(\alpha = 0.1\) result rests on a single seed
(123), where `p0.5` reaches \(0.2973\) against a cell mean near \(0.27\); on the other two
seeds `p-0.5` and `p0` respectively beat it. The \(\alpha = 1.0\) "agreement" is weaker still,
for reasons given in §3.3: much of that cell is numerically degenerate, and the entire
margin over `p0` comes from seed 42 alone.

At \(\alpha = 0.3\) — the non-selecting robustness contrast ([ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md)) — `p0.5` ranks **last of seven**, and the
comparisons against it are the only ones in the sweep that approach significance (§3.4).
That cell does not and cannot change `selected_p`, but it is the sharpest signal in the data
and is reported here rather than minimized.

**Read this document as: the pre-registered procedure selected \(p = 0.5\), on margins the
three-seed design cannot resolve.** ADR-0024 registers the extension that will either
confirm or overturn it.

---

## 2. Empirical Performance Across Skews

Mean top-1 validation accuracy at the minimum common cumulative-MB budget (\(B\)), with
dispersion across seeds 0, 42, 123. Dispersion at \(n = 3\) carries roughly \(\pm 50\%\) of
itself; the \(\pm\) figures below are descriptive, not confidence intervals.

| Degree \(p\) | Operator Category | \(\alpha = 0.1\) (\(B \approx 8058\) MB) | \(\alpha = 1.0\) (\(B \approx 8198\) MB) | \(\alpha = 0.3\) Robustness Contrast (\(B \approx 8130\) MB) |
| :--- | :--- | :---: | :---: | :---: |
| **`p1`** | Arithmetic mean (linear, fully compensatory) | \(0.2716 \pm 0.0216\) | \(0.5176 \pm 0.0097\) | \(0.4186 \pm 0.0200\) |
| **`p0.5`** | Root-mean (concave, mildly compensatory) | **\(0.2863 \pm 0.0097\)** ★ | **\(0.5331 \pm 0.0174\)** ★ | \(0.4167 \pm 0.0181\) (last) |
| **`p0`** | Geometric mean (multiplicative, historical v1 F2) | \(0.2699 \pm 0.0205\) | \(0.5305 \pm 0.0198\) | \(0.4254 \pm 0.0183\) |
| **`p-0.5`** | Sub-geometric mean | \(0.2767 \pm 0.0146\) | \(0.5061 \pm 0.0228\) | \(0.4407 \pm 0.0180\) |
| **`p-1`** | Harmonic mean | \(0.2630 \pm 0.0189\) | \(0.5269 \pm 0.0192\) | **\(0.4425 \pm 0.0291\)** ★ |
| **`p-2`** | Sub-harmonic mean | \(0.2572 \pm 0.0201\) | \(0.5249 \pm 0.0150\) | \(0.4373 \pm 0.0224\) |
| **`p-min`** | Leontief minimum operator | \(0.2498 \pm 0.0247\) | \(0.5266 \pm 0.0043\) | \(0.4297 \pm 0.0493\) |

★ marks the per-cell mean winner. The \(\alpha = 0.3\) winner is `p-1`; per ADR-0023 that cell
does not feed `resolve_power_mean_degree`.

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

**Reporting asymmetry (forward-looking).** Once ADR-0024's extension lands, the degree rows
above will carry \(n = 5\) while every control row remains at \(n = 3\); the extension adds
seeds to the seven power-mean arms only. The F4 \(\equiv\) F0 finding is and remains an
\(n = 3\) result.

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

### 3.2 Relation to Historical v1 (Formulation 2)

The superseded v1 campaign ([ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md)) froze Formulation 2 (geometric / multiplicative,
\(p = 0\)). On the continuous ladder, \(p = 0.5\) outscores \(p = 0\) in mean accuracy at
both reporting skews:

- \(\alpha = 0.1\): \(+0.0164\) **accuracy, i.e. \(+1.64\) percentage points** (about
  \(+6.1\%\) relative to \(p=0\)'s \(0.2699\)); paired \(t = 0.96\).
- \(\alpha = 1.0\): \(+0.0027\), i.e. \(+0.27\) percentage points (about \(+0.5\%\)
  relative); paired \(t = 1.00\), and the entire margin originates at seed 42 — the other
  two seeds are exact ties.

The earlier draft of this document reported these as "\(+1.64\%\) accuracy" and
"\(+0.27\%\) accuracy," conflating percentage points with relative change. The corrected
figures are above.

Neither margin is distinguishable from noise at \(n = 3\). The defensible statement is that
\(p = 0.5\) is **not worse** than the historical geometric choice at either reporting skew,
and that the ladder subsumes it as a special case.

### 3.3 Numerical Degeneracy at \(\alpha = 1.0\)

The mild-skew cell carries much less information than its row in §2 suggests, because many
arms return bit-identical accuracies:

- **Seed 123**: `p1`, `p0.5`, `p0`, and `p-0.5` all return exactly \(0.5216757070891007\).
  Four of seven degrees are indistinguishable.
- **Seed 42**: `p1`, `p0.5`, both `f4-threshold-0.3` and `f4-threshold-0.5`, and
  `resource-only` all return exactly \(0.5245548011615987\).
- **Seed 0**: `p0.5` and `p0` tie exactly; `p1`, `f4-threshold-0.3`, and `resource-only` tie
  exactly at a lower value.

At \(\alpha = 1.0\) the client distribution is close enough to uniform that the allocation
policy frequently produces the same bit-width plan regardless of \(p\), and identical plans
give identical trajectories under the repeatability gates. Consequently `p0.5` is the
per-seed winner in only one of three seeds here — `p-min` takes the other two — and the
cell's "agreement" label reflects a mean driven by seed 0.

One comparison in this cell does reach a large \(t\): `p0.5` \(-\) `p-1` gives mean
\(+0.0062\) at \(t = 5.57\). That is an artifact of a very small paired standard deviation,
not a meaningful effect — a consistent six-tenths of a percentage point. It is reported for
completeness and is not offered as support for the selection.

### 3.4 The \(\alpha = 0.3\) Contrast Is the Strongest Signal in the Sweep

Per [ADR-0023](../adr/0023-stage-1a-does-not-inherit-stage-as-heterogeneity-holdout.md), \(\alpha = 0.3\) was added as an empirical contrast that does not govern
selection, and `power_mean_degree_resolution.json` accordingly carries no \(\alpha = 0.3\)
field. That protocol position is unchanged. Its content should not be softened:

- `p0.5` ranks **7th of 7** on cell mean, and 6th, 7th, 6th per seed. It does not win a
  single seed.
- The comparisons against it are sign-consistent across all three seeds and carry the
  largest *effects* in the sweep: \(p_{-1} - p_{0.5}\) gives \(+0.0257\) at
  \(t = 3.83\) and \(p_{-0.5} - p_{0.5}\) gives \(+0.0240\) at \(t = 3.08\). The case rests
  on those two. \(p_{0} - p_{0.5}\) also runs the same way, at \(+0.0087\), and its
  \(t = 5.01\) is listed only for completeness: on a margin that small the statistic is
  driven by a near-zero paired standard deviation, exactly the artifact §3.3 discounts at
  \(t = 5.57\), and it earns no more weight here than it was given there.

The earlier draft described this as "slightly higher mean accuracies … within the
overlapping noise bounds." That is not supportable: unlike every margin favoring
\(p = 0.5\), these differences do not change sign across seeds. Comparing dispersion bars is
the wrong test here, because the comparison is paired — the same seeds run both arms.

Read plainly, the sweep says the compensatory interior of the ladder is favored at the two
reporting skews and the non-compensatory end is favored at the intermediate skew, on margins
that are weak in the first case and consistent in the second. Whether that is a real regime
dependence or three-seed noise is exactly what ADR-0024's extension is meant to settle. It
is a genuine finding for the discussion chapter, not a footnote.

---

## 4. Threats to This Selection

1. **Underpowering.** The \(\alpha = 0.1\) selection margin is \(+0.0096\) with paired
   \(\sigma \approx 0.024\). Separating it from zero at conventional significance would need
   on the order of 37 seeds. The extension to \(n = 5\) buys a better point estimate and a
   real chance the winner changes; it does not buy significance, and no reader should expect
   it to.
2. **Single-seed leverage.** Both selecting cells turn on one seed each — 123 at
   \(\alpha = 0.1\), 42 at \(\alpha = 1.0\).
3. **Winner's curse into Stage 1b.** `selected_p` was chosen as the maximum over seven arms,
   so the \((p = 0.5, \omega = 0.5)\) cells are upward-biased estimates of their own
   performance. ADR-0021 D2 has Stage 1b reuse those exact cells while adding
   \(\omega \in \{0.25, 0.75\}\), and `scripts/analysis.py:1987` breaks ties neutral-first
   (\(\omega = 0.5\) preferred). The reused arm therefore enters Stage 1b with an advantage
   that has nothing to do with \(\omega\). ADR-0024 addresses this directly.
4. **Interpretation outrunning evidence.** The mechanisms in §3.1 are plausible and are
   consistent with the two reporting skews, but §3.4 shows the ladder does not order the same
   way at every heterogeneity level. They are not established by this sweep.

---

## 5. Operational Handoff to Stage 1b

Stage 1b **cannot dispatch on this document's verdict.** The `algorithm.p=` write-in for
`conf/matrix/power_mean_omega.yaml` is blocked behind the ADR-0024 extension resolving at
\(n = 5\), since that resolution supersedes `selected_p` regardless of which value it
returns.

1. **Matrix materialization**: `conf/matrix/power_mean_omega.yaml` still carries the
   `algorithm.p=???` placeholder. Writing in the resolved \(\hat{p}\) invalidates
   `matrix_contracts.power_mean_omega.sha256` in `conf/protocol/replacement-v1.yaml:50`;
   the contract must be re-registered in the same commit as the write-in.
2. **The \(\omega = 0.5\) arm**: whether Stage 1b reuses the Stage-1a cells or runs fresh
   ones is settled in ADR-0024, not here. Threat 3 above is the reason it is open.
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
