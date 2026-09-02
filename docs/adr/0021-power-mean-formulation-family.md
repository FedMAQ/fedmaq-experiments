# ADR-0021 — The soft-quality design space is one swept power-mean family, not four sampled rules

**Status**: Accepted · 2026-08-27 (recorded 2026-08-28)
**Supersedes**: ADR-0012's *candidate set* and its selection subject. ADR-0012's criterion, its iso-byte amendment, and its frozen v1 verdict stand as the record of the superseded bundle.
**Related**: ADR-0011 (the symmetric-sweep obligation this discharges for the contribution's own parameters); ADR-0010 and the `pre-registration` tag (what the invalidation costs); ADR-0019 (the other change invalidating v1 artifacts in the same window)

## Scope

This ADR is the current formulation-design authority. It records the author's
resolved decision in [Issue #34](https://github.com/FedMAQ/fedmaq-experiments/issues/34)
and does not re-open it. The former v1 candidate set and verdict remain historical
in [ADR-0012](0012-formulation-selection-and-the-iso-byte-amendment.md).

ADR-0012 fixed a selection rule over **five candidates**: F0 (resource-only hard
cap) plus F1–F4, four separately-defined combination rules. Treating those rules as
an aggregation taxonomy exposed the problem:

- F1 (weighted arithmetic, fully compensatory) and F2 (multiplicative/geometric)
  are **two sampled points on one continuous axis**, not two designs.
- The taxonomy's **harmonic** and **minimum** cells were never covered at all.
- Omitting `min` was an inconsistency *inside FedMAQ's own design*: Tier 1
  already uses `min()` for its hard clamp, so excluding it from Tier 2 was
  arbitrary rather than argued.

The contribution's parameters require the same symmetric treatment as baseline
knobs in ADR-0011. Four separate rules still treat one compensation axis as four
disconnected labels.

## Decision

**The Tier-2 soft-quality signal is a weighted generalized power mean over one
swept compensation degree, plus two structurally separate rules and a control.**

```
s_k = M_p(g̃_k, ñ_k; ω) = (ω·g̃_k^p + (1−ω)·ñ_k^p)^(1/p)
```

The family *contains* the retired F1 and F2 as limiting cases — arithmetic at
`p=1`, geometric as `p→0` — along with harmonic (`p=−1`) and minimum (`p→−∞`),
which F1–F4 never reached. **`p` is a compensation-degree parameter, not a second
formulation identifier.** Larger `p` permits more compensation between signals;
increasingly negative `p` requires both to remain high.

Resolving #34's six open design questions:

| | Question | Resolution |
| :-- | :-- | :-- |
| D1 | Which `p` ladder | `p ∈ {1, 0.5, 0, −0.5, −1, −2}` plus the minimum limit — named operators with intermediate points |
| D2 | How `ω` interacts | **Two-stage, not a joint grid.** Sweep `p` at `ω=0.5`, then add `ω ∈ {0.25, 0.75}` at the selected `p`, reusing the existing `ω=0.5` cell |
| D3 | Limit semantics | `p→0` and `p→−∞` are **special-cased exactly**, never approximated with a small `p`. A zero in either active signal yielding `s_k = 0` for `p ≤ 0` is *intended non-compensatory semantics*, not a division-by-zero |
| D4 | Ablation signal-removal | The `ω=1` / `ω=0` endpoints, well-defined at every `p`. **Polarity is explicit**: `ω` weights the state/gradient signal `g̃_k` while `1−ω` weights the data-volume signal `ñ_k`. Therefore, `ω=1.0` isolates the state signal and removes data volume (`fedmaq_no_data`), while `ω=0.0` isolates data volume and removes the state signal (`fedmaq_no_state`). The family contains Formulation 2 as an algebraic identity at the center `(p=0, ω=0.5)` with shipped exponents `γ₁=0.5, γ₂=0.5`. The old `x⁰ = 1` convention on F2's unconstrained exponents is retired, so no `0^0` rule is required |
| D5 | Config keys | `gamma1`/`gamma2` disappear; the keys are `p` and `omega`. **A single `ω`, not `ω₁, ω₂`** |
| D6 | Manuscript narrative | "Four candidate formulations, one selected" becomes "a swept family plus two structurally different rules." Rewritten, not patched — #35 |

**F0, F3 and F4 are kept and are not folded into the family.** F3 (gradient-primary,
data-modulated, `κ`) varies a different axis; F4 (threshold-based staged rule,
`τ_g, τ_n`) is discrete. Neither is reachable by any value of `p`, and the
implementation keeps them as separate branches keyed `3`/`4` against the string
`"power_mean"` rather than as parameterizations.

The `Supersedes` marker points forward and does not rewrite ADR-0012's historical
body. This preserves the v1 vocabulary while making the current authority explicit.

## Consequences

- Issue #34's author-accepted decision crosses the full-v1 invalidation boundary.
  The formulation recut and the concurrent ADR-0019 quantizer correction make all
  v1 campaign artifacts historical and ineligible to authorize or substitute for
  recut evidence. The v1 config, evidence, and vocabulary remain interpretable
  under ADR-0012; they are not rewritten or silently reused.
- The old `pre-registration` tag cannot authorize the recut. A new Stage-1a
  registration and its audit are required before confirmatory dispatch; the
  matrix, resolved configs, and evidence define that gate.
- F0, F3, and F4 remain structurally separate; `kappa` and the threshold constants
  are not power-mean parameters.
- The compensation axis is reported as a curve rather than as four disconnected
  formulation labels.
