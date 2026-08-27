# ADR-0021 — The soft-quality design space is one swept power-mean family, not four sampled rules

**Status**: Accepted · 2026-08-27 (recorded 2026-08-28)
**Supersedes**: ADR-0012's *candidate set* and its selection subject. ADR-0012's criterion, its iso-byte amendment, and its frozen v1 verdict stand as the record of the superseded bundle.
**Related**: ADR-0011 (the symmetric-sweep obligation this discharges for the contribution's own parameters); ADR-0010 and the `pre-registration` tag (what the invalidation costs); ADR-0019 (the other change invalidating v1 artifacts in the same window)

## Context

This ADR was written on 2026-08-28, a day after the decision, because
[#11 pass 22](https://github.com/FedMAQ/fedmaq-experiments/issues/11) found that
**no ADR recorded the change at all** — zero hits for `power-mean` or `#34` across
ADR-0001 to ADR-0020 — while ADR-0012 still read as the live formulation decision.
A reader of `docs/adr/` alone would have concluded the five-candidate design was
current. The decision itself is [#34](https://github.com/FedMAQ/fedmaq-experiments/issues/34),
decided 2026-08-27 with the author's explicit acceptance of full artifact
invalidation; this file records it, it does not re-open it.

ADR-0012 fixed a selection rule over **five candidates**: F0 (resource-only hard
cap, the non-adaptive control) plus F1–F4, four separately-defined combination
rules. #32 reframed those four as an *aggregation taxonomy* — where each sits on
an operator/compensation axis — and reading them that way exposed the problem:

- F1 (weighted arithmetic, fully compensatory) and F2 (multiplicative/geometric)
  are **two sampled points on one continuous axis**, not two designs.
- The taxonomy's **harmonic** and **minimum** cells were never covered at all.
- Omitting `min` was an inconsistency *inside FedMAQ's own design*: Tier 1
  already uses `min()` for its hard clamp, so excluding it from Tier 2 was
  arbitrary rather than argued.

#33 had already committed to sweeping each formulation's own parameter (Option C,
≈90 cells) precisely because ADR-0011's logic cuts both ways: sweeping baseline
knobs five ways (#27) while the contribution's parameters stayed at one value
would have been the mirror image of the attack ADR-0011 exists to answer. But
sweeping four *separate* rules still treats the compensation axis as four
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
| D1 | Which `p` ladder | `p ∈ {1, 0.5, 0, −0.5, −1, −2}` plus the minimum limit — the named operators as landmarks, with intermediate points so the curve is readable rather than four labels |
| D2 | How `ω` interacts | **Two-stage, not a joint grid.** Stage 1 sweeps `p` at `ω=0.5` (14 arms × 2 skews × 3 seeds = 84 cells); stage 2 adds `ω ∈ {0.25, 0.75}` at the selected `p`, reusing the `ω=0.5` cell (96 total). Assumes separability; a full product does not fit the cell budget |
| D3 | Limit semantics | `p→0` and `p→−∞` are **special-cased exactly**, never approximated with a small `p`. A zero in either active signal yielding `s_k = 0` for `p ≤ 0` is *intended non-compensatory semantics*, not a division-by-zero |
| D4 | Ablation signal-removal | The `ω=1` / `ω=0` endpoints, well-defined at every `p`. **The old `x⁰ = 1` convention on F2's unconstrained exponents is retired**, so no `0^0` rule is required |
| D5 | Config keys | `gamma1`/`gamma2` disappear; the keys are `p` and `omega`. **A single `ω`, not `ω₁, ω₂`** |
| D6 | Manuscript narrative | "Four candidate formulations, one selected" becomes "a swept family plus two structurally different rules." Rewritten, not patched — #35 |

**F0, F3 and F4 are kept and are not folded into the family.** F3 (gradient-primary,
data-modulated, `κ`) varies a different axis; F4 (threshold-based staged rule,
`τ_g, τ_n`) is discrete. Neither is reachable by any value of `p`, and the
implementation keeps them as separate branches keyed `3`/`4` against the string
`"power_mean"` rather than as parameterizations.

**On introducing a `Supersedes`/`Superseded by` marker.** This repo had no such
convention — pass 22 checked all twenty prior ADRs and found none. One is adopted
here rather than leaving ADR-0012 unmarked, because the alternative failure is
worse: an accepted ADR silently describing a design that no longer exists. The
marker points *forward only* and does not edit ADR-0012's body, consistent with
this project's standing rule that outcomes are appended, never substituted.

## Consequences

- **Full v1 artifact invalidation, accepted deliberately by the author.** Every
  Study-1 cell that depended on the F1/F2 arms is superseded. This lands in the
  same window as ADR-0019's quantizer change, so #22's rerun absorbs both.
- **The `pre-registration` tag at `0dd7ef1` no longer covers the method.** It
  only ever covered the superseded v1-provisional bundle. **A new Stage-1a
  pre-registration tag must be cut before #22 dispatches** — this is a live
  gate, not a formality, and as of 2026-08-28 it has not been cut.
- **#20 shrinks.** D5 absorbs most of it inside the #34 rewrite; only F3's
  `lambda_val` → `kappa` and F4's thresholds remain. #20's title still names
  `omega1/omega2` — the retired dual-weight notation — and is stale on its face.
- **ADR-0012 is not wholly dead.** Its frozen `formulation: 2` remains live in
  `conf/algorithm/fedmaq.yaml` behind the `pre-registration` tag, and its
  iso-byte amendment is unaffected by this change. What is superseded is the
  *candidate set* the selection ranged over, and therefore the meaning of its
  verdict — **"Formulation 2 is not assumed to survive re-selection"** (#21).
- **Chapters 5 and 6 are not yet migrated.** Both still instruct their results
  sections to report a one-of-five selection outcome. They cannot be rewritten
  until the sweep produces results; #11 pass 22 enumerates the eleven sites.
- **`conf/algorithm/fedmaq.yaml`'s comment cites "the §4.3.6 formulation study's
  verdict."** That section number no longer resolves — the formulation study is
  §4.4 after #35's insertion. Cosmetic, but it will mislead.
- **The compensation axis becomes a reportable curve**, not four points. This is
  the substantive gain: the thesis can now argue *how much* compensation between
  training-state and data-richness signals helps, rather than only which of four
  named rules won.
