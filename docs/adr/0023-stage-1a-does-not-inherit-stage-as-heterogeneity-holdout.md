# ADR-0023 — Stage 1a does not inherit Stage A's heterogeneity holdout; alpha=0.3 is added as a non-selecting robustness check

**Status**: Accepted · 2026-09-13
**Related**: ADR-0011 (the matched-tuning holdout this decision is compared against), ADR-0021 (the power-mean family Stage 1a selects over)

## Context

[Issue #110](https://github.com/FedMAQ/fedmaq-experiments/issues/110) flags an
asymmetry: Stage A holds out `alpha=0.3` for tuning so no algorithm is tuned at
a heterogeneity level it is later reported at; Stage 1a
(`conf/matrix/power_mean_design.yaml`) selects the power-mean degree `p` at
`alpha ∈ {0.1, 1.0}` — the same two values `benchmark_grid.yaml` later reports
at. The split is `val` in both cases, so no reserved-test evidence is tainted;
the question is purely whether Stage A's discipline should extend one stage
later.

Stage 1a's selection mechanism is not a placeholder to be designed — it is
already implemented, in `resolve_power_mean_degree`
(`scripts/analysis.py:1947-1976`). That function hardcodes
`required = {0.1, 1.0}`, always selects the `alpha=0.1` ("severe skew")
winner as `selected_p`, and uses the `alpha=1.0` winner only to label the
outcome `"agreement"` or `"severe-skew tie-break"`. `docs/agents/execution-
model.md` calls this "the severe-skew rule" and it was written, and this
selection axis fixed, before any Stage-1a cell was dispatched — it is itself a
pre-registered decision, just one that #110 correctly observes was never
written down at the ADR level.

This ADR-level transcription itself lands while the currently-dispatched
84-cell Stage-1a sweep is still in flight, after #110's own acceptance
criterion asked for a decision recorded before Stage 1a executes. That
criterion is met at the level that actually governs the outcome — the rule
was fixed pre-dispatch, in `resolve_power_mean_degree` and in `execution-
model.md` — but not at the ADR level, and this document does not claim
otherwise. No result from any dispatched Stage-1a cell has been read in
writing it; the record is late, not informed by outcomes.

## Decision

### Keep the existing selection rule; do not extend Stage A's holdout to it

Stage 1a selects `p` at `alpha ∈ {0.1, 1.0}` exactly as `resolve_power_mean_
degree` already implements, unchanged. Two reasons, both from #110's own
"Keep it" framing:

- The power-mean degree and weight are properties of the formulation family,
  not of a heterogeneity level. Selecting the best formulation for the regime
  actually being reported, on validation data, is a legitimate use of that
  data — it is not the same act as tuning a per-algorithm hyperparameter to
  the reported setting, which is what Stage A's holdout guards against.
- Changing the selection axis now, with the sweep already dispatched under the
  existing rule, would mean amending a decision that was itself already fixed
  pre-dispatch (in code and in `execution-model.md`) rather than genuinely
  undecided. Doing so would also reopen the identical question for Stage 1b's
  omega selection, which carries the same severe-skew mechanism
  (`scripts/analysis.py:2182`) — turning one contested selection axis into
  two, mid-campaign.

### alpha=0.3 is added as a non-selecting robustness check, not a holdout

`power_mean_design.yaml` gains a third heterogeneity, `dirichlet_alpha_0.3`,
widening the matrix from 84 to 126 cells (14 runs × 3 seeds × 3 alphas). This
is **not** the holdout #110's "Change it" branch describes — the alpha=0.3
cells do not feed `resolve_power_mean_degree`, which still reads only
`entries[0.1]` and `entries[1.0]`. Their purpose is narrower: `select_power_
mean_degree_iso_byte` already computes a per-`(dataset, alpha)` winner
generically, so alpha=0.3 cells produce a third winner entry for free. This
lets the resolution report, empirically, whether the alpha=0.1-selected `p`
also wins at a skew absent from `benchmark_grid`'s reporting grid — answering
the inflation concern #110 raises with data rather than argument, without
changing what determines `selected_p`.

### The asymmetry is intentional and now recorded

Stage A's holdout exists because a per-algorithm knob (e.g. `fedprox`'s `mu`)
tuned at the reported alpha would inflate that algorithm's own reported
result. Stage 1a selects a structural property of one formulation family
against itself, at the regime it is reported in — the selection is "which
degree serves this regime best," not "what value makes this run look best
after the fact." An examiner who notices the asymmetry should find it answered
here, not on the spot.

## Consequences

- `conf/matrix/power_mean_design.yaml` widens to 126 cells; `conf/protocol/
  replacement-v1.yaml`'s `matrix_contracts.power_mean_design` moves to
  `{cell_count: 126, sha256: f22ebf8e2f883676bbd16dbfc2c550df0df6bc4ef5093996c7974aada1fe3bfd}`.
  `stage_1a`'s `preregistration_sha256` (computed over `preregistration_
  contract()`, which does not include `matrix_contracts`) is unchanged — the
  84 already-run cells and the 42 new cells stamp an identical value, so this
  is additive with respect to promotability, not a re-registration.
- `resolve_power_mean_degree` is unmodified. Selection continues to run on
  `alpha ∈ {0.1, 1.0}` exactly as before this ADR.
- `docs/agents/execution-model.md`'s Stage 1a section is updated to state the
  severe-skew rule's actual mechanics (selects on 0.1, labels via 1.0) and the
  alpha=0.3 cells' non-selecting role, so the gate document matches the code
  it certifies.
- Stage 1b (`power_mean_omega.yaml`, 12 cells) is unaffected: its own
  severe-skew omega selection at `alpha ∈ {0.1, 1.0}` is out of this ADR's
  scope and carries the identical, already-registered asymmetry, left
  undisturbed for the same reasons given above.
- If a future contributor wants alpha=0.3 (or any held-out skew) to actually
  determine `selected_p`, that is a new decision requiring a code change to
  `resolve_power_mean_degree` and its own ADR — not an implication of this
  one.
- `scripts/select_power_mean.py:48` certifies closure against the single
  `power_mean_design` group in `power_mean_expected_runs.json`, all-or-nothing
  (`closure_certificate`, `scripts/analysis.py:1446`). That manifest now lists
  126 cells, so `select_power_mean.py` refuses to run selection until all 126
  are present — the already-dispatched 84 do not unlock a verdict on their
  own. Concretely: the currently-running 84-cell sweep finishing is not
  sufficient to produce a Stage-1a selection; the new 42 alpha=0.3 cells must
  also be dispatched and complete first. There is no partial-closure path
  that resolves the original 84 ahead of the rest.
- Stage 1b's own closure check (`scripts/select_omega.py:74`) certifies only
  its own 12-cell `power_mean_omega` group and does not re-certify Stage 1a,
  so it is not gated by the widened manifest. `resolve_power_mean_omega`
  (`scripts/analysis.py:2153-2183`) mirrors `resolve_power_mean_degree`
  exactly — it reads only `entries[0.1]`/`entries[1.0]` from a generic
  per-`(dataset, alpha)` grouping — so the alpha=0.3 cells Stage 1b reuses
  from Stage 1a's `omega=0.5` rows are ignored by omega resolution the same
  way, confirmed by reading both functions in full rather than inferred from
  symmetry.
