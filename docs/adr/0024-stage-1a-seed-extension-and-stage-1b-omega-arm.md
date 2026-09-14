# ADR-0024 — Stage 1a extends the degree arms to five seeds; the Stage 1b omega=0.5 reuse is separated from it

**Status**: Accepted · 2026-09-14
**Related**: ADR-0021 (the power-mean family and its two-stage D2 plan), ADR-0023 (the
alpha=0.3 non-selecting contrast and the all-or-nothing closure rule), ADR-0012 (the
iso-byte evaluation rule), ADR-0008 (the three-seed sigma precedent this amends)

## Context

Stage 1a closed at 126 cells and `resolve_power_mean_degree` returned
`selected_p = 0.5` with `rule = "agreement"`. The rule fired exactly as pre-registered;
nothing about the procedure is in question. What is in question is whether three seeds can
carry the weight the downstream campaign puts on that verdict.

Re-deriving the paired statistics directly from
`scripts/analysis_output/power_mean_degree_selection.json` (committed in the preceding
commit for exactly this reason) gives:

- **alpha=0.1**, the selecting cell: `p0.5` beats runner-up `p-0.5` by `+0.0096` mean
  accuracy at paired `t = 0.69` over three seeds. `p0.5` ranks 2nd, 3rd, 1st across seeds
  0, 42, 123; `p-0.5` wins two of the three. The cell verdict rests on seed 123.
- **alpha=1.0**, the labelling cell: `p0.5` beats `p0` by `+0.0027` at paired `t = 1.00`,
  and the entire margin originates at seed 42 — the other two seeds are exact ties. Much of
  this cell is numerically degenerate: at seed 123, `p1`/`p0.5`/`p0`/`p-0.5` return the
  identical float, and `p-min` is the per-seed winner at two of three seeds.
- **alpha=0.3**, the non-selecting contrast: `p0.5` ranks last of seven, and the
  comparisons against it are the only sign-consistent ones in the sweep
  (`p-1 − p0.5 = +0.0257` at `t = 3.83`).

Every comparison favouring the selected `p` sits at `t` between 0.69 and 1.00. The
comparisons that reach conventional strength all run against it, in a cell that by ADR-0023
does not select. The verdict is procedurally sound and evidentially thin, and
`docs/research/2026-09-14-stage1a-power-mean-selection-analysis.md` has been rewritten to
say so.

This ADR is written **after** those numbers were read. That is the hazard it must address
head-on: extending a design after seeing its result is optional stopping unless the
decision rule is fixed in advance of dispatch and is indifferent to which outcome it
produces. The pre-registration this document provides is therefore not a claim that the
extension was planned all along — it plainly was not — but a commitment, made before any
new cell runs, that binds the campaign to the extended result whichever way it falls.

## Decision

### D1 — The seven power-mean degree arms extend from three seeds to five

`conf/matrix/power_mean_design.yaml` gains a per-run `seeds: [0, 42, 123, 7, 21]` override
on the seven `algorithm.p=` arms only. The matrix-level `seeds: [0, 42, 123]` is unchanged
and continues to govern the F0, F3, and F4 control arms.

**The seed values are not newly invented.** `{0, 42, 123, 7, 21}` is this repository's
existing deepening set, used verbatim by `conf/matrix/baseline_tuning.yaml` at lines 121,
141, 165, 184, and 205 for exactly the same purpose — an arm whose sigma other decisions
are judged against, run deeper than the rest of its matrix. Reusing it means the seed
choice rests on established repository practice rather than on a number picked after seeing
the result.

Cell arithmetic: 7 degree arms × 5 seeds × 3 heterogeneities = 105, plus 7 control arms ×
3 seeds × 3 heterogeneities = 63, for **168 cells total** — 42 new. The mechanism is
`expand_matrix`'s per-run `seeds` override (`scripts/common.py:215-270`), which its own
docstring records as "per-run overridable, and that is load-bearing rather than cosmetic."

**The `n = 5` resolution is authoritative and supersedes the `n = 3` resolution,
regardless of which `p̂` it returns.** If the extended sweep selects a degree other than
0.5, that degree becomes `selected_p` for Stage 1b and for every downstream stage, and the
`n = 3` verdict is retained only as superseded history. There is no path by which the
extension is run and then set aside because its answer is unwelcome, and no re-extension to
a third seed count if `n = 5` is also unwelcome. This clause is the whole point of writing
this ADR before dispatch rather than after.

**The extension is a robustness check, not a confirmation run.** At the observed
alpha=0.1 margin (`+0.0096`, paired sigma ≈ 0.024), separating it from zero at conventional
significance would require roughly 37 seeds. Five seeds buys a better point estimate and a
genuine chance the winner changes. It does not buy significance, and this ADR does not
license any later document to claim it did.

### D2 — Adding a fresh `omega=0.5` arm at the Stage-1a seeds is rejected as a no-op

ADR-0021 D2 has Stage 1b sweep `omega ∈ {0.25, 0.75}` at the selected `p` while "reusing
the existing `ω=0.5` cell." Those reused cells won a maximum-over-seven comparison and are
therefore upward-biased estimates of their own performance, and
`select_power_mean_omega_iso_byte` breaks ties neutral-first with `omega=0.5` preferred
(`scripts/analysis.py:1987`). Stage 1b is structurally primed to return `ω̂ = 0.5`.

The obvious repair — re-running an `omega=0.5` arm under the `power_mean_omega` group so
all three arms are "fresh" — **does not work, and is rejected here so that it is not
attempted later.** A re-run at the same `(p̂, ω=0.5)` configuration and the same seeds
`{0, 42, 123}` is the same run. Under this repository's golden-repeatability gates it
reproduces bit-identically, as the F4 `τ=0.3` ≡ F0 exact-float duplication across 8 of 9
Stage-1a cells independently demonstrates. Such an arm relabels the cell without redrawing
it: the selection bias survives untouched and six cells of GPU time are spent for nothing.

The bias is only removed by drawing all three `omega` arms at seeds outside the set used to
select `p̂` — which, after D1, is `{0, 42, 123, 7, 21}`. That is a materially larger and
differently-shaped change than the one considered when this question was first raised, and
its premise (that a fresh `omega=0.5` arm at the existing seeds would fix the bias) is now
known to be false.

**The choice between disclosing the bias in prose and redrawing all three arms at unused
seeds is deferred, and is gated before any Stage 1b dispatch.** It is genuinely downstream:
`p̂` is not final until D1 resolves, and `conf/matrix/power_mean_omega.yaml` cannot have its
`algorithm.p=???` placeholder written in before then. Deferring costs nothing and deciding
now would decide it on a `p̂` that may not survive. Whichever way it resolves, it lands in
its own ADR before a cell runs, because it amends ADR-0021 D2.

Recording the rejection is the operative decision here: the cheap-looking repair is off the
table, and the real fork is named.

## Consequences

- **Closure failure is all-or-nothing and now retroactive.**
  `select_power_mean_degree_iso_byte` raises rather than degrades when a variant's member
  set does not match the expected seed set (`scripts/analysis.py:1826`), and
  `closure_certificate` certifies the `power_mean_design` group as a unit
  (ADR-0023:130-138). The moment the manifest declares 168 cells, a single incomplete new
  cell blocks the Stage-1a verdict **entirely — including the 126 cells that already ran.**
  There is no partial-closure fallback and no way to fall back to the `n = 3` answer by
  re-running selection. This is the reason
  `scripts/analysis_output/power_mean_degree_{selection,resolution}.json` were committed
  before this ADR's matrix edit: once the manifest widens, those two files can no longer be
  regenerated, and `scripts/select_omega.py:22` reads the resolution by default.
- **The already-run 126 cells are not perturbed and do not re-run.** Scoring is per-seed:
  the iso-byte budget is `budget_by_seed[seed]`, set by the cheapest arm within that seed
  (`scripts/analysis.py:1588-1650`); the global `budget_mb` scalar is used only for
  `setter` reporting. Adding seeds 7 and 21 adds entries to that map without altering the
  existing three. Closure matching is by `identity_key`
  (`src/fedmaq/core/run_identity.py:83-95`), which carries dataset, group, config, variant,
  alpha, formulation, and seed but **no canonical index** — so the index shift from
  interleaving new seeds into the expansion does not orphan any completed run.
- **Canonical indices from the previous sweep are void.** `expand_matrix` appends new seeds
  inside each heterogeneity block, so the 42 new cells are interleaved rather than appended
  as a contiguous tail. `--start_at` windows carried over from the 126-cell dispatch are
  meaningless against the new plan. Dispatch is therefore the whole matrix under
  `--skip_completed` (`scripts/run_matrix.py:61`), which no-ops every run already holding a
  final-round checkpoint.
- **The controls stay at three seeds by construction, not by oversight.**
  `power_mean_stage_one` skips every run whose `formulation != "power_mean"` before
  collecting seed sets (`scripts/analysis.py:296-297`), so the one-shared-seed-set
  constraint at lines 304-313 binds only the seven degree arms — which all carry the
  identical five-seed override. Descriptive-control scoring is seed-tolerant (`if s in
  cv_by_seed`), so controls at `n = 3` score cleanly alongside degrees at `n = 5`. No
  selection code changes.
- **Reporting asymmetry must be stated wherever the tables appear.** Degree rows will carry
  `n = 5` and control rows `n = 3` in the same table. Any document presenting them
  side by side says so, and the F4 `τ=0.3` ≡ F0 degeneracy finding is scoped as an `n = 3`
  result, since F4 gains no seeds.
- **Contract chain, in one commit**: `conf/matrix/power_mean_design.yaml` edited;
  `matrix_contracts.power_mean_design` at `conf/protocol/replacement-v1.yaml:49` moved to
  `cell_count: 168` with the recomputed sha256; `docs/recut/power_mean_expected_runs.json`
  regenerated via `dump_expected_runs.py --power-mean-recut`; freeze certificate
  regenerated via `check_freeze.py --write`. As ADR-0023:112-115 established for the
  84→126 widening, `stage_1a`'s `preregistration_sha256` is computed over
  `preregistration_contract()`, which excludes `matrix_contracts` — so this is additive
  with respect to promotability, not a re-registration.
- **`pre-registration-stage1a` is not moved.** That tag points at `11934ba` and records the
  84-cell design. It has already been amended once, by ADR-0023, without re-tagging; this
  ADR is the second such amendment and follows the same practice for the same reason. A
  reader checking out the tag finds 84 cells; ADR-0023 and this document are the record of
  how it reached 168 and why each step was taken.
- **Stage 1b is blocked until D1 resolves.** `conf/matrix/power_mean_omega.yaml` keeps its
  `algorithm.p=???` placeholder, and `select_omega.py` will read whatever
  `power_mean_degree_resolution.json` says after the extended selection runs. Its own
  closure check certifies only the 12-cell `power_mean_omega` group
  (`scripts/select_omega.py:74`) and is not gated by the widened Stage-1a manifest.
