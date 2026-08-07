# ADR-0009 — Run identity: a run is its matrix and its config, never its algorithm name

**Status**: Accepted · 2026-07-22 through 2026-08-01
**Supersedes**: `docs/DECISIONS.md` Decisions 39, 54–57, 71, 76 (file deleted; see ADR-0014)

## Context

Every §4.3.7 ablation arm declares `name: fedmaq` so they all dispatch the same
strategy hook. `algorithm.name` therefore names the *hook*, and collapses six of
seven net-new arms onto one value. Analysis code keyed on it. The same
under-identification shows up in the output path, which keys on the algorithm
rather than the run label, so a matrix sweeping an override across one algorithm
writes every cell into one directory and keeps only the last — silently, with the
sweep exiting 0 and reporting its full task count.

This is one defect seen from several ends: **runs were identified by too little.**

## Decision

### Canonical output paths and declarative matrices

All experiment artifacts land under
`outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<heterogeneity>/seed_<seed>/`,
eliminating the ambiguity between `experiments/`, `multirun/` and `outputs/`.
Phases carry standardized round counts: `ci` (R=2), `smoke` (R=50, single seed),
`explore` (R=50, single seed), `formal` (R=100, 3 seeds).

Seventeen single-purpose ad-hoc runner scripts were replaced by declarative YAML
manifests in `conf/matrix/`, driven by `scripts/run_matrix.py` + `scripts/common.py`
— centralized cross-platform Ray cleanup, process-isolated execution, `--dry_run`,
and resumption. Hydra `--multirun` is never used (ADR-0004).

### What identifies a run

**A run is identified by the config file it composed and the matrix that dispatched
it.** `RunRecord` carries:

| Field | Source | Why it exists |
| --- | --- | --- |
| `algorithm_config` | `runtime.choices.algorithm` in `.hydra/hydra.yaml` | separates arms sharing one hook |
| `experiment_group` | canonical path segment | scopes analyses to their own study |
| `phase` | canonical path segment | separates explore from formal |
| `post_process` | resolved config | the comparison regime (ADR-0004) |
| `variant` | `<algorithm>__<variant>` path segment | separates cells differing only by override |

`manifest.py`'s `_algorithm_config_name`, never populated since it was added, now
carries the same value — the manuscript asks readers to confirm arm parity from the
run manifests.

**`get_canonical_output_dir` takes an optional `variant`**, appended as
`<algorithm>__<variant>`. Without it, three exploration matrices dispatched every
cell of a stage into a single directory: 26 tasks resolved to 5 distinct
directories, 8 to 5, and 4 to 1. Nothing failed loudly; the loss surfaces only at
analysis time as cells that appear never to have run — and `--skip_completed` makes
recovery *worse*, since after the first cell at a seed writes its checkpoint the
others resolve to the same directory and are skipped as complete.

`test_no_matrix_dispatches_two_runs_into_one_output_directory` composes every file
in `conf/matrix/` and asserts task count equals distinct-directory count. It is
written across all matrices rather than the three that were wrong: the next matrix
to sweep an override on one algorithm has no reason to know the rule exists.

### Analyses scope on the group, never on the algorithm

`select_winner` selected candidates as `algorithm_config == "fedmaq" and
formulation is not None`, excluding only the ablation group. Three populations
satisfy that and should not:

1. `pass2_factorial` and `pass3_freeze_confirm`, which dispatch `algorithm=fedmaq`
   at the held-out α = 0.3 where no reference exists by design — these manufacture
   a verdict at a skew the design excludes.
2. **The grid's own FedMAQ rows** — same dataset, skews and seeds as the study,
   differing only by `post_process=true`. Pooled into a formulation's cell they
   credit the *incumbent* with the pipeline savings the study exists to withhold,
   so the contamination flatters the status quo rather than perturbing it randomly.
3. The same collision in `compare_to_baselines`, resolved by dict-iteration order,
   which could report pipeline-free FedMAQ against pipeline-era baselines — breaking
   the regime rule in the one direction that understates FedMAQ.

All three close by scoping on `experiment_group`: candidates from
`formulation_study`, floor and headline comparison from `benchmark_grid`.
`compare_to_baselines` no longer takes its targets from the winner verdict at all —
it enumerates the grid and uses the verdict only to assert the grid ran the frozen
formulation, printing a `FREEZE DRIFT` line when it did not. Test fixtures carry an
`experiment_group`, since a fixture without one describes a run no matrix could
produce.

### The accuracy floor was circularly dependent on the grid it configures

The formulation study's target accuracy is defined as a fraction of the
uncompressed FedAvg reference, "reusing the FedAvg runs already present in the
benchmark grid" — but FedAvg appeared only in `benchmark_grid.yaml`, dispatched
*after* the tag that freezes the formulation those very runs are needed to select.
Running the analysis at the point the runbook says to resolve the verdict raised
`ValueError: No FedAvg reference runs found` before writing anything.

**Resolved by pulling the grid's CIFAR-10 `fedavg` rows forward into a new Stage
1c**, keeping the grid's `experiment_group`, overrides and output directories so
the later grid sweep's `--skip_completed` passes over them. Rows *moved*, not added.

*Rejected: a dedicated FedAvg reference cell inside the formulation study at
`phase: explore`.* It keeps every confirmatory run behind the tag, which is the
honest attraction, but creates two FedAvg populations at the same config, skew and
seeds — the floor computed from one, the grid reported from the other. A reader
recomputing the floor from the published table would then be entitled to a
different disqualification set than the one that actually fired, which is a worse
property for a pre-registered rule than an ordering exception. The exception is
admissible and disclosed: FedAvg is invariant to all three artifacts the tag locks
— no refinement layer, no formulation, no tunable constant, which is also why
ADR-0011 excludes it as the uncompressed control. **It extends to nothing else**,
and specifically not to the ablation arms, which are removals from frozen FedMAQ.

### Ablation table assembly refuses to report an invalid design

`build_ablation_table()` assembles the ablation's table — every configuration at
both skews, accuracy and cumulative MB as mean ± seed SD, the per-arm formulation
column so a fallback arm is visible in the table rather than only in the prose, and
the pipeline regime as a note. `parity.attributable` goes **false** on a pipelined
arm, a refinement-layer deviation not recorded as inapplicable, a missing seed, or
a fallback arm whose formulation-study anchor does not exist yet. That last check is
what the calendar ordering exists to guarantee, now verified rather than assumed.

The formulation study is dispatched by `conf/matrix/formulation_study.yaml`, not a
hand-typed `--multirun`; it had been a comment in `conf/config.yaml`, which put its
runs in a date-keyed tree with no `experiment_group`.

## Consequences

- **Never key an analysis on `algorithm.name`.** Scope on `experiment_group`, and
  identify an arm by `algorithm_config`.
- Any matrix that lists one algorithm twice under different overrides **must** set
  `variant`. The guard test fails in CI rather than at analysis time.
- Stage 1c's FedAvg rows are the single disclosed ordering exception. Do not extend
  the reasoning to any other arm.
