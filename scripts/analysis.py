"""Post-hoc analysis harness (chapter_4.tex, Sections "Formulation Study" and
headline baseline comparison).

Discovers Hydra multirun job directories, joins each job's ``experiment_log.csv``
against its resolved ``.hydra/config.yaml`` (dataset, alpha, algorithm, formulation,
seed — none of which are logged into the CSV itself), then:

1. Formulation-study winner selection (``select_winner``): per (dataset, alpha),
   the formulation that reaches the target accuracy using the least mean
   cumulative communication (MB) across its 3 seeds wins; any formulation with a
   seed that never crosses the target-accuracy floor within the fixed round
   budget is disqualified regardless of payload. If the MB margin between the
   top-2 candidates is smaller than their pooled seed-to-seed variability
   (statistical near-tie), the winner is re-selected by higher mean top-1
   accuracy at R=100 instead.

2. Headline baseline comparison (``compare_to_baselines``): pairs the winning
   formulation's 3 seed runs against each baseline algorithm's 3 seed runs by
   seed, reporting the accuracy-at-R100 delta (mean + min/max, no bootstrap/CI)
   and rounds-to-target for both sides.

The target-accuracy floor is 90% of the mean final-round top-1 accuracy of the
uncompressed FedAvg reference, averaged across FedAvg's 3 seeds for that
(dataset, alpha), reusing FedAvg runs already present in the main benchmark grid.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


@dataclass
class RunRecord:
    job_dir: Path
    dataset: str
    alpha: float
    algorithm: str
    formulation: int | None
    seed: int
    csv_path: Path
    # Refinement-flag state, needed to identify the unrefined reference cell that
    # defines the exploration noise margin (§4.3.1).
    refinements: tuple[bool, bool, bool] = (False, False, False)
    # Which conf/algorithm/*.yaml the run composed. NOT the same thing as
    # ``algorithm`` above: every §4.3.7 FedMAQ ablation arm sets ``name: fedmaq``,
    # so ``algorithm`` collapses six of the seven net-new arms onto one value and
    # cannot identify an arm. See :func:`algorithm_config_name`.
    algorithm_config: str = ""
    # Which matrix produced the run, read off the canonical output path. This is
    # the only field that separates Ablation Configuration 7 from FedMAQ's
    # primary-grid rows, since the two differ in nothing else the analysis reads.
    experiment_group: str | None = None
    # ``explore`` or ``formal``. §4.3.1 separates the exploration phase from the
    # confirmatory grid; this is what tells them apart without enumerating the
    # exploration matrices by name.
    phase: str | None = None
    # §4.3's post-processing pipeline. A per-matrix override, so it is a property
    # of the run and not of the algorithm config; the ablation table reports it as
    # a regime note.
    post_process: bool = False

    def __post_init__(self) -> None:
        # A record built without an explicit config name (a hand-constructed
        # fixture, or a run predating .hydra/hydra.yaml) falls back to the
        # algorithm name, which is correct for every config whose file name and
        # ``name:`` field agree -- i.e. everything except the ablation arms.
        if not self.algorithm_config:
            self.algorithm_config = self.algorithm


def algorithm_config_name(job_dir: Path, fallback: str) -> str:
    """The ``conf/algorithm/*.yaml`` name the run composed.

    ``algorithm.name`` cannot serve here. ``fedmaq_no_resource``,
    ``fedmaq_no_data``, ``fedmaq_no_state``, ``fedmaq_no_kd`` and
    ``fedmaq_no_refinements`` all declare ``name: fedmaq`` (deliberately -- they
    dispatch the same hook), so keying on it makes the §4.3.7 arms
    indistinguishable from full FedMAQ and from each other. Hydra records the
    chosen config file under ``hydra.runtime.choices`` in ``.hydra/hydra.yaml``,
    which is written for every run in both layouts.
    """
    hydra_path = job_dir / ".hydra" / "hydra.yaml"
    if not hydra_path.exists():
        return fallback
    try:
        choice = OmegaConf.select(OmegaConf.load(hydra_path), "hydra.runtime.choices.algorithm")
    except Exception:
        return fallback
    return str(choice) if choice is not None else fallback


def phase_and_group_of(job_dir: Path, experiments_root: Path) -> tuple[str | None, str | None]:
    """The phase and matrix-group segments of a canonical output path.

    ``scripts/common.get_canonical_output_dir`` lays runs out as
    ``outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<het>/seed_<n>``.
    Runs from outside a matrix (a bare ``scripts/run.py`` invocation) carry
    neither and return ``(None, None)``.
    """
    try:
        parts = job_dir.resolve().relative_to(experiments_root.resolve()).parts
    except ValueError:
        return None, None
    if len(parts) == 7 and parts[0] == "outputs":
        return parts[1], parts[3]
    return None, None


def discover_runs(experiments_root: Path) -> list[RunRecord]:
    """Join every run's telemetry CSV against its resolved Hydra config.

    Two layouts must both be found. ``--multirun`` sweeps (the confirmatory grid)
    land under ``multirun/<date>/<time>/<job_idx>/``; ``scripts/run_matrix.py``
    (the exploration phase, §4.3.1) sets ``hydra.run.dir`` explicitly and lands
    under ``outputs/<phase>/.../seed_N/`` at an unrelated depth. Globbing only
    the former silently hides every exploration run from
    :func:`exploration_noise_margin`, which is the analysis that decides the
    surviving refinement set.
    """
    runs: list[RunRecord] = []
    config_paths = sorted(
        {
            *experiments_root.glob("multirun/*/*/*/.hydra/config.yaml"),
            *experiments_root.glob("outputs/**/.hydra/config.yaml"),
        }
    )
    for config_path in config_paths:
        job_dir = config_path.parent.parent
        csv_path = job_dir / "experiment_log.csv"
        if not csv_path.exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        algorithm = cfg["algorithm"]["name"]
        phase, group = phase_and_group_of(job_dir, experiments_root)
        runs.append(
            RunRecord(
                job_dir=job_dir,
                dataset=cfg["dataset"]["name"],
                alpha=float(cfg["heterogeneity"]["alpha"]),
                algorithm=algorithm,
                formulation=cfg["algorithm"].get("formulation"),
                seed=int(cfg["seed"]),
                csv_path=csv_path,
                refinements=(
                    bool(cfg["algorithm"].get("soft_voting", False)),
                    bool(cfg["algorithm"].get("ema_student", False)),
                    bool(cfg["algorithm"].get("grad_norm_ema", False)),
                ),
                algorithm_config=algorithm_config_name(job_dir, algorithm),
                experiment_group=group,
                phase=phase,
                post_process=bool(cfg["algorithm"].get("post_process", False)),
            )
        )
    return runs


# The §4.3.7 ablation runs share a dataset, both skews, all three seeds, and
# ``algorithm.name`` with the runs the formulation study and the headline
# baseline comparison read. They must therefore be excluded from both by group,
# not by algorithm: Configuration 7 is ``fedmaq`` on the winning formulation and
# Configuration 6 is ``fedavg_kd``, so an algorithm-level filter admits both.
ABLATION_GROUP = "ablation"

# The two groups the formulation study reads across. Both are required, and they
# are not interchangeable: §4.3.6 evaluates the five candidate formulations
# pipeline-free (``formulation_study``) against an accuracy floor defined from the
# uncompressed FedAvg rows of the confirmatory grid (``benchmark_grid``).
#
# Scoping the candidate side by group is load-bearing, not defensive. Every
# exploration matrix (``pass2_factorial``, ``pass3_freeze_confirm``, and the
# conditional recheck) dispatches ``algorithm=fedmaq``, and ``fedmaq.yaml``
# carries ``formulation: 3``, so those runs are indistinguishable from
# formulation-study candidates on algorithm, config name and formulation alike.
# So are the grid's own FedMAQ rows -- which additionally run with
# ``post_process=true``. Selecting on algorithm alone therefore (a) manufactures a
# verdict at the held-out alpha = 0.3, where no FedAvg reference exists and the
# floor cannot be computed at all, and (b) pools the grid's pipeline-on runs into
# Formulation 3's cell, crediting the incumbent formulation with the payload
# savings of a pipeline the study exists to withhold. See Decision 71.
FORMULATION_STUDY_GROUP = "formulation_study"
GRID_GROUP = "benchmark_grid"


def confirmatory_runs(runs: list[RunRecord]) -> list[RunRecord]:
    """Runs eligible for the formulation study and the baseline comparison.

    Without this filter, ``select_winner`` counts every Formulation-3 ablation arm
    as a Formulation-3 formulation-study candidate, and ``compare_to_baselines``
    builds its ``{seed: run}`` maps by overwriting -- so an arm silently stands in
    for FedMAQ or for a baseline depending on directory iteration order.
    """
    return [r for r in runs if r.experiment_group != ABLATION_GROUP]


REFINEMENT_NAMES = ("soft_voting", "ema_student", "grad_norm_ema")
EXPLORATION_ALPHA = 0.3
# The ``phase:`` every conf/matrix exploration file declares (pass2_explore,
# pass2_factorial, pass3_freeze_confirm). Confirmatory matrices declare ``formal``.
EXPLORATION_PHASE = "explore"
# The keep-or-drop calls belong to the factorial (§4.3.1), not to the single-seed
# screening sweep that precedes it or the two-arm confirmation that follows. All
# three declare ``phase: explore`` at alpha 0.3 with ``name: fedmaq``, and two of
# them contain a (False, False, False) cell, so phase alone would pool R=50 and
# R=100 runs into one sigma and duplicate seed 0 inside it.
EXPLORATION_GROUP = "pass2_factorial"


def _lower_regularized_gamma(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x), series/continued-fraction form.

    Present so the sigma interval below needs nothing outside the standard
    library. scipy is importable in the current environment but is not declared
    in pyproject.toml, and this module has to keep running at the frozen tag that
    chapter 6 §6.2 offers as the reproducible configuration.
    """
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:  # series expansion converges fast here
        term = 1.0 / a
        total = term
        n = 0
        while n < 1000:
            n += 1
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for the upper tail, then complement (Lentz's method).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0.0 else 1.0 / tiny
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return 1.0 - upper


def _chi2_quantile(p: float, df: int) -> float:
    """Inverse chi-square CDF by bisection on the CDF. Adequate for CI endpoints."""
    lo, hi = 0.0, max(float(df) * 10.0, 10.0)
    while _lower_regularized_gamma(df / 2.0, hi / 2.0) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _lower_regularized_gamma(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sigma_confidence_interval(sigma: float, n: int, level: float = 0.95) -> dict:
    """Chi-square interval for a standard deviation estimated from ``n`` samples.

    The point of reporting this is that §4.3.1 calls the margin "measured rather
    than asserted", and a measurement without its own uncertainty is closer to an
    assertion than that phrasing admits. At n=3 the interval spans roughly a
    factor of twelve; the reference cells are deepened to n=5 to cut that to about
    five. Assumes seed-to-seed accuracy is approximately normal -- the usual
    caveat for this interval, and worth stating in the write-up.
    """
    if n < 2 or sigma <= 0.0:
        return {"level": level, "low": None, "high": None, "n": n}
    df = n - 1
    tail = (1.0 - level) / 2.0
    return {
        "level": level,
        "n": n,
        "low": sigma * math.sqrt(df / _chi2_quantile(1.0 - tail, df)),
        "high": sigma * math.sqrt(df / _chi2_quantile(tail, df)),
    }


def _normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def exploration_noise_margin(
    runs: list[RunRecord],
    alpha: float = EXPLORATION_ALPHA,
    experiment_group: str = EXPLORATION_GROUP,
) -> dict:
    """Measure the exploration phase's noise margin and apply the keep-or-drop rule.

    Implements chapter_4.tex §4.3.1 directly, which requires the margin to be
    "measured rather than asserted":

    1. Characterize the seed-to-seed standard deviation ``sigma`` of the
       *unrefined* FedMAQ configuration (all three refinements off) at the
       held-out exploration skew, across whatever seeds that reference cell was
       run at in *this* stage. The count is not fixed here: §4.4 deepens the
       reference cell relative to the arms it is judged against, so it is read
       from the runs rather than assumed.
    2. Scale the margin above sigma, because a delta between two runs carries the
       variance of both: sd(A - B) = sqrt(2) * sigma for independent runs of
       equal variance. Comparing a delta against bare sigma would be the error
       §4.3.1 explicitly guards against.
    3. Retain a mechanism only when its delta *clears* that margin. A mechanism
       that merely scores highest is not retained -- selecting the best of
       several noisy measurements manufactures winners.

    Only runs at ``alpha``, and only runs from ``experiment_group``, are
    considered. §4.3.1 holds exploration at a held-out skew, so a
    confirmatory-skew run appearing here means the sweep config drifted from the
    manuscript and the verdict would be contaminated; that case is reported
    rather than silently averaged in. The group scope is the same guard applied
    to stages: the three exploration matrices differ in round budget and seed
    count and share the unrefined cell, so pooling them would compute sigma from
    a mixture of horizons with seed 0 counted twice. Runs outside the requested
    group are refused, never merged -- pass ``experiment_group`` explicitly to
    measure a different stage (``pass3_freeze_confirm`` for the R=100 gate).

    Alongside the verdicts the result carries the uncertainty on sigma itself and
    the multiplicity accounting for the seven comparisons, so the write-up can
    quote both rather than the point estimate alone. Neither changes the rule.
    """
    # Scope to the exploration phase before anything else. Every confirmatory
    # FedMAQ run -- benchmark grid, formulation study, six of the seven ablation
    # arms -- also declares ``name: fedmaq`` at alpha 0.1/1.0, so an
    # algorithm-level filter reports all of them as contaminating skews and buries
    # the one case the warning exists to catch: an *exploration* matrix that
    # drifted onto a reported skew.
    runs = [r for r in runs if r.phase == EXPLORATION_PHASE]
    fedmaq_runs = [r for r in runs if r.algorithm == "fedmaq"]
    groups_present = sorted({r.experiment_group or "<none>" for r in fedmaq_runs})
    scoped = [r for r in fedmaq_runs if r.experiment_group == experiment_group]
    if not scoped:
        return {
            "alpha": alpha,
            "experiment_group": experiment_group,
            "error": (
                f"no exploration runs found in experiment_group '{experiment_group}'. "
                f"Groups present: {groups_present}. Stages are never pooled -- pass "
                "experiment_group= explicitly to measure a different one."
            ),
            "groups_present": groups_present,
        }

    candidates = [r for r in scoped if abs(r.alpha - alpha) < 1e-9]
    contaminated = sorted({r.alpha for r in scoped if abs(r.alpha - alpha) >= 1e-9})

    by_cell: dict[tuple[bool, bool, bool], list[float]] = {}
    for r in candidates:
        acc = accuracy_at_round(load_round_metrics(r.csv_path), 10**9)
        by_cell.setdefault(r.refinements, []).append(acc)

    unrefined = by_cell.get((False, False, False), [])
    if len(unrefined) < 3:
        return {
            "alpha": alpha,
            "experiment_group": experiment_group,
            "error": (
                "cannot measure sigma: the unrefined cell (all refinements off) has "
                f"{len(unrefined)} seed(s); §4.3.1 requires at least three, and the "
                "matrix deepens this cell to five."
            ),
            "groups_present": groups_present,
            "other_skews_present": contaminated,
        }

    sigma = statistics.stdev(unrefined)
    margin = sigma * math.sqrt(2.0)
    baseline_mean = statistics.fmean(unrefined)

    n_ref = len(unrefined)
    verdicts = {}
    per_comparison_rates = []
    for cell, accs in sorted(by_cell.items()):
        if cell == (False, False, False):
            continue
        delta = statistics.fmean(accs) - baseline_mean
        # strict: a cell key that is not one flag per REFINEMENT_NAMES entry means
        # the roster changed under the analysis. Truncating silently would label a
        # verdict with the wrong mechanism set, which is worse than crashing here.
        active = [n for n, on in zip(REFINEMENT_NAMES, cell, strict=True) if on]
        # Standard error of the delta between two cell means. The threshold is
        # sqrt(2)*sigma regardless -- that is the pre-registered rule and is not
        # recomputed here -- but expressing it in SE units is what makes the
        # per-comparison false-positive rate quotable.
        se_delta = sigma * math.sqrt(1.0 / n_ref + 1.0 / len(accs)) if sigma > 0 else 0.0
        z = margin / se_delta if se_delta > 0 else float("inf")
        p_null = _normal_sf(z)
        per_comparison_rates.append(p_null)
        verdicts["+".join(active) or "none"] = {
            "active": active,
            "seeds": len(accs),
            "mean_accuracy": statistics.fmean(accs),
            "delta_vs_unrefined": delta,
            "delta_standard_error": se_delta,
            "margin_in_standard_errors": z,
            "false_positive_rate_if_null": p_null,
            "clears_margin": delta > margin,
            "retained": delta > margin,
        }

    # THE SURVIVING SET IS A CELL, NOT A UNION.
    #
    # Taking the union of every mechanism appearing in any clearing cell would
    # ship a combination the factorial never measured: if soft_voting alone and
    # ema_student alone each clear, the union ships both together, and their
    # joint cell may well be one of the seven that did not clear. §4.3.1's
    # keep-or-drop rule selects among measured cells.
    #
    # Among the cells that clear, the smallest wins -- fewest moving parts for
    # the same measured benefit, and it is the choice that survives an examiner
    # asking why a mechanism is in the frozen configuration. Ties on size break
    # toward the larger delta. If nothing clears, the set is empty and FedMAQ
    # freezes unrefined; that is a real outcome, not a failure to select.
    clearing = [(k, v) for k, v in verdicts.items() if v["retained"]]
    clearing.sort(key=lambda kv: (len(kv[1]["active"]), -kv[1]["delta_vs_unrefined"]))
    surviving_cell, surviving = clearing[0] if clearing else (None, None)

    return {
        "alpha": alpha,
        "experiment_group": experiment_group,
        "sigma_unrefined": sigma,
        "sigma_confidence_interval": sigma_confidence_interval(sigma, n_ref),
        "noise_margin": margin,
        "margin_rule": "sqrt(2) * sigma — a delta carries the variance of both runs",
        "unrefined_mean_accuracy": baseline_mean,
        "unrefined_seeds": n_ref,
        "verdicts": verdicts,
        "surviving_cell": surviving_cell,
        "surviving_refinement_set": list(surviving["active"]) if surviving else [],
        "selection_rule": (
            "smallest cell clearing the margin; ties by larger delta; empty if none clears"
        ),
        # Mechanisms the factorial actually measured and did not ship. Scoped to
        # what appeared in some tested cell rather than to REFINEMENT_NAMES: a
        # mechanism absent from the matrix was never judged, and reporting it as
        # "discarded" would claim evidence against it that was never collected.
        "discarded": sorted(
            {m for v in verdicts.values() for m in v["active"]}
            - set(surviving["active"] if surviving else [])
        ),
        # Seven cells are tested against one reference at one threshold. Reported,
        # not corrected: conf/matrix/pass2_factorial.yaml pre-registers that the
        # independent R=100 confirmation is what controls this, and an alpha
        # correction here would cost power the confirmation already covers.
        "multiplicity": {
            "comparisons": len(per_comparison_rates),
            "family_wise_false_positive_rate": (
                1.0 - math.prod(1.0 - p for p in per_comparison_rates)
                if per_comparison_rates
                else 0.0
            ),
            "note": (
                "Rate that at least one of the cells clears the margin by chance "
                "under the null. Controlled by the pass3_freeze_confirm stage, "
                "which re-tests the selected cell on fresh runs at R=100, not by "
                "correcting the threshold here."
            ),
        },
        "groups_present": groups_present,
        "other_skews_present": contaminated,
    }


def load_round_metrics(csv_path: Path) -> pd.DataFrame:
    """Load a single job's per-round telemetry CSV."""
    return pd.read_csv(csv_path)


def compute_target_floor(runs: list[RunRecord], dataset: str, alpha: float) -> float:
    """90% of the mean final-round (R=100) top-1 accuracy of the uncompressed FedAvg
    reference, averaged across FedAvg's 3 seeds for this (dataset, alpha).

    The reference is the confirmatory grid's own FedAvg rows -- chapter_4.tex:312
    defines the floor as "reusing the FedAvg runs already present in the benchmark
    grid", and pinning the group here is what makes that sentence true of the code
    rather than merely of the intent. Those six CIFAR-10 rows are dispatched ahead
    of the rest of the grid (docs/RUNBOOK.md Stage 1c) precisely so this function
    has something to read when the formulation study needs it.
    """
    fedavg_runs = [
        r
        for r in runs
        if r.dataset == dataset
        and r.alpha == alpha
        and r.algorithm == "fedavg"
        and r.experiment_group == GRID_GROUP
    ]
    if not fedavg_runs:
        raise ValueError(
            f"No FedAvg reference runs found in experiment_group={GRID_GROUP!r} for "
            f"dataset={dataset}, alpha={alpha}. The accuracy floor is defined against "
            f"the grid's uncompressed control (chapter_4.tex:312). If the formulation "
            f"study has run and this fails, Stage 1c was skipped: dispatch "
            f"`run_matrix.py --matrix benchmark_grid --only fedavg` first."
        )
    final_accs = [load_round_metrics(r.csv_path)["test/accuracy"].iloc[-1] for r in fedavg_runs]
    return 0.9 * (sum(final_accs) / len(final_accs))


def accuracy_at_round(run_df: pd.DataFrame, round_num: int) -> float:
    """test/accuracy at the given round; falls back to the last logged round
    if that exact round number isn't present (e.g. a run stopped early)."""
    exact = run_df[run_df["round"] == round_num]
    if not exact.empty:
        return float(exact.iloc[0]["test/accuracy"])
    return float(run_df.iloc[-1]["test/accuracy"])


def first_crossing(run_df: pd.DataFrame, floor: float) -> tuple[int | None, float | None]:
    """First round at which test/accuracy >= floor, and cumulative MB at that round."""
    crossing = run_df[run_df["test/accuracy"] >= floor]
    if crossing.empty:
        return None, None
    row = crossing.iloc[0]
    return int(row["round"]), float(row["communication/cumulative_mb"])


def select_winner(runs: list[RunRecord]) -> dict:
    """Apply the pre-registered winner rule independently per (dataset, alpha).

    For each formulation (0-4), a formulation is disqualified if ANY of its 3 seeds
    never crosses the target-accuracy floor. Among qualified formulations, the winner
    minimizes mean cumulative-MB-to-target across seeds.

    Per-seed, not per-mean: one failing seed out of three disqualifies the
    formulation. §4.3.6 states this explicitly as of 2026-08-01 -- it is the
    stricter reading and the one that has always run, and it refuses to average
    away a seed that never converged. At alpha = 0.1, where seed spread is
    widest, that distinction is where a wipeout would come from.

    This returns one verdict *per skew* and is therefore not the freeze. The
    formulation study runs CIFAR-10 at both skews, so two verdicts are structural;
    :func:`resolve_frozen_formulation` collapses them into the single formulation
    written into ``conf/algorithm/fedmaq.yaml``, including the branches for a
    split verdict and for a field in which nothing qualified.
    """
    result: dict = {}
    runs = confirmatory_runs(runs)
    # Three filters, each removing a different impostor:
    #   experiment_group -- the exploration factorial, the R=100 confirmation and
    #     the grid all dispatch ``algorithm=fedmaq`` at ``formulation: 3``;
    #   algorithm_config -- the §4.3.7 ablation arms declare ``name: fedmaq``;
    #   formulation      -- a run that never resolved one is not a candidate.
    # See FORMULATION_STUDY_GROUP above for what each admission would cost.
    fedmaq_runs = [
        r
        for r in runs
        if r.experiment_group == FORMULATION_STUDY_GROUP
        and r.algorithm_config == "fedmaq"
        and r.formulation is not None
    ]
    datasets_alphas = sorted({(r.dataset, r.alpha) for r in fedmaq_runs})

    for dataset, alpha in datasets_alphas:
        floor = compute_target_floor(runs, dataset, alpha)
        formulations = sorted(
            {r.formulation for r in fedmaq_runs if r.dataset == dataset and r.alpha == alpha}
        )

        detail: dict[int, dict] = {}
        for formulation in formulations:
            seed_runs = [
                r
                for r in fedmaq_runs
                if r.dataset == dataset and r.alpha == alpha and r.formulation == formulation
            ]
            seed_results: dict[int, dict] = {}
            disqualified = False
            crossing_mbs = []
            r100_accs = []
            for r in seed_runs:
                run_df = load_round_metrics(r.csv_path)
                round_num, cumulative_mb = first_crossing(run_df, floor)
                seed_results[r.seed] = {
                    "round": round_num,
                    "cumulative_mb": cumulative_mb,
                }
                r100_accs.append(accuracy_at_round(run_df, 100))
                if round_num is None:
                    disqualified = True
                else:
                    crossing_mbs.append(cumulative_mb)
            mean_mb = None
            if crossing_mbs and not disqualified:
                mean_mb = sum(crossing_mbs) / len(crossing_mbs)
            detail[formulation] = {
                "seeds": seed_results,
                "disqualified": disqualified,
                "mean_cumulative_mb": mean_mb,
                "crossing_mbs": crossing_mbs,
                "mean_accuracy_r100": sum(r100_accs) / len(r100_accs),
            }

        qualified = {f: d for f, d in detail.items() if not d["disqualified"]}
        if qualified:
            by_mb = sorted(qualified, key=lambda f: qualified[f]["mean_cumulative_mb"])
            winner = by_mb[0]
            margin = None
            if len(by_mb) > 1:
                top1, top2 = by_mb[0], by_mb[1]
                margin = (
                    qualified[top2]["mean_cumulative_mb"] - qualified[top1]["mean_cumulative_mb"]
                )
                # Statistical tie-break (chapter_4.tex tie-break rule): if the MB
                # margin between the top-2 candidates is smaller than the larger
                # of their two seed-to-seed standard deviations, it's noise, not
                # a real gap -- re-select by higher mean top-1 accuracy at R=100.
                #
                # WITHIN-CANDIDATE, NEVER POOLED. This concatenated both
                # candidates' crossing-MB values and took the stdev of the
                # combined six until 2026-08-01, which folded the between-
                # candidate separation into the threshold: with within-candidate
                # sd s and separation d, the combined sample variance is
                # (4s^2 + 1.5d^2)/5, so the rule fired at d < 1.069s and the
                # threshold grew with the very margin it was judging. A
                # pre-registered rule cannot be self-referential, and the
                # published rule says "either candidate's seed-to-seed
                # variability", not their pooled spread. See Decision 66.
                stdevs = [
                    statistics.stdev(qualified[f]["crossing_mbs"])
                    if len(qualified[f]["crossing_mbs"]) > 1
                    else 0.0
                    for f in (top1, top2)
                ]
                tie_threshold = max(stdevs)
                if margin < tie_threshold:
                    winner = max((top1, top2), key=lambda f: qualified[f]["mean_accuracy_r100"])
        else:
            winner = None
            margin = None

        key = f"{dataset}_alpha_{alpha}"
        result[key] = {
            "dataset": dataset,
            "alpha": alpha,
            "target_accuracy_floor": floor,
            "formulations": detail,
            "winner": winner,
            "margin_mb": margin,
        }

    return result


def resolve_frozen_formulation(winner_result: dict, dataset: str = "cifar10") -> dict:
    """Collapse the per-skew verdicts of :func:`select_winner` into the one
    formulation that is frozen into ``conf/algorithm/fedmaq.yaml``.

    ``select_winner`` is deliberately per-``(dataset, alpha)``: the formulation
    study runs CIFAR-10 at both skews, so it structurally produces two verdicts.
    The freeze takes one scalar, the 105-run grid runs one FedMAQ, and every
    §4.3.7 arm inherits one formulation -- and three downstream branches are
    conditioned on there being a single answer (Configuration 4's expressibility,
    the reserved recheck, and the freeze itself). This function is the
    pre-registered rule that produces it. See Decisions 64-65 and §4.3.6.

    The rule, in order:

    1. **Agreement.** If the same formulation wins at both skews, it is frozen.
    2. **Divergence.** If the skews disagree, the alpha = 0.1 winner is frozen
       and the split is reported as a headline finding -- §4.3.6 already promises
       that a formulation shifting with skew is itself a result. Severe skew
       breaks the tie because it is the regime this thesis's claims are staked
       on, so selecting where the problem is hardest is the conservative
       direction rather than the flattering one.
    3. **One-sided wipeout.** If every formulation is disqualified at one skew
       but not the other, the surviving skew's winner is frozen outright. Rule 2
       does not apply: there is only one valid verdict to compare against.
    4. **Total wipeout.** If every formulation is disqualified at both skews, the
       accuracy-floor guard has caught the entire field. The winner is taken by
       highest mean top-1 at R=100 at alpha = 0.1 -- the criterion already
       pre-registered as the near-tie break, so no new parameter enters -- and
       ``contribution_withdrawn`` is set. §4.3.6's framing of formulation
       selection as the thesis's primary methodological contribution does not
       survive a field in which nothing reached the FedAvg-relative target, and
       §5.7 records that rather than letting a rescued winner paper over it.

    Rule 4 deliberately does *not* mirror Decision 60's empty-freeze branch,
    which withdraws a claim instead of manufacturing a winner. The disanalogy is
    that "ship unrefined" is a coherent configuration and "ship no formulation"
    is not: ``fedmaq.yaml`` gets a number either way, so the only real choice is
    whether it is picked empirically or left at whatever the implementation
    already had. Freezing the incumbent by default would mean the thesis's
    self-described primary contribution was settled by a default value.
    """
    entries = {e["alpha"]: e for e in winner_result.values() if e["dataset"] == dataset}
    if not entries:
        raise ValueError(
            f"No formulation-study verdicts for dataset {dataset!r}. "
            f"Available: {sorted({e['dataset'] for e in winner_result.values()})}"
        )

    severe, moderate = 0.1, 1.0
    missing = {severe, moderate} - set(entries)
    if missing:
        raise ValueError(
            f"resolve_frozen_formulation needs both skews for {dataset!r}; "
            f"missing alpha {sorted(missing)}. The formulation study runs "
            f"alpha in {{0.1, 1.0}} by design (conf/matrix/formulation_study.yaml) "
            f"and the freeze rule is undefined on a partial sweep."
        )

    severe_entry, moderate_entry = entries[severe], entries[moderate]
    severe_winner = severe_entry["winner"]
    moderate_winner = moderate_entry["winner"]

    def _verdict(rule: str, formulation, **extra) -> dict:
        out = {
            "dataset": dataset,
            "frozen_formulation": formulation,
            "rule": rule,
            "alpha_0.1_winner": severe_winner,
            "alpha_1.0_winner": moderate_winner,
            "skews_agree": severe_winner == moderate_winner,
            "contribution_withdrawn": False,
            "recheck_required": None,
        }
        out.update(extra)
        # conf/matrix/formulation_study.yaml reserves a 6-run recheck of the
        # frozen refinement layer whenever the winner is not Formulation 3, the
        # formulation the layer was selected under. It is a veto on that layer,
        # never a second search.
        if out["frozen_formulation"] is not None:
            out["recheck_required"] = out["frozen_formulation"] != 3
        return out

    if severe_winner is None and moderate_winner is None:
        # Rule 4. Fall back to accuracy at the severe skew, and withdraw the
        # contribution claim that a qualified field would have supported.
        detail = severe_entry["formulations"]
        fallback = max(detail, key=lambda f: detail[f]["mean_accuracy_r100"])
        return _verdict(
            "total_disqualification_accuracy_fallback",
            fallback,
            contribution_withdrawn=True,
            fallback_mean_accuracy_r100=detail[fallback]["mean_accuracy_r100"],
            target_accuracy_floor=severe_entry["target_accuracy_floor"],
        )

    if severe_winner is None or moderate_winner is None:
        # Rule 3. Exactly one skew produced a qualified field.
        surviving_alpha = moderate if severe_winner is None else severe
        surviving = moderate_winner if severe_winner is None else severe_winner
        return _verdict(
            "one_sided_disqualification",
            surviving,
            surviving_alpha=surviving_alpha,
        )

    if severe_winner == moderate_winner:
        return _verdict("agreement", severe_winner)  # Rule 1.

    return _verdict("divergence_severe_skew_breaks", severe_winner)  # Rule 2.


def compare_to_baselines(runs: list[RunRecord], winner_result: dict) -> dict:
    """Headline baseline comparison (chapter_4.tex Section 4, statistical
    procedure + convergence-stability metric).

    For each (dataset, alpha) in the confirmatory grid, pairs FedMAQ's 3 seed runs
    against each baseline algorithm's 3 seed runs by seed number. Reports the
    paired per-seed accuracy-at-R100 delta (FedMAQ - baseline: mean + min/max, no
    bootstrap/CI) and rounds-to-target for both sides against the same
    target-accuracy floor the winner rule uses.

    **Both sides are read from the grid group, never from the formulation study.**
    The study runs FedMAQ on CIFAR-10 at both grid skews with the same three
    seeds, differing only in that it withholds the §4.3 post-processing pipeline.
    Selecting FedMAQ's rows by ``(dataset, alpha, formulation)`` alone therefore
    builds ``{seed: run}`` from twelve candidates for six slots and resolves the
    collision by iteration order -- so the headline table could report
    pipeline-free FedMAQ against pipeline-era baselines, understating FedMAQ's
    own communication figures and breaking chapter_4.tex:312's rule that a FedMAQ
    run carries the pipeline if and only if what it is compared against does.
    ``winner_result`` is consulted only to check that the grid actually ran the
    frozen formulation. See Decision 71.
    """
    result: dict = {}
    runs = [r for r in confirmatory_runs(runs) if r.experiment_group == GRID_GROUP]
    frozen = {
        (e["dataset"], e["alpha"]): e["winner"]
        for e in winner_result.values()
        if e["winner"] is not None
    }
    grid_targets = sorted({(r.dataset, r.alpha) for r in runs if r.algorithm_config == "fedmaq"})
    for dataset, alpha in grid_targets:
        floor = compute_target_floor(runs, dataset, alpha)
        fedmaq_by_seed = {
            r.seed: r
            for r in runs
            if r.dataset == dataset and r.alpha == alpha and r.algorithm_config == "fedmaq"
        }
        formulations = {r.formulation for r in fedmaq_by_seed.values()}
        formulation = next(iter(formulations)) if len(formulations) == 1 else sorted(formulations)
        # The grid runs one frozen FedMAQ. More than one formulation here means
        # the freeze was edited mid-grid, which §4.3.1's tag forbids outright.
        formulation_disagreement = len(formulations) > 1
        expected = frozen.get((dataset, alpha))
        baseline_algorithms = sorted(
            {
                r.algorithm
                for r in runs
                if r.dataset == dataset and r.alpha == alpha and r.algorithm != "fedmaq"
            }
        )
        for baseline_algo in baseline_algorithms:
            baseline_by_seed = {
                r.seed: r
                for r in runs
                if r.dataset == dataset and r.alpha == alpha and r.algorithm == baseline_algo
            }
            common_seeds = sorted(set(fedmaq_by_seed) & set(baseline_by_seed))
            if not common_seeds:
                continue

            per_seed: dict[int, dict] = {}
            deltas = []
            for seed in common_seeds:
                fedmaq_df = load_round_metrics(fedmaq_by_seed[seed].csv_path)
                baseline_df = load_round_metrics(baseline_by_seed[seed].csv_path)
                fedmaq_acc = accuracy_at_round(fedmaq_df, 100)
                baseline_acc = accuracy_at_round(baseline_df, 100)
                delta = fedmaq_acc - baseline_acc
                deltas.append(delta)
                per_seed[seed] = {
                    "fedmaq_accuracy": fedmaq_acc,
                    "baseline_accuracy": baseline_acc,
                    "delta": delta,
                    "fedmaq_rounds_to_target": first_crossing(fedmaq_df, floor)[0],
                    "baseline_rounds_to_target": first_crossing(baseline_df, floor)[0],
                }

            result[f"{dataset}_alpha_{alpha}_vs_{baseline_algo}"] = {
                "dataset": dataset,
                "alpha": alpha,
                "fedmaq_formulation": formulation,
                # None where the formulation study produced no verdict for this
                # (dataset, alpha) -- it runs CIFAR-10 only, so CIFAR-100 and
                # FEMNIST inherit the freeze without a local verdict to check.
                "frozen_formulation": expected,
                "formulation_matches_freeze": (expected is None or formulation == expected)
                and not formulation_disagreement,
                "baseline": baseline_algo,
                "per_seed": per_seed,
                "mean_delta": sum(deltas) / len(deltas),
                "min_delta": min(deltas),
                "max_delta": max(deltas),
            }

    return result


# Manuscript §4.3.7's eight configurations, mapped to the algorithm config each
# one dispatches. Configuration 1 is the only inherited arm: it is read from the
# primary benchmark grid rather than re-run, because it has no compression stage
# for the post-processing pipeline to attach to and its telemetry is therefore
# identical in either regime.
ABLATION_CONFIGURATIONS: dict[int, tuple[str, str]] = {
    1: ("fedavg", "Uncompressed FedAvg (control, inherited from the primary grid)"),
    2: ("fedmaq_no_resource", "FedMAQ without resource awareness"),
    3: ("fedmaq_no_data", "FedMAQ without data awareness (DynFed-style reference point)"),
    4: ("fedmaq_no_state", "FedMAQ without state awareness (pre-registered fallback arm)"),
    5: ("fedmaq_no_kd", "FedMAQ without knowledge distillation"),
    6: ("fedavg_kd", "FedMAQ without quantization"),
    7: ("fedmaq", "Full FedMAQ (the study's parity anchor)"),
    8: ("fedmaq_no_refinements", "Full FedMAQ without the frozen refinement layer"),
}

INHERITED_CONFIGURATIONS = {1}

# §4.3.7 records a mechanism as inapplicable, rather than disabled, where an arm
# leaves it with no signal to act on. These are the only permitted deviations
# from the shared refinement layer; anything else is a parity violation.
REFINEMENT_EXCEPTIONS: dict[int, set[str]] = {
    5: {"soft_voting"},  # nothing is distilled, so there are no teacher logits to weight
    6: {"soft_voting", "grad_norm_ema"},  # nothing is quantized
    8: set(),  # removing the layer *is* this arm's removal; handled separately
}


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def build_ablation_table(runs: list[RunRecord], dataset: str = "cifar10") -> dict:
    """Assemble the §4.3.7 / §5.4 ablation matrix from run telemetry.

    Emits, per configuration and skew, the final-round top-1 accuracy and
    cumulative communication as mean and seed-to-seed SD, alongside the two
    design facts the manuscript's table note must carry: each arm's formulation
    (so Configuration 4's fallback to Formulation 1 is visible in the table rather
    than only in the prose) and the post-processing regime.

    ``parity`` is the check §5.4 requires *before* any delta is attributed to an
    awareness signal: that every arm carried the identical frozen refinement layer
    and the identical pipeline regime. A non-empty ``violations`` list means the
    contrasts are not attributable and the table must not be reported.
    """
    by_config: dict[int, dict] = {}
    violations: list[str] = []

    for config_num, (alg_config, description) in ABLATION_CONFIGURATIONS.items():
        expected_group = None if config_num in INHERITED_CONFIGURATIONS else ABLATION_GROUP
        matched = [
            r
            for r in runs
            if r.dataset == dataset
            and r.algorithm_config == alg_config
            and (
                r.experiment_group == ABLATION_GROUP
                if expected_group == ABLATION_GROUP
                else r.experiment_group != ABLATION_GROUP
            )
        ]

        cells: dict[str, dict] = {}
        for alpha in sorted({r.alpha for r in matched}):
            seed_runs = sorted((r for r in matched if r.alpha == alpha), key=lambda r: r.seed)
            accs, mbs = [], []
            for r in seed_runs:
                df = load_round_metrics(r.csv_path)
                accs.append(accuracy_at_round(df, 100))
                mbs.append(float(df["communication/cumulative_mb"].iloc[-1]))
            cells[f"alpha_{alpha}"] = {
                "seeds": [r.seed for r in seed_runs],
                "accuracy_r100": _mean_sd(accs),
                "cumulative_mb": _mean_sd(mbs),
            }
            if len(seed_runs) != 3:
                violations.append(
                    f"Configuration {config_num} ({alg_config}) has {len(seed_runs)} "
                    f"seed(s) at alpha={alpha}; §4.3.7 specifies three."
                )

        by_config[config_num] = {
            "algorithm_config": alg_config,
            "description": description,
            "inherited": config_num in INHERITED_CONFIGURATIONS,
            "formulation": next((r.formulation for r in matched), None),
            "post_process": next((r.post_process for r in matched), None),
            "refinements": next((r.refinements for r in matched), None),
            "cells": cells,
        }
        if not matched:
            violations.append(
                f"Configuration {config_num} ({alg_config}) has no runs; "
                f"expected them in the "
                f"{'primary grid' if expected_group is None else ABLATION_GROUP} group."
            )

    # Regime: §4.3.7 withholds the pipeline from every arm, Configuration 1
    # excepted because it is inherited and has no compression stage at all.
    for config_num, entry in by_config.items():
        if config_num in INHERITED_CONFIGURATIONS or entry["post_process"] is None:
            continue
        if entry["post_process"]:
            violations.append(
                f"Configuration {config_num} ran with the §4.3 post-processing "
                f"pipeline on. §4.3.7 withholds it from every arm, or each arm is "
                f"two removals from its reference rather than one."
            )

    # Refinement-layer parity, against Configuration 7 as the anchor.
    anchor = by_config.get(7, {}).get("refinements")
    if anchor is not None:
        for config_num, entry in by_config.items():
            if config_num in INHERITED_CONFIGURATIONS or config_num in (7, 8):
                continue
            arm = entry["refinements"]
            if arm is None:
                continue
            deviating = {
                name for name, a, b in zip(REFINEMENT_NAMES, anchor, arm, strict=True) if a != b
            }
            unexplained = deviating - REFINEMENT_EXCEPTIONS.get(config_num, set())
            if unexplained:
                violations.append(
                    f"Configuration {config_num} deviates from the frozen refinement "
                    f"layer on {sorted(unexplained)} without §4.3.7 recording those "
                    f"mechanisms as inapplicable to it."
                )

    # Configuration 4 is anchored to the formulation study's own runs on its
    # formulation, not to Configuration 7 (§4.3.7's pre-registered fallback rule).
    # Reading its delta against Configuration 7 would price the formulation change
    # rather than the removed signal, so the anchor is resolved here rather than
    # left to whoever writes §5.4.
    config4 = by_config.get(4, {})
    if config4.get("formulation") is not None and config4["formulation"] != by_config.get(
        7, {}
    ).get("formulation"):
        anchor_cells: dict[str, dict] = {}
        for alpha_key in config4["cells"]:
            alpha = float(alpha_key.removeprefix("alpha_"))
            anchor_runs = sorted(
                (
                    r
                    for r in runs
                    if r.dataset == dataset
                    and r.alpha == alpha
                    and r.algorithm_config == "fedmaq"
                    # The study group, not merely "not the ablation": the grid
                    # also runs ``fedmaq`` at this dataset and skew, and the
                    # anchor's whole purpose is to sit in the arm's own regime.
                    # ``post_process`` is checked as well because the two facts
                    # are separable in principle and a silent regime mismatch
                    # here would price the pipeline as if it were state awareness.
                    and r.experiment_group == FORMULATION_STUDY_GROUP
                    and r.formulation == config4["formulation"]
                    and not r.post_process
                ),
                key=lambda r: r.seed,
            )
            if not anchor_runs:
                violations.append(
                    f"Configuration 4 runs on Formulation {config4['formulation']} but "
                    f"the formulation study's own Formulation {config4['formulation']} "
                    f"runs at alpha={alpha} are not discoverable; §4.3.7's fallback "
                    f"rule leaves the arm without a parity anchor."
                )
                continue
            accs = [accuracy_at_round(load_round_metrics(r.csv_path), 100) for r in anchor_runs]
            anchor_cells[alpha_key] = {
                "seeds": [r.seed for r in anchor_runs],
                "accuracy_r100": _mean_sd(accs),
            }
        config4["parity_anchor"] = {
            "source": "formulation study",
            "formulation": config4["formulation"],
            "cells": anchor_cells,
            "note": (
                "compared against this, not against Configuration 7 -- the winning "
                "formulation cannot express the removal of state awareness"
            ),
        }

    return {
        "dataset": dataset,
        "configurations": by_config,
        "parity": {
            "refinement_anchor": anchor,
            "pipeline_regime": (
                "withheld from every arm; inapplicable to Configuration 6, which "
                "produces no quantized codes for it to act on"
            ),
            "violations": violations,
            "attributable": not violations,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formulation-study winner selection + headline baseline comparison"
    )
    parser.add_argument("--experiments-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/analysis_output/formulation_winner.json"),
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=Path("scripts/analysis_output/baseline_comparison.json"),
    )
    parser.add_argument(
        "--exploration-output",
        type=Path,
        default=Path("scripts/analysis_output/exploration_margin.json"),
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=Path("scripts/analysis_output/ablation_table.json"),
    )
    args = parser.parse_args()

    runs = discover_runs(args.experiments_root)
    result = select_winner(runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote formulation-winner verdict to {args.output}")

    baseline_result = compare_to_baselines(runs, result)
    args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.baseline_output, "w", encoding="utf-8") as f:
        json.dump(baseline_result, f, indent=2)
    print(f"Wrote baseline-comparison report to {args.baseline_output}")
    drifted = sorted(
        {
            (e["dataset"], e["alpha"], str(e["fedmaq_formulation"]), str(e["frozen_formulation"]))
            for e in baseline_result.values()
            if not e["formulation_matches_freeze"]
        }
    )
    for dataset, alpha, ran, frozen_formulation in drifted:
        print(
            f"  FREEZE DRIFT: {dataset} alpha={alpha} ran formulation {ran}, "
            f"frozen verdict was {frozen_formulation}. §4.3.1 forbids editing a frozen "
            "config downstream of the tag; the grid rows must be re-dispatched, not reported."
        )

    # Exploration-phase margin and keep-or-drop verdicts (§4.3.1). Chapter 5 §5.1
    # reports sigma, the derived margin, and the surviving set from this file.
    exploration = exploration_noise_margin(runs)
    args.exploration_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.exploration_output, "w", encoding="utf-8") as f:
        json.dump(exploration, f, indent=2)
    print(f"Wrote exploration noise-margin verdict to {args.exploration_output}")
    if exploration.get("other_skews_present"):
        print(
            "  WARNING: FedMAQ runs found at non-exploration skews "
            f"{exploration['other_skews_present']}. §4.3.1 holds exploration at "
            f"alpha={EXPLORATION_ALPHA}; check conf/matrix/pass2_explore.yaml."
        )

    # §4.3.7 / §5.4 ablation matrix.
    ablation = build_ablation_table(runs)
    args.ablation_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.ablation_output, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)
    print(f"Wrote ablation matrix to {args.ablation_output}")
    for violation in ablation["parity"]["violations"]:
        print(f"  PARITY VIOLATION: {violation}")
    if ablation["parity"]["violations"]:
        print(
            "  The §5.4 contrasts are NOT attributable while any violation stands; "
            "resolve them before reporting the table."
        )


if __name__ == "__main__":
    main()
