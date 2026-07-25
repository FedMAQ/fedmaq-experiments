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


def experiment_group_of(job_dir: Path, experiments_root: Path) -> str | None:
    """The matrix group segment of a canonical output path, or ``None``.

    ``scripts/common.get_canonical_output_dir`` lays runs out as
    ``outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<het>/seed_<n>``.
    Raw ``--multirun`` jobs (the formulation study) carry no group and return
    ``None``.
    """
    try:
        parts = job_dir.resolve().relative_to(experiments_root.resolve()).parts
    except ValueError:
        return None
    if len(parts) == 7 and parts[0] == "outputs":
        return parts[3]
    return None


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
                experiment_group=experiment_group_of(job_dir, experiments_root),
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


def exploration_noise_margin(runs: list[RunRecord], alpha: float = EXPLORATION_ALPHA) -> dict:
    """Measure the exploration phase's noise margin and apply the keep-or-drop rule.

    Implements chapter_4.tex §4.3.1 directly, which requires the margin to be
    "measured rather than asserted":

    1. Characterize the seed-to-seed standard deviation ``sigma`` of the
       *unrefined* FedMAQ configuration (all three refinements off) at the
       held-out exploration skew, across its three seeds.
    2. Scale the margin above sigma, because a delta between two runs carries the
       variance of both: sd(A - B) = sqrt(2) * sigma for independent runs of
       equal variance. Comparing a delta against bare sigma would be the error
       §4.3.1 explicitly guards against.
    3. Retain a mechanism only when its delta *clears* that margin. A mechanism
       that merely scores highest is not retained -- selecting the best of
       several noisy measurements manufactures winners.

    Only runs at ``alpha`` are considered. §4.3.1 holds exploration at a held-out
    skew, so a confirmatory-skew run appearing here means the sweep config
    drifted from the manuscript and the verdict would be contaminated; that case
    is reported rather than silently averaged in.
    """
    # The ablation runs at alpha 0.1/1.0 also declare ``name: fedmaq``; without the
    # group filter every one of them is reported as a contaminating skew.
    runs = confirmatory_runs(runs)
    candidates = [r for r in runs if r.algorithm == "fedmaq" and abs(r.alpha - alpha) < 1e-9]
    contaminated = sorted(
        {r.alpha for r in runs if r.algorithm == "fedmaq" and abs(r.alpha - alpha) >= 1e-9}
    )

    by_cell: dict[tuple[bool, bool, bool], list[float]] = {}
    for r in candidates:
        acc = accuracy_at_round(load_round_metrics(r.csv_path), 10**9)
        by_cell.setdefault(r.refinements, []).append(acc)

    unrefined = by_cell.get((False, False, False), [])
    if len(unrefined) < 2:
        return {
            "alpha": alpha,
            "error": (
                "cannot measure sigma: the unrefined cell (all refinements off) has "
                f"{len(unrefined)} seed(s); §4.3.1 requires three."
            ),
            "other_skews_present": contaminated,
        }

    sigma = statistics.stdev(unrefined)
    margin = sigma * math.sqrt(2.0)
    baseline_mean = statistics.fmean(unrefined)

    verdicts = {}
    for cell, accs in sorted(by_cell.items()):
        if cell == (False, False, False):
            continue
        delta = statistics.fmean(accs) - baseline_mean
        active = [n for n, on in zip(REFINEMENT_NAMES, cell) if on]
        verdicts["+".join(active) or "none"] = {
            "active": active,
            "seeds": len(accs),
            "mean_accuracy": statistics.fmean(accs),
            "delta_vs_unrefined": delta,
            "clears_margin": delta > margin,
            "retained": delta > margin,
        }

    return {
        "alpha": alpha,
        "sigma_unrefined": sigma,
        "noise_margin": margin,
        "margin_rule": "sqrt(2) * sigma — a delta carries the variance of both runs",
        "unrefined_mean_accuracy": baseline_mean,
        "unrefined_seeds": len(unrefined),
        "verdicts": verdicts,
        "surviving_refinement_set": sorted(
            {m for v in verdicts.values() if v["retained"] for m in v["active"]}
        ),
        # A mechanism retained in any cell is not "discarded" merely because some
        # other cell containing it failed; the factorial can pair it with a loser.
        "discarded": sorted(
            {m for v in verdicts.values() if not v["retained"] for m in v["active"]}
            - {m for v in verdicts.values() if v["retained"] for m in v["active"]}
        ),
        "other_skews_present": contaminated,
    }


def load_round_metrics(csv_path: Path) -> pd.DataFrame:
    """Load a single job's per-round telemetry CSV."""
    return pd.read_csv(csv_path)


def compute_target_floor(runs: list[RunRecord], dataset: str, alpha: float) -> float:
    """90% of the mean final-round (R=100) top-1 accuracy of the uncompressed FedAvg
    reference, averaged across FedAvg's 3 seeds for this (dataset, alpha)."""
    fedavg_runs = [
        r for r in runs if r.dataset == dataset and r.alpha == alpha and r.algorithm == "fedavg"
    ]
    if not fedavg_runs:
        raise ValueError(f"No FedAvg reference runs found for dataset={dataset}, alpha={alpha}")
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
    """
    result: dict = {}
    runs = confirmatory_runs(runs)
    # ``algorithm_config``, not ``algorithm``: the ablation arms declare
    # ``name: fedmaq`` and would otherwise be counted as formulation candidates.
    fedmaq_runs = [r for r in runs if r.algorithm_config == "fedmaq" and r.formulation is not None]
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
                # margin between the top-2 candidates is smaller than their
                # pooled seed-to-seed variability, it's noise, not a real gap --
                # re-select by higher mean top-1 accuracy at R=100 instead.
                pooled = qualified[top1]["crossing_mbs"] + qualified[top2]["crossing_mbs"]
                pooled_stdev = statistics.stdev(pooled) if len(pooled) > 1 else 0.0
                if margin < pooled_stdev:
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


def compare_to_baselines(runs: list[RunRecord], winner_result: dict) -> dict:
    """Headline baseline comparison (chapter_4.tex Section 4, statistical
    procedure + convergence-stability metric).

    For each (dataset, alpha) with a qualified formulation winner, pairs that
    winning formulation's 3 FedMAQ seed runs against each baseline algorithm's
    3 seed runs by seed number. Reports the paired per-seed accuracy-at-R100
    delta (FedMAQ - baseline: mean + min/max, no bootstrap/CI) and
    rounds-to-target for both sides using the same target-accuracy floor the
    winner rule already computed.
    """
    result: dict = {}
    runs = confirmatory_runs(runs)
    for entry in winner_result.values():
        if entry["winner"] is None:
            continue
        dataset, alpha, formulation, floor = (
            entry["dataset"],
            entry["alpha"],
            entry["winner"],
            entry["target_accuracy_floor"],
        )
        fedmaq_by_seed = {
            r.seed: r
            for r in runs
            if r.dataset == dataset
            and r.alpha == alpha
            and r.algorithm_config == "fedmaq"
            and r.formulation == formulation
        }
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
                    and r.experiment_group != ABLATION_GROUP
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
