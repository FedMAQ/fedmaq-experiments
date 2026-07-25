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
        runs.append(
            RunRecord(
                job_dir=job_dir,
                dataset=cfg["dataset"]["name"],
                alpha=float(cfg["heterogeneity"]["alpha"]),
                algorithm=cfg["algorithm"]["name"],
                formulation=cfg["algorithm"].get("formulation"),
                seed=int(cfg["seed"]),
                csv_path=csv_path,
                refinements=(
                    bool(cfg["algorithm"].get("soft_voting", False)),
                    bool(cfg["algorithm"].get("ema_student", False)),
                    bool(cfg["algorithm"].get("grad_norm_ema", False)),
                ),
            )
        )
    return runs


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
    fedmaq_runs = [r for r in runs if r.algorithm == "fedmaq" and r.formulation is not None]
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
            and r.algorithm == "fedmaq"
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


if __name__ == "__main__":
    main()
