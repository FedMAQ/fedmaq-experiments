"""Post-hoc formulation-study winner-selection harness (chapter_4.tex, Section "Formulation Study").

Discovers Hydra multirun job directories, joins each job's ``experiment_log.csv``
against its resolved ``.hydra/config.yaml`` (dataset, alpha, algorithm, formulation,
seed — none of which are logged into the CSV itself), and applies the pre-registered
winner rule: per (dataset, alpha), the formulation that reaches the target accuracy
using the least mean cumulative communication (MB) across its 3 seeds wins; any
formulation with a seed that never crosses the target-accuracy floor within the
fixed round budget is disqualified for that (dataset, alpha) regardless of payload.

The target-accuracy floor is 90% of the mean final-round top-1 accuracy of the
uncompressed FedAvg reference, averaged across FedAvg's 3 seeds for that
(dataset, alpha), reusing FedAvg runs already present in the main benchmark grid.
"""

from __future__ import annotations

import argparse
import json
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


def discover_runs(experiments_root: Path) -> list[RunRecord]:
    """Walk multirun/<date>/<time>/<job_idx>/ dirs and join each job's resolved config."""
    runs: list[RunRecord] = []
    for config_path in sorted(experiments_root.glob("multirun/*/*/*/.hydra/config.yaml")):
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
            )
        )
    return runs


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
            for r in seed_runs:
                round_num, cumulative_mb = first_crossing(load_round_metrics(r.csv_path), floor)
                seed_results[r.seed] = {"round": round_num, "cumulative_mb": cumulative_mb}
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
            }

        qualified = {f: d for f, d in detail.items() if not d["disqualified"]}
        if qualified:
            winner = min(qualified, key=lambda f: qualified[f]["mean_cumulative_mb"])
            sorted_mbs = sorted(d["mean_cumulative_mb"] for d in qualified.values())
            margin = (sorted_mbs[1] - sorted_mbs[0]) if len(sorted_mbs) > 1 else None
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Formulation-study winner-selection harness")
    parser.add_argument("--experiments-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/analysis_output/formulation_winner.json")
    )
    args = parser.parse_args()

    runs = discover_runs(args.experiments_root)
    result = select_winner(runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote formulation-winner verdict to {args.output}")


if __name__ == "__main__":
    main()
