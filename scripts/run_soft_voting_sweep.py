"""Soft-voting ablation + hyperparameter sweep for FedMAQ.

Phase 1 — Ablation (4 runs, 50 rounds):
    soft_voting on/off × {α=0.1 (ema_decay=0.7), α=1.0 (ema_decay=0.1)}
    Quantifies the isolated contribution of soft-voting.

Phase 2 — Hyperparameter sweep (32 runs, 40 rounds):
    entropy_weight × precision_weight = {0.5, 1.0, 2.0, 4.0}²
    × {α=0.1 (ema_decay=0.7), α=1.0 (ema_decay=0.1)}
    Finds the best soft-voting operating point per heterogeneity regime.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime


def kill_ray_processes():
    print("Stopping Ray and cleaning up lingering processes...")
    subprocess.run(
        ["uv", "run", "ray", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "raylet.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "gcs_server.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(3)


# Best EMA decay per heterogeneity regime (from EMA sweep results)
BEST_EMA = {
    "dirichlet_alpha_0.1": 0.7,
    "dirichlet_alpha_1.0": 0.1,
}


def build_ablation_runs():
    """Phase 1: soft-voting on/off at best EMA settings (50 rounds)."""
    runs = []
    heterogeneities = ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]

    for het in heterogeneities:
        for sv_enabled in [True, False]:
            label = f"sv_{'on' if sv_enabled else 'off'}"
            runs.append(
                {
                    "het": het,
                    "ema_decay": BEST_EMA[het],
                    "soft_voting": str(sv_enabled).lower(),
                    "entropy_weight": 1.0,
                    "precision_weight": 1.0,
                    "label": label,
                    "total_rounds": 50,
                    "phase": "ablation",
                }
            )
    return runs


def build_sweep_runs():
    """Phase 2: entropy_weight × precision_weight grid (40 rounds)."""
    runs = []
    heterogeneities = ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]
    gamma_values = [0.5, 1.0, 2.0, 4.0]

    for het in heterogeneities:
        for ew in gamma_values:
            for pw in gamma_values:
                label = f"ew{ew}_pw{pw}"
                runs.append(
                    {
                        "het": het,
                        "ema_decay": BEST_EMA[het],
                        "soft_voting": "true",
                        "entropy_weight": ew,
                        "precision_weight": pw,
                        "label": label,
                        "total_rounds": 40,
                        "phase": "sweep",
                    }
                )
    return runs


def run_single(run, index, total, output_dir_base):
    """Execute a single experiment run in a fresh subprocess."""
    target_dir = f"{output_dir_base}/{index}"
    alpha_str = run["het"].split("_")[-1]
    print("\n==================================================")
    print(
        f"Starting run {index + 1}/{total}: "
        f"{run['label']} | alpha={alpha_str} | "
        f"ema={run['ema_decay']} | "
        f"R={run['total_rounds']} | "
        f"phase={run['phase']}"
    )
    print(f"Output: {target_dir}")
    print("==================================================")

    kill_ray_processes()

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run.py",
        "dataset=cifar10",
        f"heterogeneity={run['het']}",
        "experiment=preliminary",
        "algorithm=fedmaq",
        "algorithm.formulation=3",
        "algorithm.ema_student=true",
        f"algorithm.ema_decay={run['ema_decay']}",
        f"algorithm.soft_voting={run['soft_voting']}",
        f"algorithm.entropy_weight={run['entropy_weight']}",
        f"algorithm.precision_weight={run['precision_weight']}",
        f"experiment.total_rounds={run['total_rounds']}",
        "experiment.local_epochs=5",
        "seed=0",
        f"hydra.run.dir={target_dir}",
    ]

    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)

    if res.returncode != 0:
        print(
            f"\n[ERROR] Run {run['label']} (alpha={alpha_str}) failed with exit code {res.returncode}"
        )
    else:
        print(f"\n[SUCCESS] Run {run['label']} (alpha={alpha_str}) completed.")

    time.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description="Soft-voting ablation + hyperparameter sweep for FedMAQ."
    )
    parser.add_argument(
        "--phase",
        choices=["all", "ablation", "sweep"],
        default="all",
        help=(
            "Which phase(s) to run: 'ablation' (4 runs, 50R), "
            "'sweep' (32 runs, 40R), or 'all' (36 runs)"
        ),
    )
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    output_dir_base = f"multirun/{date_str}/{time_str}-soft-voting-sweep"

    runs = []
    if args.phase in ("all", "ablation"):
        runs.extend(build_ablation_runs())
    if args.phase in ("all", "sweep"):
        runs.extend(build_sweep_runs())

    # Assign sequential indices
    for i, run in enumerate(runs):
        run["index"] = i

    total = len(runs)
    ablation_count = sum(1 for r in runs if r["phase"] == "ablation")
    sweep_count = sum(1 for r in runs if r["phase"] == "sweep")

    print("==================================================")
    print(f"Soft-Voting Sweep: {output_dir_base}")
    print(f"Total Runs: {total} (ablation: {ablation_count}, sweep: {sweep_count})")
    print("==================================================")

    for run in runs:
        run_single(run, run["index"], total, output_dir_base)

    # Final cleanup
    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
