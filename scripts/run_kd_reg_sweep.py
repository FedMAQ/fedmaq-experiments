"""Process-isolated runner for the Client-Side KD Regularization sweep.

Sweeps kd_reg_alpha ∈ {0.1, 0.3, 0.5, 0.7} × kd_reg_temp ∈ {1.0, 2.0}
plus a no-regularization baseline, across both Dirichlet α ∈ {0.1, 1.0}.

Per-alpha overrides (from tuned FedMAQ-Lite configs):
  α=0.1: ema_decay=0.7, entropy_weight=4.0, precision_weight=1.0
  α=1.0: ema_decay=0.1, entropy_weight=2.0, precision_weight=0.5

Total: 9 configs × 2 alphas = 18 runs.
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


# Per-alpha tuned hyperparameters (from FedMAQ-Lite sweep results)
ALPHA_OVERRIDES = {
    "dirichlet_alpha_0.1": {
        "ema_decay": 0.7,
        "entropy_weight": 4.0,
        "precision_weight": 1.0,
    },
    "dirichlet_alpha_1.0": {
        "ema_decay": 0.1,
        "entropy_weight": 2.0,
        "precision_weight": 0.5,
    },
}

# KD regularization sweep grid
KD_REG_ALPHAS = [0.1, 0.3, 0.5, 0.7]
KD_REG_TEMPS = [1.0, 2.0]


def build_runs(total_rounds: int, output_dir_base: str):
    """Build the list of (label, cmd) run configurations."""
    runs = []

    for het, overrides in ALPHA_OVERRIDES.items():
        # 1. Baseline: no client-side regularization
        label = f"{het}/baseline_no_reg"
        target_dir = f"{output_dir_base}/{het}/baseline_no_reg"
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run.py",
            "dataset=cifar10",
            f"heterogeneity={het}",
            "experiment=preliminary",
            "algorithm=fedmaq",
            f"experiment.total_rounds={total_rounds}",
            "seed=0",
            "experiment.client_gpus=1.0",
            f"algorithm.ema_decay={overrides['ema_decay']}",
            f"algorithm.entropy_weight={overrides['entropy_weight']}",
            f"algorithm.precision_weight={overrides['precision_weight']}",
            "algorithm.client_kd_reg=false",
            f"hydra.run.dir={target_dir}",
        ]
        runs.append((label, cmd))

        # 2. KD regularization sweep
        for alpha in KD_REG_ALPHAS:
            for temp in KD_REG_TEMPS:
                label = f"{het}/kd_reg_alpha_{alpha}_temp_{temp}"
                target_dir = f"{output_dir_base}/{het}/kd_reg_alpha_{alpha}_temp_{temp}"
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/run.py",
                    "dataset=cifar10",
                    f"heterogeneity={het}",
                    "experiment=preliminary",
                    "algorithm=fedmaq",
                    f"experiment.total_rounds={total_rounds}",
                    "seed=0",
                    "experiment.client_gpus=1.0",
                    f"algorithm.ema_decay={overrides['ema_decay']}",
                    f"algorithm.entropy_weight={overrides['entropy_weight']}",
                    f"algorithm.precision_weight={overrides['precision_weight']}",
                    "algorithm.client_kd_reg=true",
                    f"algorithm.kd_reg_alpha={alpha}",
                    f"algorithm.kd_reg_temp={temp}",
                    f"hydra.run.dir={target_dir}",
                ]
                runs.append((label, cmd))

    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Run Client-Side KD Regularization sweep for FedMAQ (ResNet18GN)."
    )
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=40,
        help="Total number of communication rounds (default: 40)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/client-kd-reg-sweep-7-15",
        help="Base output directory for sweep results",
    )
    args = parser.parse_args()

    runs = build_runs(args.total_rounds, args.output_dir)

    print("=" * 60)
    print("Client-Side KD Regularization Sweep for FedMAQ (ResNet18GN)")
    print(f"Total Rounds: {args.total_rounds}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total Runs: {len(runs)}")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("=" * 60)

    completed = 0
    failed = 0
    for i, (label, cmd) in enumerate(runs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(runs)}] Starting: {label}")
        print(f"{'=' * 60}")

        kill_ray_processes()

        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(f"\n[ERROR] Run '{label}' failed with exit code {res.returncode}")
            failed += 1
        else:
            print(f"\n[SUCCESS] Run '{label}' completed successfully.")
            completed += 1

        time.sleep(5)

    # Final cleanup
    print(f"\n{'=' * 60}")
    print("Final clean up of Ray processes...")
    kill_ray_processes()
    print(f"\nSweep complete at {datetime.now().isoformat()}")
    print(f"  Completed: {completed}/{len(runs)}")
    print(f"  Failed:    {failed}/{len(runs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
