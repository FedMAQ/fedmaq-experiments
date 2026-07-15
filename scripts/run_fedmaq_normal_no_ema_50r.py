"""Process-isolated runner for the FedMAQ Normal (ResNet18GN) without EMA experiment.

Runs FedMAQ normal for 50 rounds under Dirichlet values 0.1 and 1.0,
incorporating the best regularization parameters found so far:
  α=0.1: kd_reg_alpha=0.5, kd_reg_temp=1.0, entropy_weight=1.0, precision_weight=1.0, kd_prox_mu=0.1, ema_decay=0.7
  α=1.0: kd_reg_alpha=0.3, kd_reg_temp=2.0, entropy_weight=2.0, precision_weight=0.5, kd_prox_mu=0.1, ema_decay=0.1

EMA is explicitly disabled (algorithm.ema_student=false).
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime


def kill_ray_processes() -> None:
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


# Per-alpha tuned hyperparameters (the best ones found so far)
ALPHA_CONFIGS = {
    "dirichlet_alpha_0.1": {
        "ema_decay": 0.7,
        "entropy_weight": 1.0,
        "precision_weight": 1.0,
        "kd_reg_alpha": 0.5,
        "kd_reg_temp": 1.0,
        "kd_prox_mu": 0.1,
    },
    "dirichlet_alpha_1.0": {
        "ema_decay": 0.1,
        "entropy_weight": 2.0,
        "precision_weight": 0.5,
        "kd_reg_alpha": 0.3,
        "kd_reg_temp": 2.0,
        "kd_prox_mu": 0.1,
    },
}


def build_runs(total_rounds: int, output_dir_base: str) -> list[tuple[str, list[str]]]:
    """Build the list of (label, cmd) run configurations."""
    runs = []

    for het, config in ALPHA_CONFIGS.items():
        label = f"{het}/no_ema_50r"
        target_dir = f"{output_dir_base}/{het}"
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
            "algorithm.ema_student=false",  # Explicitly disable student EMA
            f"algorithm.ema_decay={config['ema_decay']}",
            f"algorithm.entropy_weight={config['entropy_weight']}",
            f"algorithm.precision_weight={config['precision_weight']}",
            "algorithm.client_kd_reg=true",
            f"algorithm.kd_reg_alpha={config['kd_reg_alpha']}",
            f"algorithm.kd_reg_temp={config['kd_reg_temp']}",
            f"algorithm.kd_prox_mu={config['kd_prox_mu']}",
            f"hydra.run.dir={target_dir}",
        ]
        runs.append((label, cmd))

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 50-round FedMAQ normal (ResNet18GN) without EMA sweeps."
    )
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=50,
        help="Total number of communication rounds (default: 50)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/fedmaq-normal-no-ema-50r",
        help="Base output directory for sweep results",
    )
    args = parser.parse_args()

    runs = build_runs(args.total_rounds, args.output_dir)

    print("=" * 60)
    print("FedMAQ Normal (ResNet18GN) 50-Round Sweeps (No EMA)")
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
