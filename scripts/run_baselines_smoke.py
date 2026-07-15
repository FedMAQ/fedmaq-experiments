"""Process-isolated runner for MobileNetV2GN smoke tests (Additional Baselines).

Runs DAdaQuant, FedPAQ, and FedKD for 50 rounds each under Dirichlet values 0.1 and 1.0.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MobileNetV2GN smoke tests for DAdaQuant, FedPAQ, and FedKD."
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
        default="experiments/mobilenetv2-smoke-50r",
        help="Base output directory for smoke test results",
    )
    args = parser.parse_args()

    # Define the run configurations
    runs = []
    for het in ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]:
        # 1. DAdaQuant
        runs.append(
            {
                "alg": "dadaquant",
                "het": het,
                "overrides": [],
                "label": f"dadaquant-{het}",
            }
        )
        # 2. FedPAQ
        runs.append(
            {
                "alg": "fedpaq",
                "het": het,
                "overrides": [],
                "label": f"fedpaq-{het}",
            }
        )
        # 3. FedKD
        runs.append(
            {
                "alg": "fedkd",
                "het": het,
                "overrides": [],
                "label": f"fedkd-{het}",
            }
        )

    print("=" * 60)
    print("MobileNetV2GN 50-Round Compression Baseline Smoke Tests")
    print(f"Total Rounds: {args.total_rounds}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total Runs: {len(runs)}")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("=" * 60)

    completed = 0
    failed = 0
    for i, run_info in enumerate(runs, 1):
        label = run_info["label"]
        alg = run_info["alg"]
        het = run_info["het"]

        target_dir = f"{args.output_dir}/{alg}/{het}"

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(runs)}] Starting: {label}")
        print(f"{'=' * 60}")

        kill_ray_processes()

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run.py",
            "dataset=cifar10",
            f"heterogeneity={het}",
            "experiment=preliminary",
            f"algorithm={alg}",
            f"experiment.total_rounds={args.total_rounds}",
            "seed=0",
            "experiment.client_gpus=1.0",
            f"hydra.run.dir={target_dir}",
        ] + run_info["overrides"]

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
