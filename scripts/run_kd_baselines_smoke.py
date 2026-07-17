"""Process-isolated runner for MobileNetV2GN smoke tests (KD Baselines, minus FedMD).

Runs FedDistill, CFD, and FedAvg+KD for 50 rounds each under Dirichlet values 0.1 and 1.0.
FedMD is excluded pending a feasibility decision on its pretrain-epoch budget (see
docs/DECISIONS.md and HANDOFF.md) -- too slow to include in this pass.
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
        description="Run MobileNetV2GN smoke tests for FedDistill, CFD, and FedAvg+KD."
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
    parser.add_argument(
        "--start_at",
        type=int,
        default=1,
        help="1-based run index to resume from (skips earlier completed runs)",
    )
    args = parser.parse_args()

    # Define the run configurations
    runs = []
    for het in ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]:
        for alg in ["feddistill", "cfd", "fedavg_kd"]:
            runs.append(
                {
                    "alg": alg,
                    "het": het,
                    "overrides": [],
                    "label": f"{alg}-{het}",
                }
            )

    print("=" * 60)
    print("MobileNetV2GN 50-Round KD Baseline Smoke Tests (FedMD excluded)")
    print(f"Total Rounds: {args.total_rounds}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total Runs: {len(runs)}")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("=" * 60)

    completed = 0
    failed = 0
    for i, run_info in enumerate(runs, 1):
        if i < args.start_at:
            continue
        label = run_info["label"]
        alg = run_info["alg"]
        het = run_info["het"]

        target_dir = f"{args.output_dir}/{alg}/{het}"

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run.py",
            "dataset=cifar10",
            f"heterogeneity={het}",
            "experiment=default",
            f"algorithm={alg}",
            f"experiment.total_rounds={args.total_rounds}",
            "seed=0",
            "experiment.client_gpus=1.0",
            f"hydra.run.dir={target_dir}",
        ] + run_info["overrides"]

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(runs)}] Starting: {label} (attempt {attempt}/{max_attempts})")
            print(f"{'=' * 60}")

            kill_ray_processes()

            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd)

            if res.returncode == 0:
                print(f"\n[SUCCESS] Run '{label}' completed successfully.")
                completed += 1
                break

            print(
                f"\n[ERROR] Run '{label}' failed with exit code {res.returncode} "
                f"(attempt {attempt}/{max_attempts}) -- likely a Ray node crash "
                "(known Windows Ray instability). Retrying from scratch."
            )
            if attempt == max_attempts:
                print(f"[GIVING UP] Run '{label}' failed after {max_attempts} attempts.")
                failed += 1

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
