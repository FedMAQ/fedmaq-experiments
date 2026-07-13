import subprocess
import time
import os
import sys
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Run smoke test for federated learning algorithms."
    )
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=10,
        help="Total number of communication rounds",
    )
    parser.add_argument(
        "--heterogeneity",
        type=str,
        default="dirichlet_alpha_1.0",
        help="Data heterogeneity configuration",
    )
    args = parser.parse_args()

    # Generate a single unified timestamp for this smoke run session
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    output_dir_base = f"multirun/{date_str}/{time_str}-smoke"

    runs = [
        {"alg": "fedavg", "index": 0},
        {"alg": "fedprox", "index": 1},
        {"alg": "fedpaq", "index": 2},
        {"alg": "dadaquant", "index": 3},
        {"alg": "cfd", "index": 4},
        {"alg": "fedkd", "index": 5},
        # {"alg": "fedmd", "index": 6},
        {"alg": "feddistill", "index": 7},
        {"alg": "fedmaq", "index": 8},
    ]

    print(f"==================================================")
    print(f"Smoke Run Base Directory: {output_dir_base}")
    print(f"Heterogeneity: {args.heterogeneity}")
    print(f"Total Rounds: {args.total_rounds}")
    print(f"==================================================")

    for run in runs:
        target_dir = f"{output_dir_base}/{run['index']}"
        print(f"\n==================================================")
        print(f"Starting run for {run['alg']} in {target_dir}")
        print(f"==================================================")

        kill_ray_processes()

        # We use experiment.client_gpus=1.0 to prevent concurrent Ray actors from exhausting GPU VRAM
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run.py",
            "dataset=cifar10",
            f"heterogeneity={args.heterogeneity}",
            "experiment=preliminary",
            f"algorithm={run['alg']}",
            f"experiment.total_rounds={args.total_rounds}",
            "experiment.local_epochs=5",
            "seed=0",
            "experiment.client_gpus=1.0",
            f"hydra.run.dir={target_dir}",
        ]

        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(
                f"\n[ERROR] Run for {run['alg']} failed with exit code {res.returncode}"
            )
        else:
            print(f"\n[SUCCESS] Run for {run['alg']} completed successfully.")

        time.sleep(5)

    # Final cleanup
    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
