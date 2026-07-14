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


def main():
    parser = argparse.ArgumentParser(
        description="Run 40-round full FedMAQ (ResNet18GN) baseline sweeps."
    )
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=40,
        help="Total number of communication rounds",
    )
    args = parser.parse_args()

    # Base output directory
    output_dir_base = "experiments/baseline-comparison-resnet18"

    runs = [
        {
            "het": "dirichlet_alpha_0.1",
            "target_dir": f"{output_dir_base}/dirichlet_alpha_0.1",
        },
        {
            "het": "dirichlet_alpha_1.0",
            "target_dir": f"{output_dir_base}/dirichlet_alpha_1.0",
        },
    ]

    print("==================================================")
    print("Running Full FedMAQ (ResNet18GN) Sweeps")
    print(f"Total Rounds: {args.total_rounds}")
    print("==================================================")

    for run in runs:
        print("\n==================================================")
        print(f"Starting run on {run['het']} in {run['target_dir']}")
        print("==================================================")

        kill_ray_processes()

        # Run with default hyperparams from fedmaq.yaml, using preliminary config profile (K=50, C=0.2)
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run.py",
            "dataset=cifar10",
            f"heterogeneity={run['het']}",
            "experiment=preliminary",
            "algorithm=fedmaq",
            f"experiment.total_rounds={args.total_rounds}",
            "seed=0",
            "experiment.client_gpus=1.0",
            f"hydra.run.dir={run['target_dir']}",
        ]

        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(
                f"\n[ERROR] Run for {run['het']} failed with exit code {res.returncode}"
            )
        else:
            print(f"\n[SUCCESS] Run for {run['het']} completed successfully.")

        time.sleep(5)

    # Final cleanup
    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
