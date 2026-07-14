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
    parser = argparse.ArgumentParser(description="Run EMA sweep for FedMAQ.")
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=50,
        help="Total number of communication rounds",
    )
    args = parser.parse_args()

    # Generate a single unified timestamp for this EMA sweep session
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    output_dir_base = f"multirun/{date_str}/{time_str}-ema-sweep"

    # We sweep ema_decay in 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ema_decays = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    heterogeneities = ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]

    runs = []
    idx = 0
    for het in heterogeneities:
        for decay in ema_decays:
            runs.append(
                {
                    "alg": "fedmaq",
                    "het": het,
                    "formulation": 3,
                    "ema_student": "true",
                    "ema_decay": decay,
                    "label": f"ema_{decay}",
                    "index": idx,
                }
            )
            idx += 1

    print("==================================================")
    print(f"EMA Sweep Base Directory: {output_dir_base}")
    print(f"Total Scheduled Runs: {len(runs)}")
    print(f"Total Rounds per run: {args.total_rounds}")
    print("==================================================")

    for run in runs:
        target_dir = f"{output_dir_base}/{run['index']}"
        print("\n==================================================")
        print(
            f"Starting run {run['index'] + 1}/{len(runs)}: {run['alg']} ({run['label']}) on {run['het']} in {target_dir}"
        )
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
            f"algorithm={run['alg']}",
            f"algorithm.formulation={run['formulation']}",
            f"algorithm.ema_student={run['ema_student']}",
            f"algorithm.ema_decay={run['ema_decay']}",
            f"experiment.total_rounds={args.total_rounds}",
            "experiment.local_epochs=5",
            "seed=0",
            f"hydra.run.dir={target_dir}",
        ]

        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(
                f"\n[ERROR] Run for {run['alg']} ({run['label']}) on {run['het']} failed with exit code {res.returncode}"
            )
        else:
            print(
                f"\n[SUCCESS] Run for {run['alg']} ({run['label']}) on {run['het']} completed successfully."
            )

        time.sleep(5)

    # Final cleanup
    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
