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
        description="Run smoke test for federated learning algorithms."
    )
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=50,
        help="Total number of communication rounds",
    )
    args = parser.parse_args()

    # Generate a single unified timestamp for this smoke run session
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    output_dir_base = f"multirun/{date_str}/{time_str}-smoke"

    # DO NOT REMOVE the original full algorithm sweep below (kept commented out):
    # runs = [
    #     {"alg": "fedavg", "index": 0},
    #     {"alg": "fedprox", "index": 1},
    #     {"alg": "fedpaq", "index": 2},
    #     {"alg": "dadaquant", "index": 3},
    #     {"alg": "cfd", "index": 4},
    #     {"alg": "fedkd", "index": 5},
    #     # {"alg": "fedmd", "index": 6},
    #     {"alg": "feddistill", "index": 7},
    #     {"alg": "fedmaq", "index": 8},
    # ]

    # Sweep across both Dirichlet alpha regimes (0.1 and 1.0) and run without EMA
    runs = []
    idx = 0
    for het in ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]:
        # Run: No EMA
        runs.append(
            {
                "alg": "fedmaq",
                "het": het,
                "formulation": 3,
                "ema_student": "false",
                "ema_decay": 0.99,
                "label": "no_ema",
                "index": idx,
            }
        )
        idx += 1

    print("==================================================")
    print(f"Smoke Run Base Directory: {output_dir_base}")
    print("Running FedMAQ Formulation 3 without EMA")
    print(f"Total Rounds: {args.total_rounds}")
    print("==================================================")

    for run in runs:
        target_dir = f"{output_dir_base}/{run['index']}"
        print("\n==================================================")
        print(f"Starting run for {run['alg']} ({run['label']}) on {run['het']} in {target_dir}")
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
