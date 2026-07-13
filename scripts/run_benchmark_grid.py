import subprocess
import time
import os
import sys
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
    else:
        subprocess.run(
            ["pkill", "-9", "-f", "raylet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["pkill", "-9", "-f", "gcs_server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(3)


# Generate a single unified timestamp for this benchmark grid session
date_str = datetime.now().strftime("%Y-%m-%d")
time_str = datetime.now().strftime("%H-%M-%S")
output_dir_base = f"multirun/{date_str}/{time_str}-benchmark"

datasets = ["cifar10", "cifar100"]
heterogeneities = ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"]
algorithms = [
    "fedavg",
    "fedprox",
    "fedpaq",
    "dadaquant",
    "cfd",
    "fedkd",
    "fedmd",
    "feddistill",
    "fedmaq",
]
seeds = [0, 42, 123]

# Build the queue of all runs
runs = []
run_idx = 0
for dataset in datasets:
    for alpha in heterogeneities:
        for alg in algorithms:
            for seed in seeds:
                runs.append(
                    {
                        "dataset": dataset,
                        "heterogeneity": alpha,
                        "algorithm": alg,
                        "seed": seed,
                        "index": run_idx,
                    }
                )
                run_idx += 1

print(f"==================================================")
print(f"Benchmark Grid Base Directory: {output_dir_base}")
print(f"Total Scheduled Jobs: {len(runs)}")
print(f"==================================================")

for run in runs:
    target_dir = f"{output_dir_base}/{run['index']}"
    print(f"\n==================================================")
    print(
        f"Job {run['index'] + 1}/{len(runs)} | Starting {run['algorithm']} on {run['dataset']} ({run['heterogeneity']}) seed={run['seed']}"
    )
    print(f"Target Directory: {target_dir}")
    print(f"==================================================")

    kill_ray_processes()

    # We use experiment.client_gpus=1.0 to prevent concurrent Ray actors from exhausting GPU VRAM
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run.py",
        f"dataset={run['dataset']}",
        f"heterogeneity={run['heterogeneity']}",
        "experiment=default",
        f"algorithm={run['algorithm']}",
        f"seed={run['seed']}",
        "experiment.client_gpus=1.0",
        f"hydra.run.dir={target_dir}",
    ]

    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)

    if res.returncode != 0:
        print(f"\n[ERROR] Job failed with exit code {res.returncode}")
    else:
        print(f"\n[SUCCESS] Job completed successfully.")

    time.sleep(5)

# Final cleanup
print("\nFinal clean up of Ray processes...")
kill_ray_processes()
print("Done.")
