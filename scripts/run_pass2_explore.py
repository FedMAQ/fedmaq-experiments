"""Priority 1 exploration, Pass 2 -- capacity-EMA / grad-norm-smoothing /
client-KD-reg+proximal isolation ablations (MobileNetV2GN).

Scoped via grilling session 2026-07-18 (HANDOFF.md Priority 1 / DECISIONS.md
entries 29, 33-35).

- explore-alpha = 0.3, 50 rounds, single seed (seed=0), matches Pass 1
- Base config: soft-voting fixed at Pass 1's provisional pick
  (entropy_weight=2.0, precision_weight=0.5, soft_voting=true)
- Three independent isolation arms (grouped, largely orthogonal per Decision 29):
    1. capacity-EMA: ema_decay in {0.3, 0.5, 0.7, 0.9} + off (ema_student=false) -- 5 runs
    2. grad-norm-smoothing: grad_norm_beta in {0.3, 0.5, 0.7, 0.9} + off (grad_norm_ema=false) -- 5 runs
    3. client-KD-reg+proximal: kd_prox_mu in {0.01, 0.1, 1.0} (client_kd_reg=true,
       kd_reg_alpha=0.5, kd_reg_temp=2.0 fixed at yaml defaults) + off
       (client_kd_reg=false, kd_prox_mu=0.0) -- 4 runs
- Whichever mechanism isn't being swept in a given arm stays at its
  algorithm/fedmaq.yaml default.
- Decision rule: noise margin applied manually when reading results (Decision 30),
  not encoded here.
"""

import subprocess
import sys
import time
from datetime import datetime

EXPLORE_HET = "dirichlet_alpha_0.3"
TOTAL_ROUNDS = 50
SEED = 0

# Pass 1 provisional pick (DECISIONS.md entry 34)
ENTROPY_WEIGHT = 2.0
PRECISION_WEIGHT = 0.5


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


def build_ema_runs():
    """Capacity-EMA isolation: ema_decay sweep + off control."""
    runs = []
    for decay in [0.3, 0.5, 0.7, 0.9]:
        runs.append(
            {
                "overrides": [
                    "algorithm.ema_student=true",
                    f"algorithm.ema_decay={decay}",
                ],
                "label": f"ema_decay{decay}",
                "phase": "ema",
            }
        )
    runs.append(
        {
            "overrides": ["algorithm.ema_student=false"],
            "label": "ema_off",
            "phase": "ema",
        }
    )
    return runs


def build_grad_norm_runs():
    """Grad-norm-smoothing isolation: grad_norm_beta sweep + off control."""
    runs = []
    for beta in [0.3, 0.5, 0.7, 0.9]:
        runs.append(
            {
                "overrides": [
                    "algorithm.grad_norm_ema=true",
                    f"algorithm.grad_norm_beta={beta}",
                ],
                "label": f"grad_norm_beta{beta}",
                "phase": "grad_norm",
            }
        )
    runs.append(
        {
            "overrides": ["algorithm.grad_norm_ema=false"],
            "label": "grad_norm_off",
            "phase": "grad_norm",
        }
    )
    return runs


def build_kd_reg_runs():
    """Client-KD-reg+proximal isolation: kd_prox_mu sub-sweep + off control."""
    runs = []
    for mu in [0.01, 0.1, 1.0]:
        runs.append(
            {
                "overrides": [
                    "algorithm.client_kd_reg=true",
                    "algorithm.kd_reg_alpha=0.5",
                    "algorithm.kd_reg_temp=2.0",
                    f"algorithm.kd_prox_mu={mu}",
                ],
                "label": f"kd_reg_mu{mu}",
                "phase": "kd_reg",
            }
        )
    runs.append(
        {
            "overrides": [
                "algorithm.client_kd_reg=false",
                "algorithm.kd_prox_mu=0.0",
            ],
            "label": "kd_reg_off",
            "phase": "kd_reg",
        }
    )
    return runs


def run_single(run, index, total, output_dir_base):
    target_dir = f"{output_dir_base}/{index}"
    print("\n==================================================")
    print(
        f"Starting run {index + 1}/{total}: {run['label']} | "
        f"alpha=0.3 | R={TOTAL_ROUNDS} | phase={run['phase']}"
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
        f"heterogeneity={EXPLORE_HET}",
        "experiment=preliminary",
        "algorithm=fedmaq",
        "algorithm.formulation=3",
        "algorithm.soft_voting=true",
        f"algorithm.entropy_weight={ENTROPY_WEIGHT}",
        f"algorithm.precision_weight={PRECISION_WEIGHT}",
        *run["overrides"],
        f"experiment.total_rounds={TOTAL_ROUNDS}",
        "experiment.local_epochs=5",
        f"seed={SEED}",
        f"hydra.run.dir={target_dir}",
    ]

    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)

    if res.returncode != 0:
        print(f"\n[ERROR] Run {run['label']} failed with exit code {res.returncode}")
    else:
        print(f"\n[SUCCESS] Run {run['label']} completed.")

    time.sleep(5)


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    output_dir_base = f"multirun/{date_str}/{time_str}-pass2-explore-mobilenetv2"

    runs = build_ema_runs() + build_grad_norm_runs() + build_kd_reg_runs()
    for i, run in enumerate(runs):
        run["index"] = i

    total = len(runs)
    print("==================================================")
    print(f"Pass 2 Explore (MobileNetV2GN): {output_dir_base}")
    print(f"Total Runs: {total} (ema: 5, grad_norm: 5, kd_reg: 4)")
    print("==================================================")

    for run in runs:
        run_single(run, run["index"], total, output_dir_base)

    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
