"""Priority 1 exploration, Pass 1 — soft-voting sweep (MobileNetV2GN).

Scoped via grilling session 2026-07-18 (HANDOFF.md Priority 1 / DECISIONS.md).
Supersedes the ResNet18GN-era ``run_soft_voting_sweep.py`` for MobileNetV2GN.

- explore-alpha = 0.3 (distinct from the report grid {0.1, 1.0}; conf/heterogeneity/dirichlet_alpha_0.3.yaml)
- 50 rounds, single seed (seed=0) per run
- Ablation arm: soft_voting on/off (control), at default entropy_weight/precision_weight=1.0
- Sweep arm: entropy_weight x precision_weight = {0.5, 1.0, 2.0, 4.0}^2, soft_voting=true
- EMA/grad-norm/client-KD-reg held at algorithm/fedmaq.yaml defaults (Pass 2's job to resolve)
- Decision rule: keep/drop/revise entropy_weight/precision_weight against a noise
  margin, not just the single best run (single-seed, no variance estimate) --
  applied manually when reading results, not encoded here.
"""

import subprocess
import sys
import time
from datetime import datetime

EXPLORE_HET = "dirichlet_alpha_0.3"
TOTAL_ROUNDS = 50
SEED = 0


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


def build_ablation_runs():
    """Soft-voting on/off control, at default weights (explore-alpha=0.3)."""
    runs = []
    for sv_enabled in [True, False]:
        runs.append(
            {
                "soft_voting": str(sv_enabled).lower(),
                "entropy_weight": 1.0,
                "precision_weight": 1.0,
                "label": f"sv_{'on' if sv_enabled else 'off'}",
                "phase": "ablation",
            }
        )
    return runs


def build_sweep_runs():
    """entropy_weight x precision_weight grid (explore-alpha=0.3)."""
    runs = []
    gamma_values = [0.5, 1.0, 2.0, 4.0]
    for ew in gamma_values:
        for pw in gamma_values:
            runs.append(
                {
                    "soft_voting": "true",
                    "entropy_weight": ew,
                    "precision_weight": pw,
                    "label": f"ew{ew}_pw{pw}",
                    "phase": "sweep",
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
        f"algorithm.soft_voting={run['soft_voting']}",
        f"algorithm.entropy_weight={run['entropy_weight']}",
        f"algorithm.precision_weight={run['precision_weight']}",
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
    output_dir_base = f"multirun/{date_str}/{time_str}-soft-voting-explore-mobilenetv2"

    runs = build_ablation_runs() + build_sweep_runs()
    for i, run in enumerate(runs):
        run["index"] = i

    total = len(runs)
    print("==================================================")
    print(f"Soft-Voting Explore (Pass 1, MobileNetV2GN): {output_dir_base}")
    print(f"Total Runs: {total} (ablation: 2, sweep: {total - 2})")
    print("==================================================")

    for run in runs:
        run_single(run, run["index"], total, output_dir_base)

    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
