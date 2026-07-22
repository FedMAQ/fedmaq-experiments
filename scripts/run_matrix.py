"""Declarative Matrix Sweep Runner for FedMAQ Experiments.

Reads an experiment matrix specification (YAML) from ``conf/matrix/`` and executes
the experiment grid using process isolation, cross-platform Ray cleanup, canonical pathing,
and optional resume support.

Usage:
    uv run python scripts/run_matrix.py --matrix conf/matrix/ci_test.yaml
    uv run python scripts/run_matrix.py --matrix ci_test --dry_run
    uv run python scripts/run_matrix.py --matrix pass2_explore --start_at 3
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path if not present
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from scripts.common import (
    build_run_command,
    get_canonical_output_dir,
    kill_ray_processes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedmaq.run_matrix")


def resolve_matrix_path(matrix_arg: str) -> Path:
    """Resolve matrix file path from argument string."""
    path = Path(matrix_arg)
    if path.exists() and path.is_file():
        return path

    # Try appending .yaml or looking in conf/matrix/
    if not matrix_arg.endswith(".yaml"):
        matrix_arg += ".yaml"

    path_in_conf = Path("conf/matrix") / matrix_arg
    if path_in_conf.exists() and path_in_conf.is_file():
        return path_in_conf

    raise FileNotFoundError(
        f"Matrix file not found: '{matrix_arg}' (checked path and conf/matrix/)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Declarative Matrix Sweep Runner for FedMAQ Experiments"
    )
    parser.add_argument(
        "--matrix",
        type=str,
        required=True,
        help="Path or name of the matrix YAML file (e.g. 'ci_test' or 'conf/matrix/mobilenetv2_smoke_50r.yaml')",
    )
    parser.add_argument(
        "--start_at",
        type=int,
        default=1,
        help="1-indexed run number to resume execution from (default: 1)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Display planned execution grid without running commands",
    )
    args = parser.parse_args()

    matrix_path = resolve_matrix_path(args.matrix)
    logger.info(f"Loading experiment matrix from {matrix_path}")
    cfg = OmegaConf.load(matrix_path)

    phase = cfg.get("phase", "smoke")
    exp_group = cfg.get("experiment_group", matrix_path.stem)
    dataset = cfg.get("dataset", "cifar10")
    model = cfg.get("model", "mobilenetv2")
    total_rounds = int(cfg.get("total_rounds", 50))
    client_gpus = float(cfg.get("client_gpus", 1.0))
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    heterogeneities = list(cfg.get("heterogeneities", ["dirichlet_alpha_0.1"]))
    runs_spec = cfg.get("runs", [])

    # Expand all permutations into concrete run tasks
    tasks = []
    for het in heterogeneities:
        for seed in seeds:
            for run_item in runs_spec:
                alg = run_item.get("alg")
                label = run_item.get("label", alg)
                overrides = list(run_item.get("overrides", []))

                output_dir = get_canonical_output_dir(
                    phase=phase,
                    dataset=dataset,
                    model=model,
                    exp_group=exp_group,
                    algorithm=alg,
                    heterogeneity=het,
                    seed=seed,
                )

                cmd = build_run_command(
                    dataset=dataset,
                    heterogeneity=het,
                    algorithm=alg,
                    total_rounds=total_rounds,
                    seed=seed,
                    client_gpus=client_gpus,
                    target_dir=output_dir,
                    overrides=overrides,
                )

                tasks.append(
                    {
                        "label": f"{label}-{het}-seed{seed}",
                        "alg": alg,
                        "het": het,
                        "seed": seed,
                        "output_dir": output_dir,
                        "cmd": cmd,
                    }
                )

    print("=" * 70)
    print(f"FedMAQ Matrix Sweep: {exp_group.upper()}")
    print(f"Phase: {phase} | Dataset: {dataset} | Model: {model}")
    print(f"Total Rounds: {total_rounds} | Client GPUs: {client_gpus}")
    print(f"Heterogeneities: {heterogeneities}")
    print(f"Seeds: {seeds}")
    print(f"Total Runs Scheduled: {len(tasks)}")
    if args.start_at > 1:
        print(f"Resuming from Run Index: {args.start_at}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN MODE] The following commands would be executed:")
        for idx, task in enumerate(tasks, 1):
            skip_mark = " (SKIPPED)" if idx < args.start_at else ""
            print(f"\nTask {idx}/{len(tasks)} [{task['label']}]{skip_mark}")
            print(f" Target Dir: {task['output_dir']}")
            print(f" Command:    {' '.join(task['cmd'])}")
        print("\nDry run completed successfully.")
        sys.exit(0)

    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for idx, task in enumerate(tasks, 1):
        if idx < args.start_at:
            logger.info(
                f"[{idx}/{len(tasks)}] Skipping run '{task['label']}' (--start_at {args.start_at})"
            )
            skipped += 1
            continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{idx}/{len(tasks)}] Starting: {task['label']}")
        logger.info(f"Target Dir: {task['output_dir']}")
        logger.info(f"Command: {' '.join(task['cmd'])}")
        logger.info(f"{'=' * 70}")

        kill_ray_processes()

        run_start = time.time()
        res = subprocess.run(task["cmd"])
        elapsed = time.time() - run_start

        if res.returncode != 0:
            logger.error(
                f"[FAILED] Task [{idx}/{len(tasks)}] '{task['label']}' exited with code {res.returncode} ({elapsed:.1f}s)"
            )
            failed += 1
        else:
            logger.info(
                f"[SUCCESS] Task [{idx}/{len(tasks)}] '{task['label']}' completed in {elapsed:.1f}s"
            )
            completed += 1

        time.sleep(3)

    logger.info(f"\n{'=' * 70}")
    logger.info("Cleaning up Ray processes after sweep...")
    kill_ray_processes()

    total_elapsed = time.time() - start_time
    logger.info(f"\nSweep '{exp_group}' finished at {datetime.now().isoformat()}")
    logger.info(f"Total time elapsed: {total_elapsed / 60.0:.2f} minutes")
    logger.info(f"  Completed: {completed}/{len(tasks)}")
    logger.info(f"  Failed:    {failed}/{len(tasks)}")
    logger.info(f"  Skipped:   {skipped}/{len(tasks)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
