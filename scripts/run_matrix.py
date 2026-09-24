"""Declarative Matrix Sweep Runner for FedMAQ Experiments.

The planner owns matrix expansion and command construction. The executor owns
subprocess lifecycle, cleanup, resume, timeout, and status persistence.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import kill_ray_processes
from scripts.matrix_executor import TIMEOUT_RETURNCODE as _TIMEOUT_RETURNCODE
from scripts.matrix_executor import MatrixExecutor
from scripts.matrix_planner import MatrixPlan, plan_matrix
from scripts.run_guard import LockHeldError, prior_evidence, sweep_locks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedmaq.run_matrix")


TIMEOUT_RETURNCODE = _TIMEOUT_RETURNCODE


def resolve_matrix_path(matrix_arg: str) -> Path:
    """Resolve a matrix file path from a path or a name under ``conf/matrix``."""

    path = Path(matrix_arg)
    if path.exists() and path.is_file():
        return path
    if not matrix_arg.endswith(".yaml"):
        matrix_arg += ".yaml"
    path_in_conf = Path("conf/matrix") / matrix_arg
    if path_in_conf.exists() and path_in_conf.is_file():
        return path_in_conf
    raise FileNotFoundError(
        f"Matrix file not found: '{matrix_arg}' (checked path and conf/matrix/)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Declarative Matrix Sweep Runner for FedMAQ Experiments"
    )
    parser.add_argument("--matrix", required=True, help="Path or name of a matrix YAML file")
    parser.add_argument(
        "--start_at", type=int, default=1, help="1-indexed canonical run to resume from"
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip runs whose output directory has a final-round checkpoint",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Display planned commands without running them"
    )
    parser.add_argument("--shard", metavar="I/N", help="Dispatch deterministic shard I of N")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Extra Hydra override applied before each matrix row's own overrides",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        dest="only_labels",
        metavar="LABEL",
        help="Dispatch only rows carrying these labels; cannot combine with --shard",
    )
    parser.add_argument("--max_consecutive_failures", type=int, default=3)
    parser.add_argument("--run_timeout_seconds", type=int, default=0)
    return parser


def _print_plan(plan: MatrixPlan, args: argparse.Namespace) -> None:
    print("=" * 70)
    print(f"FedMAQ Matrix Sweep: {plan.experiment_group.upper()}")
    print(f"Phase: {plan.phase} | Dataset: {plan.dataset} | Model: {plan.model}")
    print(f"Stage: {plan.stage} | Split: {plan.split} | Ledger: {plan.ledger}")
    if plan.experiment:
        print(f"Experiment group: {plan.experiment}")
    print(f"Total Rounds: {plan.total_rounds} | Client GPUs: {plan.client_gpus}")
    print(f"Heterogeneities: {list(plan.heterogeneities)}")
    print(f"Seeds: {list(plan.seeds)}")
    if args.only_labels:
        print(f"Label filter (--only): {args.only_labels}")
    print(f"Canonical Runs: {len(plan.canonical_tasks)}")
    print(f"Total Runs Scheduled: {len(plan.tasks)}")
    if plan.shard is not None:
        print(f"Shard: {plan.shard[0]}/{plan.shard[1]}")
    if args.overrides:
        print(f"Sweep-wide overrides: {' '.join(args.overrides)}")
    if args.start_at > 1:
        print(f"Resuming from Run Index: {args.start_at}")
    if args.skip_completed:
        print("Skipping runs that already hold a final-round checkpoint")
    print("=" * 70)


def _print_dry_run(plan: MatrixPlan, executor: MatrixExecutor) -> None:
    print("\n[DRY RUN MODE] The following commands would be executed:")
    for position, task in enumerate(plan.tasks, 1):
        reason = executor.skip_reason(task)
        skip_mark = f" (SKIPPED: {reason})" if reason else ""
        if not reason and prior_evidence(task.output_dir):
            skip_mark = " (WILL REFUSE: holds an earlier attempt's records; move it aside)"
        print(
            f"\nTask {task.canonical_index}/{len(plan.canonical_tasks)} "
            f"(shard position {position}/{len(plan.tasks)}) [{task.label}]{skip_mark}"
        )
        print(f" Target Dir: {task.output_dir}")
        print(f" Command:    {' '.join(task.command)}")
    print("\nDry run completed successfully.")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.start_at < 1:
        parser.error("--start_at must be >= 1")
    if args.max_consecutive_failures < 0:
        parser.error("--max_consecutive_failures must be >= 0")
    if args.run_timeout_seconds < 0:
        parser.error("--run_timeout_seconds must be >= 0")

    shard = None
    if args.shard:
        from scripts.common import parse_shard

        try:
            shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))
    if shard is not None and args.only_labels:
        parser.error("--only cannot be combined with --shard; shard the full matrix")

    matrix_path = resolve_matrix_path(args.matrix)
    logger.info("Loading experiment matrix from %s", matrix_path)
    cfg = OmegaConf.load(matrix_path)
    try:
        plan = plan_matrix(
            matrix_path,
            cfg,
            overrides=args.overrides,
            only_labels=args.only_labels,
            shard=shard,
        )
    except ValueError as exc:
        parser.error(str(exc))

    executor = MatrixExecutor(
        plan,
        start_at=args.start_at,
        skip_completed=args.skip_completed,
        max_consecutive_failures=args.max_consecutive_failures,
        run_timeout_seconds=args.run_timeout_seconds,
        run=subprocess.run,
        cleanup=kill_ray_processes,
        sleep=time.sleep,
    )
    _print_plan(plan, args)
    if args.dry_run:
        _print_dry_run(plan, executor)
        return

    # Both locks are taken before the executor's first Ray cleanup, so a refused
    # sweep kills nothing that belongs to the live one. Cells that would refuse to
    # append are caught here, under the lock, rather than as per-cell failures that
    # count toward --max_consecutive_failures.
    try:
        with sweep_locks(plan.status_path):
            refusing = [
                task.output_dir
                for task in plan.tasks
                if not executor.skip_reason(task) and prior_evidence(task.output_dir)
            ]
            if refusing:
                listed = "\n  ".join(str(path) for path in refusing)
                raise SystemExit(
                    "[run_matrix] Refusing to start: these run directories hold an "
                    f"earlier attempt's records:\n  {listed}\n"
                    "Preserve and move each aside, then rerun."
                )
            result = executor.execute()
    except LockHeldError as exc:
        raise SystemExit(f"[run_matrix] {exc}") from exc
    logger.info("Sweep status written to %s", plan.status_path)
    if result["abort_reason"] is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
