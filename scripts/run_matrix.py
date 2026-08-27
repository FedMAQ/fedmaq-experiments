"""Declarative Matrix Sweep Runner for FedMAQ Experiments.

Reads an experiment matrix specification (YAML) from ``conf/matrix/`` and executes
the experiment grid using process isolation, cross-platform Ray cleanup, canonical pathing,
and optional resume support.

Usage:
    uv run python scripts/run_matrix.py --matrix conf/matrix/ci_test.yaml
    uv run python scripts/run_matrix.py --matrix ci_test --dry_run
    uv run python scripts/run_matrix.py --matrix pass2_explore --start_at 3
    uv run python scripts/run_matrix.py --matrix benchmark_grid --skip_completed
    uv run python scripts/run_matrix.py --matrix benchmark_grid --only fedavg
    uv run python scripts/run_matrix.py --matrix benchmark_grid \
        -o ray.temp_dir=/tmp/ray-cjb -o ray.object_store_gb=4 \
        --run_timeout_seconds 5400
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from scripts.common import (
    SWEEP_STATUS_FILENAME,
    build_run_command,
    expand_matrix,
    get_canonical_output_dir,
    get_sweep_group_dir,
    is_run_complete,
    kill_ray_processes,
    parse_shard,
    partition_tasks,
    sharded_sweep_status_filename,
    validate_unique_output_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedmaq.run_matrix")

# Recorded in sweep_status.json for a run killed by --run_timeout_seconds. Distinct
# from any exit code scripts/run.py itself produces, so a hang is separable from a
# crash when reading the file back. Mirrors the shell's 128+SIGALRM convention.
TIMEOUT_RETURNCODE = -128


def resolve_matrix_path(matrix_arg: str) -> Path:
    """Resolve matrix file path from argument string."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Declarative Matrix Sweep Runner for FedMAQ Experiments"
    )
    parser.add_argument(
        "--matrix",
        type=str,
        required=True,
        help=(
            "Path or name of the matrix YAML file "
            "(e.g. 'ci_test' or 'conf/matrix/mobilenetv2_smoke_50r.yaml')"
        ),
    )
    parser.add_argument(
        "--start_at",
        type=int,
        default=1,
        help="1-indexed run number to resume execution from (default: 1)",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help=(
            "Skip any run whose output directory already holds a final-round "
            "checkpoint. Use to fill gaps after a partial sweep, instead of "
            "computing a --start_at index by hand."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Display planned execution grid without running commands",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="I/N",
        help=(
            "Dispatch only deterministic shard I of N over the canonical matrix "
            "run list (for example, 2/4); completion state is not consulted"
        ),
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help=(
            "Extra Hydra override applied to every run of the sweep; repeatable. For "
            "host settings that belong to the machine rather than the matrix, chiefly "
            "ray.temp_dir and ray.object_store_gb. The alternative is hand-editing "
            "conf/config.yaml on the host and carrying that diff for the length of a "
            "multi-day grid, where a single git pull silently reverts it partway "
            "through. Applied before each run's own overrides, so a matrix file's "
            "deliberate per-run setting always wins."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="LABEL",
        dest="only_labels",
        help=(
            "Dispatch only the matrix rows carrying these labels; repeatable. The "
            "rows keep the matrix's own phase, experiment_group, rounds and "
            "overrides, so they land in exactly the directories the full sweep "
            "would have written and a later full dispatch with --skip_completed "
            "passes over them. This exists for one pre-registered case: the "
            "formulation study's accuracy floor is defined against the benchmark "
            "grid's uncompressed FedAvg rows, which must therefore be dispatched "
            "before the study that the rest of the grid waits on (docs/agents/execution-model.md "
            "Stage 1c). Do not use it to run a grid piecemeal for convenience."
        ),
    )
    parser.add_argument(
        "--max_consecutive_failures",
        type=int,
        default=3,
        help=(
            "Abort the sweep after this many consecutive run failures (default: 3; "
            "0 disables). A systemic condition (a co-tenant VRAM spike, a leaked Ray "
            "actor) fails one run and then every run after it, and a detached sweep "
            "has no terminal to notice from."
        ),
    )
    parser.add_argument(
        "--run_timeout_seconds",
        type=int,
        default=0,
        help=(
            "Kill any single run exceeding this many seconds and record it as a "
            "failure (default: 0, no timeout). Guards against hangs, not slowness: "
            "Ray can deadlock waiting on an actor that never starts, which is the "
            "documented low-system-RAM failure mode and leaves the sweep frozen "
            "forever. Set generously from a measured run time."
        ),
    )
    args = parser.parse_args()

    if args.max_consecutive_failures < 0:
        parser.error("--max_consecutive_failures must be >= 0")
    if args.run_timeout_seconds < 0:
        parser.error("--run_timeout_seconds must be >= 0")

    try:
        shard = parse_shard(args.shard) if args.shard else None
    except ValueError as exc:
        parser.error(str(exc))
    if shard is not None and args.only_labels:
        parser.error("--only cannot be combined with --shard; shard the full matrix")

    matrix_path = resolve_matrix_path(args.matrix)
    logger.info(f"Loading experiment matrix from {matrix_path}")
    cfg = OmegaConf.load(matrix_path)

    phase = cfg.get("phase", "smoke")
    exp_group = cfg.get("experiment_group", matrix_path.stem)
    dataset = cfg.get("dataset", "cifar10")
    model = cfg.get("model", "mobilenetv2")
    total_rounds = int(cfg.get("total_rounds", 50))
    client_gpus = float(cfg.get("client_gpus", 1.0))
    experiment = cfg.get("experiment", None)
    matrix = OmegaConf.to_container(cfg, resolve=True)
    seeds = [int(s) for s in matrix.get("seeds", [0])]
    heterogeneities = list(matrix.get("heterogeneities", ["dirichlet_alpha_0.1"]))
    runs_spec = matrix.get("runs", []) or []

    if args.only_labels:
        available = [str(r.get("label", r.get("alg"))) for r in runs_spec]
        unknown = [label for label in args.only_labels if label not in available]
        if unknown:
            # Hard error, never a silent empty sweep. A mistyped --only on a
            # shared allocation otherwise reports "Total Runs Scheduled: 0" and
            # exits successfully, which reads as a completed stage.
            parser.error(
                f"--only label(s) {unknown} not in {matrix_path.name}; "
                f"available labels: {available}"
            )
        runs_spec = [r for r in runs_spec if str(r.get("label", r.get("alg"))) in args.only_labels]
        matrix["runs"] = runs_spec

    # A run may declare its own ``seeds:``, overriding the matrix-level list. This
    # exists so a single cell can be measured at more seeds than the rest of its
    # matrix: the exploration factorial's unrefined reference cell defines the
    # sigma that every keep-or-drop call is judged against (§4.3.1), and sigma
    # estimated from three seeds carries roughly +/-50% of itself. Deepening only
    # that one cell buys the precision for two extra runs instead of sixteen.
    def seeds_for(run_item) -> list[int]:
        return [int(s) for s in run_item.get("seeds", seeds)]

    canonical_tasks = []
    for spec in expand_matrix(matrix, matrix_path.stem):
        output_dir = get_canonical_output_dir(
            phase=spec["phase"],
            dataset=spec["dataset"],
            model=spec["model"],
            exp_group=spec["experiment_group"],
            algorithm=spec["algorithm_config"],
            heterogeneity=spec["heterogeneity"],
            seed=spec["seed"],
            variant=spec["variant"],
        )
        # Sweep-wide host overrides first, so a matrix file's own per-run override
        # wins any collision. Several of those are load-bearing
        # (``algorithm.post_process`` fixes which regime a run is comparable in),
        # and a hand-typed flag must not be able to silently displace one.
        overrides = list(args.overrides) + spec["overrides"]

        cmd = build_run_command(
            dataset=spec["dataset"],
            heterogeneity=spec["heterogeneity"],
            algorithm=spec["algorithm_config"],
            total_rounds=total_rounds,
            seed=spec["seed"],
            client_gpus=client_gpus,
            target_dir=output_dir,
            overrides=overrides,
            experiment=experiment,
        )

        canonical_tasks.append(
            {
                "canonical_index": spec["canonical_index"],
                "label": f"{spec['label']}-{spec['heterogeneity']}-seed{spec['seed']}",
                "alg": spec["algorithm_config"],
                "het": spec["heterogeneity"],
                "seed": spec["seed"],
                "output_dir": output_dir,
                "cmd": cmd,
            }
        )

    validate_unique_output_dirs(canonical_tasks)
    tasks = (
        partition_tasks(canonical_tasks, *shard)
        if shard is not None
        else canonical_tasks
    )
    host = socket.gethostname()

    print("=" * 70)
    print(f"FedMAQ Matrix Sweep: {exp_group.upper()}")
    print(f"Phase: {phase} | Dataset: {dataset} | Model: {model}")
    if experiment:
        print(f"Experiment group: {experiment}")
    print(f"Total Rounds: {total_rounds} | Client GPUs: {client_gpus}")
    print(f"Heterogeneities: {heterogeneities}")
    print(f"Seeds: {seeds}")
    for run_item in runs_spec:
        extra = [s for s in seeds_for(run_item) if s not in seeds]
        if extra:
            print(f"  + {run_item.get('label', run_item.get('alg'))}: deepened with {extra}")
    if args.only_labels:
        print(f"Label filter (--only): {args.only_labels}")
    print(f"Canonical Runs: {len(canonical_tasks)}")
    print(f"Total Runs Scheduled: {len(tasks)}")
    if shard is not None:
        print(f"Shard: {shard[0]}/{shard[1]} (host: {host})")
    if args.overrides:
        print(f"Sweep-wide overrides: {' '.join(args.overrides)}")
    if args.start_at > 1:
        print(f"Resuming from Run Index: {args.start_at}")
    if args.skip_completed:
        print("Skipping runs that already hold a final-round checkpoint")
    print("=" * 70)

    def skip_reason(idx: int, task: dict) -> str | None:
        """Why this task would not run, or ``None`` if it would."""
        if task["canonical_index"] < args.start_at:
            return f"--start_at {args.start_at}"
        if args.skip_completed and is_run_complete(task["output_dir"]):
            return "already complete"
        return None

    if args.dry_run:
        print("\n[DRY RUN MODE] The following commands would be executed:")
        for idx, task in enumerate(tasks, 1):
            reason = skip_reason(idx, task)
            skip_mark = f" (SKIPPED: {reason})" if reason else ""
            print(
                f"\nTask {task['canonical_index']}/{len(canonical_tasks)} "
                f"(shard position {idx}/{len(tasks)}) [{task['label']}]{skip_mark}"
            )
            print(f" Target Dir: {task['output_dir']}")
            print(f" Command:    {' '.join(task['cmd'])}")
        print("\nDry run completed successfully.")
        sys.exit(0)

    completed = 0
    failed = 0
    skipped = 0
    consecutive_failures = 0
    abort_reason: str | None = None
    failures: list[dict] = []
    start_time = time.time()
    started_at = datetime.now().isoformat()
    status_filename = (
        sharded_sweep_status_filename(*shard) if shard is not None else SWEEP_STATUS_FILENAME
    )
    status_path = get_sweep_group_dir(phase, dataset, model, exp_group) / status_filename
    run_records = {
        task["canonical_index"]: {
            "index": task["canonical_index"],
            "label": task["label"],
            "output_dir": task["output_dir"].as_posix(),
            "source_root": task["output_dir"].resolve().as_posix(),
            "state": "pending",
            "host": None,
        }
        for task in tasks
    }
    child_env = os.environ.copy()
    child_env["FEDMAQ_SWEEP_HOST"] = host
    if shard is not None:
        child_env["FEDMAQ_SWEEP_SHARD_INDEX"] = str(shard[0])
        child_env["FEDMAQ_SWEEP_SHARD_COUNT"] = str(shard[1])
    else:
        child_env.pop("FEDMAQ_SWEEP_SHARD_INDEX", None)
        child_env.pop("FEDMAQ_SWEEP_SHARD_COUNT", None)

    def save_status(state: str) -> None:
        """Persist which task indices failed, after every task rather than at the end.

        A multi-day unattended sweep can end without reaching its summary: a
        dropped connection, an idle-culled kernel, an evicted allocation on a
        shared host. Without this the only record of *which* index failed is the
        log stream, and because ``--start_at`` is positional, recovering from a
        failure at index 57 of 183 means arithmetic against that log. Written on
        each iteration so the file is useful precisely when the sweep did not
        finish.

        Scoped to one invocation: the file is replaced, not appended, so a
        gap-filling re-run reflects that re-run's outcome rather than
        accumulating stale failures from the sweep it is repairing.

        ``state`` is one of ``running``, ``finished``, or ``aborted``. An aborted
        sweep is never recorded as finished: a detached sweep has to be able to
        explain itself from this file alone, and ``abort_reason`` is where it does.
        """
        payload = {
            "schema_version": 2,
            "matrix": matrix_path.as_posix(),
            "experiment_group": exp_group,
            "state": state,
            "host": host,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "total_tasks": len(canonical_tasks),
            "shard_tasks": len(tasks),
            "shard": (
                {
                    "index": shard[0],
                    "count": shard[1],
                    "canonical_indices": [task["canonical_index"] for task in tasks],
                }
                if shard is not None
                else None
            ),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "failed_indices": [f["index"] for f in failures],
            "failures": failures,
            "runs": list(run_records.values()),
            "abort_reason": abort_reason,
        }
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            # Never lose a sweep to its own bookkeeping.
            logger.error(f"Could not write sweep status to {status_path}: {exc}")

    save_status("running")

    for idx, task in enumerate(tasks, 1):
        canonical_index = task["canonical_index"]
        reason = skip_reason(idx, task)
        if reason:
            logger.info(f"[{idx}/{len(tasks)}] Skipping run '{task['label']}' ({reason})")
            skipped += 1
            run_records[canonical_index].update({"state": "skipped", "reason": reason})
            continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{idx}/{len(tasks)}] Starting: {task['label']}")
        logger.info(f"Target Dir: {task['output_dir']}")
        logger.info(f"Command: {' '.join(task['cmd'])}")
        logger.info(f"{'=' * 70}")

        kill_ray_processes()

        run_start = time.time()
        timed_out = False
        try:
            returncode = subprocess.run(
                task["cmd"], timeout=args.run_timeout_seconds or None, env=child_env
            ).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = TIMEOUT_RETURNCODE
            logger.error(
                f"[TIMEOUT] Task [{idx}/{len(tasks)}] '{task['label']}' exceeded "
                f"{args.run_timeout_seconds}s and was killed"
            )
            # subprocess.run kills the direct child on timeout, but Ray's raylet and
            # actors are grandchildren and survive it, still holding VRAM. Reap them
            # now rather than at the top of the next iteration, so a timeout that
            # ends the sweep does not leave the cluster running.
            kill_ray_processes()
        elapsed = time.time() - run_start

        if returncode != 0:
            if not timed_out:
                logger.error(
                    f"[FAILED] Task [{idx}/{len(tasks)}] '{task['label']}' "
                    f"exited with code {returncode} ({elapsed:.1f}s)"
                )
            failed += 1
            consecutive_failures += 1
            failures.append(
                {
                    "index": canonical_index,
                    "label": task["label"],
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "output_dir": task["output_dir"].as_posix(),
                    "elapsed_seconds": round(elapsed, 1),
                    "command": " ".join(task["cmd"]),
                    "host": host,
                }
            )
            run_records[canonical_index].update(
                {
                    "state": "failed",
                    "returncode": returncode,
                    "elapsed_seconds": round(elapsed, 1),
                    "host": host,
                }
            )
        else:
            logger.info(
                f"[SUCCESS] Task [{idx}/{len(tasks)}] '{task['label']}' completed in {elapsed:.1f}s"
            )
            completed += 1
            consecutive_failures = 0
            run_records[canonical_index].update(
                {
                    "state": "completed",
                    "returncode": 0,
                    "elapsed_seconds": round(elapsed, 1),
                    "host": host,
                }
            )

        if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
            abort_reason = (
                f"{consecutive_failures} consecutive failures at task index {idx} "
                f"of {len(tasks)} (threshold {args.max_consecutive_failures})"
            )
            logger.error(f"\n{'=' * 70}")
            logger.error(f"[ABORTED] {abort_reason}")
            logger.error(
                "Consecutive failures indicate a systemic condition rather than a bad "
                "run. Diagnose, then re-run this matrix with --skip_completed to "
                "resume from the last finished run."
            )
            logger.error(f"{'=' * 70}")
            save_status("aborted")
            break

        save_status("running")
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
    if failures:
        logger.error(
            f"  Failed indices: {[f['index'] for f in failures]} "
            f"-- re-run with --skip_completed to fill the gaps"
        )
    if abort_reason is None:
        save_status("finished")
    else:
        # The loop already wrote "aborted". Re-save so the final counters land, but
        # never as "finished": an aborted sweep left runs undispatched, and reading
        # the file is the only way a detached sweep reports that.
        save_status("aborted")
    logger.info(f"Sweep status written to {status_path}")
    logger.info("=" * 70)
    if abort_reason is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
