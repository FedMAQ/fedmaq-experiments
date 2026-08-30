"""Injectable execution lifecycle for a :class:`~scripts.matrix_planner.MatrixPlan`."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from scripts.common import is_run_complete, kill_ray_processes
from scripts.matrix_planner import MatrixPlan, MatrixTask

logger = logging.getLogger("fedmaq.run_matrix")

TIMEOUT_RETURNCODE = -128


class MatrixExecutor:
    """Run a planned matrix with observable, persisted lifecycle state."""

    def __init__(
        self,
        plan: MatrixPlan,
        *,
        start_at: int = 1,
        skip_completed: bool = False,
        max_consecutive_failures: int = 3,
        run_timeout_seconds: int = 0,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        cleanup: Callable[[], None] = kill_ray_processes,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        now: Callable[[], str] = lambda: datetime.now().isoformat(),
        hostname: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.plan = plan
        self.start_at = start_at
        self.skip_completed = skip_completed
        self.max_consecutive_failures = max_consecutive_failures
        self.run_timeout_seconds = run_timeout_seconds
        self.run = run
        self.cleanup = cleanup
        self.sleep = sleep
        self.clock = clock
        self.now = now
        self.hostname = hostname or socket.gethostname()
        self.environment = environment or os.environ.copy()

    def skip_reason(self, task: MatrixTask) -> str | None:
        if task.canonical_index < self.start_at:
            return f"--start_at {self.start_at}"
        if self.skip_completed and is_run_complete(task.output_dir):
            return "already complete"
        return None

    def _child_environment(self) -> dict[str, str]:
        child_env = self.environment.copy()
        child_env["FEDMAQ_SWEEP_HOST"] = self.hostname
        if self.plan.shard is not None:
            index, count = self.plan.shard
            child_env["FEDMAQ_SWEEP_SHARD_INDEX"] = str(index)
            child_env["FEDMAQ_SWEEP_SHARD_COUNT"] = str(count)
        else:
            child_env.pop("FEDMAQ_SWEEP_SHARD_INDEX", None)
            child_env.pop("FEDMAQ_SWEEP_SHARD_COUNT", None)
        return child_env

    def _initial_records(self) -> dict[int, dict[str, Any]]:
        return {
            task.canonical_index: {
                "index": task.canonical_index,
                "label": task.label,
                "output_dir": task.output_dir.as_posix(),
                "source_root": task.output_dir.resolve().as_posix(),
                "state": "pending",
                "host": None,
            }
            for task in self.plan.tasks
        }

    def execute(self) -> dict[str, Any]:
        """Execute tasks and return the final status payload.

        Status is written before the first task and after every task. Cleanup is
        also run in a ``finally`` block so an unexpected runner exception cannot
        leave Ray alive for the next invocation.
        """

        completed = failed = skipped = 0
        consecutive_failures = 0
        abort_reason: str | None = None
        failures: list[dict[str, Any]] = []
        records = self._initial_records()
        started_at = self.now()
        child_env = self._child_environment()

        def payload(state: str) -> dict[str, Any]:
            return {
                "schema_version": 2,
                "matrix": self.plan.matrix_path.as_posix(),
                "experiment_group": self.plan.experiment_group,
                "state": state,
                "host": self.hostname,
                "started_at": started_at,
                "updated_at": self.now(),
                "total_tasks": len(self.plan.canonical_tasks),
                "shard_tasks": len(self.plan.tasks),
                "shard": (
                    {
                        "index": self.plan.shard[0],
                        "count": self.plan.shard[1],
                        "canonical_indices": [task.canonical_index for task in self.plan.tasks],
                    }
                    if self.plan.shard is not None
                    else None
                ),
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
                "failed_indices": [failure["index"] for failure in failures],
                "failures": failures,
                "runs": list(records.values()),
                "abort_reason": abort_reason,
            }

        def save_status(state: str) -> dict[str, Any]:
            document = payload(state)
            try:
                self.plan.status_path.parent.mkdir(parents=True, exist_ok=True)
                self.plan.status_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
            except OSError as exc:
                logger.error("Could not write sweep status to %s: %s", self.plan.status_path, exc)
            return document

        save_status("running")
        try:
            for position, task in enumerate(self.plan.tasks, 1):
                reason = self.skip_reason(task)
                if reason:
                    logger.info(
                        "[%s/%s] Skipping run '%s' (%s)",
                        position,
                        len(self.plan.tasks),
                        task.label,
                        reason,
                    )
                    skipped += 1
                    records[task.canonical_index].update({"state": "skipped", "reason": reason})
                    save_status("running")
                    continue

                logger.info("[%s/%s] Starting: %s", position, len(self.plan.tasks), task.label)
                logger.info("Target Dir: %s", task.output_dir)
                logger.info("Command: %s", " ".join(task.command))
                # Ray actors outlive a timed-out driver, so clean before every
                # task and again after a timeout and after the whole sweep.
                self.cleanup()

                run_start = self.clock()
                timed_out = False
                try:
                    returncode = self.run(
                        task.command_list,
                        timeout=self.run_timeout_seconds or None,
                        env=child_env,
                    ).returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = TIMEOUT_RETURNCODE
                    logger.error(
                        "[TIMEOUT] Task [%s/%s] '%s' exceeded %ss and was killed",
                        position,
                        len(self.plan.tasks),
                        task.label,
                        self.run_timeout_seconds,
                    )
                    self.cleanup()
                elapsed = self.clock() - run_start

                if returncode != 0:
                    failed += 1
                    consecutive_failures += 1
                    failure = {
                        "index": task.canonical_index,
                        "label": task.label,
                        "returncode": returncode,
                        "timed_out": timed_out,
                        "output_dir": task.output_dir.as_posix(),
                        "elapsed_seconds": round(elapsed, 1),
                        "command": " ".join(task.command),
                        "host": self.hostname,
                    }
                    failures.append(failure)
                    records[task.canonical_index].update(
                        {
                            "state": "failed",
                            "returncode": returncode,
                            "elapsed_seconds": round(elapsed, 1),
                            "host": self.hostname,
                        }
                    )
                else:
                    completed += 1
                    consecutive_failures = 0
                    records[task.canonical_index].update(
                        {
                            "state": "completed",
                            "returncode": 0,
                            "elapsed_seconds": round(elapsed, 1),
                            "host": self.hostname,
                        }
                    )

                if (
                    self.max_consecutive_failures
                    and consecutive_failures >= self.max_consecutive_failures
                ):
                    abort_reason = (
                        f"{consecutive_failures} consecutive failures at task index "
                        f"{position} of {len(self.plan.tasks)} "
                        f"(threshold {self.max_consecutive_failures})"
                    )
                    save_status("aborted")
                    break

                save_status("running")
                # The pause gives Ray and the OS time to release resources before
                # the next isolated subprocess starts.
                self.sleep(3)
        finally:
            self.cleanup()

        state = "aborted" if abort_reason is not None else "finished"
        result = save_status(state)
        return result


__all__ = ["MatrixExecutor", "TIMEOUT_RETURNCODE"]
