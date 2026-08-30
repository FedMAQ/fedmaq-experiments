"""Validate and merge per-host matrix sweep status files.

Sharded runners deliberately write separate status files because independent hosts
cannot safely read/modify/write one JSON file on a shared filesystem. This command
is the explicit reconciliation step that produces the familiar ``sweep_status.json``
after the host-local files have been collected in one sweep directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.common import SWEEP_STATUS_FILENAME


def _status_paths(group_dir: Path) -> list[Path]:
    return sorted(group_dir.glob("sweep_status.shard-*-of-*.json"))


def merge_statuses(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge validated shard payloads, rejecting overlap, gaps, and drift."""
    if not statuses:
        raise ValueError("no shard status files supplied")

    first = statuses[0]
    shared_fields = ("matrix", "experiment_group", "total_tasks")
    for status in statuses[1:]:
        for field in shared_fields:
            if status.get(field) != first.get(field):
                raise ValueError(f"shard status mismatch for {field!r}")

    total_tasks = int(first["total_tasks"])
    expected_shard_count: int | None = None
    by_index: dict[int, dict[str, Any]] = {}
    failures: dict[int, dict[str, Any]] = {}
    shard_records = []

    for status in statuses:
        shard = status.get("shard")
        if not isinstance(shard, dict):
            raise ValueError("every merged status must identify its shard")
        index = int(shard["index"])
        count = int(shard["count"])
        if expected_shard_count is None:
            expected_shard_count = count
        elif count != expected_shard_count:
            raise ValueError("shard status files disagree about shard count")
        if not 1 <= index <= count:
            raise ValueError(f"invalid shard selector {index}/{count}")
        if any(record["index"] == index for record in shard_records):
            raise ValueError(f"duplicate shard status for {index}/{count}")

        declared_indices = [int(value) for value in shard.get("canonical_indices", [])]
        expected_indices_for_shard = [
            run_index
            for run_index in range(1, total_tasks + 1)
            if (run_index - 1) % count == index - 1
        ]
        if declared_indices != expected_indices_for_shard:
            raise ValueError(f"shard {index}/{count} does not match canonical partition")
        records = status.get("runs", [])
        record_indices = [int(record["index"]) for record in records]
        if record_indices != declared_indices:
            raise ValueError(f"shard {index}/{count} run records do not match membership")
        for record in records:
            run_index = int(record["index"])
            if run_index in by_index:
                raise ValueError(f"run index {run_index} appears in multiple shards")
            if not 1 <= run_index <= total_tasks:
                raise ValueError(f"run index {run_index} is outside 1..{total_tasks}")
            by_index[run_index] = record
        for failure in status.get("failures", []):
            failure_index = int(failure["index"])
            if failure_index in failures:
                raise ValueError(f"failure index {failure_index} appears more than once")
            failures[failure_index] = failure
        shard_records.append(
            {
                "index": index,
                "count": count,
                "host": status.get("host"),
                "state": status.get("state"),
                "status_updated_at": status.get("updated_at"),
                "canonical_indices": declared_indices,
            }
        )

    expected_indices = set(range(1, total_tasks + 1))
    actual_indices = set(by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(f"shard union is not the matrix: missing={missing}, extra={extra}")
    if expected_shard_count is not None and {item["index"] for item in shard_records} != set(
        range(1, expected_shard_count + 1)
    ):
        raise ValueError("shard union is incomplete: one or more shard files are missing")

    runs = [by_index[index] for index in sorted(by_index)]
    counts = {
        state: sum(record.get("state") == state for record in runs)
        for state in (
            "completed",
            "failed",
            "skipped",
            "pending",
        )
    }
    if counts["pending"]:
        state = "running"
    elif any(item["state"] == "aborted" for item in shard_records):
        state = "aborted"
    else:
        state = "finished"

    return {
        "schema_version": 2,
        "matrix": first["matrix"],
        "experiment_group": first["experiment_group"],
        "state": state,
        "started_at": min(status.get("started_at", "") for status in statuses),
        "updated_at": max(status.get("updated_at", "") for status in statuses),
        "total_tasks": total_tasks,
        "shard_tasks": total_tasks,
        "shard": None,
        "completed": counts["completed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "failed_indices": sorted(failures),
        "failures": [failures[index] for index in sorted(failures)],
        "runs": runs,
        "hosts": sorted({item["host"] for item in shard_records if item["host"]}),
        "shards": sorted(shard_records, key=lambda item: item["index"]),
        "abort_reason": next(
            (status.get("abort_reason") for status in statuses if status.get("abort_reason")),
            None,
        ),
    }


def merge_group_statuses(group_dir: Path, output: Path | None = None) -> Path:
    """Merge all shard status files in ``group_dir`` into one aggregate file."""
    paths = _status_paths(group_dir)
    if not paths:
        raise ValueError(f"no sharded status files found in {group_dir}")
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    merged = merge_statuses(statuses)
    output = output or group_dir / SWEEP_STATUS_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-dir",
        type=Path,
        required=True,
        help="Sweep group directory containing sweep_status.shard-I-of-N.json files",
    )
    parser.add_argument(
        "--output", type=Path, help="Aggregate path (default: group/sweep_status.json)"
    )
    args = parser.parse_args()
    try:
        output = merge_group_statuses(args.group_dir, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Merged sweep status: {output}")


if __name__ == "__main__":
    main()
