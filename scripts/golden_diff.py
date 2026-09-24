"""Capture and classify assurance outputs for the replacement pipeline.

The old-output comparison is a transition diagnostic: it records what changed
between two candidates, but it is never a release gate.  The release gate is
the repeatability check, which compares two independent captures of the same
clean candidate and requires matching provenance as well as bit-exact output.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import build_run_command, kill_ray_processes
from scripts.run_guard import LockHeldError, ray_cleanup_lock

GOLDEN_SET: list[str] = [
    "fedavg",
    "fedprox",
    "fedpaq",
    "fedavg_kd",
    "dadaquant",
    "fedmaq",
    "fedkd",
    "feddistill",
    "cfd",
]

IGNORED_COLUMNS = {"system/wall_time_sec", "system/cumulative_wall_time_sec"}
GOLDEN_ROOT = Path("outputs/golden/step2")
TRANSITION_ROOT = Path("outputs/golden/step2_transition")
REPEATABILITY_ROOT = Path("outputs/golden/step2_repeatability")
SEED = 42
SSD = "dirichlet_alpha_0.1"
PERSISTENCE_DIR = Path(".data_partitions/fedmd_models")


def _run(algorithm: str, target_dir: Path, *, client_gpus: float = 1.0) -> None:
    if PERSISTENCE_DIR.exists():
        shutil.rmtree(PERSISTENCE_DIR)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    kill_ray_processes()
    cmd = build_run_command(
        dataset="cifar10",
        heterogeneity=SSD,
        algorithm=algorithm,
        total_rounds=2,
        seed=SEED,
        client_gpus=client_gpus,
        target_dir=target_dir,
        overrides=["experiment=ci", "protocol_stage=assurance", "split=none"],
    )
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _semantic_csv_bytes(path: Path) -> bytes:
    """Canonicalize only the explicitly non-semantic wall-clock columns."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [column for column in reader.fieldnames or [] if column not in IGNORED_COLUMNS]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in reader)
    return output.getvalue().encode("utf-8")


def _diff(
    expected_rows: list[dict[str, str]], actual_rows: list[dict[str, str]], label: str
) -> list[str]:
    """Return human-readable semantic differences, excluding wall-clock fields."""
    problems: list[str] = []
    if len(expected_rows) != len(actual_rows):
        problems.append(
            f"[{label}] row count differs: expected={len(expected_rows)} actual={len(actual_rows)}"
        )
        return problems

    expected_cols = set(expected_rows[0].keys()) if expected_rows else set()
    actual_cols = set(actual_rows[0].keys()) if actual_rows else set()
    if expected_cols != actual_cols:
        problems.append(
            f"[{label}] column set differs: only-in-expected={expected_cols - actual_cols} "
            f"only-in-actual={actual_cols - expected_cols}"
        )

    for index, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=False)):
        for column in expected_cols & actual_cols:
            if column in IGNORED_COLUMNS:
                continue
            if expected.get(column) != actual.get(column):
                problems.append(
                    f"[{label}] row {index} column {column!r}: "
                    f"expected={expected.get(column)!r} actual={actual.get(column)!r}"
                )
    return problems


def _classified_diff(
    expected_rows: list[dict[str, str]], actual_rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Classify output changes so a transition cannot masquerade as a pass."""
    result: dict[str, Any] = {
        "row_count": {"expected": len(expected_rows), "actual": len(actual_rows)},
        "column_added": [],
        "column_removed": [],
        "semantic_value_changes": [],
        "ignored_runtime_changes": [],
    }
    expected_cols = set(expected_rows[0].keys()) if expected_rows else set()
    actual_cols = set(actual_rows[0].keys()) if actual_rows else set()
    result["column_added"] = sorted(actual_cols - expected_cols)
    result["column_removed"] = sorted(expected_cols - actual_cols)
    for index, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=False)):
        for column in sorted(expected_cols & actual_cols):
            if expected.get(column) == actual.get(column):
                continue
            change = {
                "row": index,
                "column": column,
                "expected": expected.get(column),
                "actual": actual.get(column),
            }
            if column in IGNORED_COLUMNS:
                result["ignored_runtime_changes"].append(change)
            else:
                result["semantic_value_changes"].append(change)
    return result


def _metadata(directory: Path) -> dict[str, Any]:
    path = directory / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing provenance manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid provenance manifest: {path}")
    git = value.get("git")
    environment = value.get("environment")
    if not isinstance(git, Mapping) or not isinstance(environment, Mapping):
        raise ValueError(f"manifest lacks git/environment records: {path}")
    return {
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "config_sha256": value.get("config_sha256"),
        "environment": dict(environment),
    }


def _capture(directory: Path) -> dict[str, Any]:
    csv_path = directory / "experiment_log.csv"
    return {"metadata": _metadata(directory), "rows": _read_csv(csv_path)}


def transition_diagnostic(old_directory: Path, new_directory: Path, label: str) -> dict[str, Any]:
    """Record a classified old-to-new differential; this result is not pass-eligible."""
    old = _capture(old_directory)
    new = _capture(new_directory)
    return {
        "schema_version": 1,
        "operation": "old_to_new_transition_diagnostic",
        "candidate": label,
        "ledger": "assurance",
        "status": "RECORDED",
        "pass": False,
        "pass_eligible": False,
        "old": old["metadata"],
        "new": new["metadata"],
        "old_output": str(old_directory),
        "new_output": str(new_directory),
        "classified_differential": _classified_diff(old["rows"], new["rows"]),
    }


def repeatability_report(
    first_directory: Path, second_directory: Path, label: str
) -> dict[str, Any]:
    """Gate two independent captures of one clean candidate on exact equality."""
    if first_directory.resolve() == second_directory.resolve():
        return {
            "schema_version": 1,
            "operation": "same_candidate_repeatability",
            "candidate": label,
            "ledger": "assurance",
            "status": "FAIL",
            "pass": False,
            "reasons": ["repeatability captures must use independent directories"],
        }
    first = _capture(first_directory)
    second = _capture(second_directory)
    first_meta = first["metadata"]
    second_meta = second["metadata"]
    reasons: list[str] = []
    if first_meta["dirty"] is not False or second_meta["dirty"] is not False:
        reasons.append("both captures must record a clean working tree")
    for field in ("commit", "config_sha256"):
        if not first_meta[field] or first_meta[field] != second_meta[field]:
            reasons.append(f"{field} differs or is missing")
    if first_meta["environment"] != second_meta["environment"]:
        reasons.append("environment records differ")
    differential = _classified_diff(first["rows"], second["rows"])
    if (
        differential["row_count"]["expected"] != differential["row_count"]["actual"]
        or differential["column_added"]
        or differential["column_removed"]
        or differential["semantic_value_changes"]
    ):
        reasons.append("semantic output is not bit-exact")
    if _semantic_csv_bytes(first_directory / "experiment_log.csv") != _semantic_csv_bytes(
        second_directory / "experiment_log.csv"
    ):
        reasons.append("canonical output bytes differ")
    return {
        "schema_version": 1,
        "operation": "same_candidate_repeatability",
        "candidate": label,
        "ledger": "assurance",
        "status": "PASS" if not reasons else "FAIL",
        "pass": not reasons,
        "reasons": reasons,
        "first": first_meta,
        "second": second_meta,
        "first_output": str(first_directory),
        "second_output": str(second_directory),
        "classified_differential": differential,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def capture() -> None:
    for algorithm in GOLDEN_SET:
        _run(algorithm, GOLDEN_ROOT / algorithm)
    print(f"\nGolden output captured under {GOLDEN_ROOT}/")


def transition() -> None:
    for algorithm in GOLDEN_SET:
        old_directory = GOLDEN_ROOT / algorithm
        if not old_directory.exists():
            raise FileNotFoundError(f"run capture first: {old_directory}")
        new_directory = TRANSITION_ROOT / algorithm
        _run(algorithm, new_directory)
        report = transition_diagnostic(old_directory, new_directory, algorithm)
        _write_report(TRANSITION_ROOT / f"{algorithm}.json", report)
        print(f"[{algorithm}] transition diagnostic recorded")


def repeatability() -> None:
    for algorithm in GOLDEN_SET:
        first_directory = REPEATABILITY_ROOT / algorithm / "capture_a"
        second_directory = REPEATABILITY_ROOT / algorithm / "capture_b"
        _run(algorithm, first_directory, client_gpus=0.5)
        _run(algorithm, second_directory, client_gpus=0.5)
        report = repeatability_report(first_directory, second_directory, algorithm)
        _write_report(REPEATABILITY_ROOT / f"{algorithm}.json", report)
        if not report["pass"]:
            print(f"[{algorithm}] repeatability failed: {report['reasons']}")
            raise SystemExit(1)
        print(f"[{algorithm}] PASS — independent captures are bit-exact")


def main() -> None:
    operations = {
        "capture": capture,
        "compare": transition,
        "transition": transition,
        "repeatability": repeatability,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        print(f"Usage: {Path(sys.argv[0]).name} [{', '.join(operations)}]")
        raise SystemExit(1)
    try:
        with ray_cleanup_lock(f"golden_diff {sys.argv[1]}"):
            operations[sys.argv[1]]()
    except LockHeldError as exc:
        raise SystemExit(f"[golden_diff] {exc}") from exc


if __name__ == "__main__":
    main()
