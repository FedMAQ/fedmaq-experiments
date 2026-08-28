"""Audit Study 1 artifacts without importing the experiment environment.

The scanner is intentionally standard-library-only so it can run on the
JupyterHub allocation without installing packages. It writes one JSON report
and never modifies the experiment output tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STUDY1_GROUPS = (
    "benchmark_grid",
    "formulation_study",
    "ablation",
    "uniform_memory_control",
)
RUN_MARKERS = {
    "experiment_log.csv",
    "experiment_log.jsonl",
    "run_manifest.json",
    "final_global_model.pt",
}
# Deliberately independent: this auditor is the required-column schema tripwire.
REQUIRED_COLUMNS = {
    "round",
    "test/loss",
    "test/accuracy",
    "communication/round_bytes",
    "communication/cumulative_bytes",
    "communication/cumulative_mb",
    "system/round_time_sec",
    "system/cumulative_time_sec",
    "system/client_sim_time_sec",
    "system/cumulative_client_time_sec",
    "system/server_sim_time_sec",
    "system/cumulative_server_time_sec",
    "system/wall_time_sec",
    "system/cumulative_wall_time_sec",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def identity_key(
    dataset: str,
    experiment_group: str,
    algorithm_config: str,
    variant: str,
    alpha: float,
    formulation: int | None,
    seed: int,
) -> str:
    form = "none" if formulation is None else str(int(formulation))
    return (
        f"{dataset}|{experiment_group}|{algorithm_config}|{variant}|"
        f"a{float(alpha)!r}|f{form}|s{int(seed)}"
    )


def relative_text(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def canonical_path_fields(job_dir: Path, repo_root: Path) -> dict[str, Any]:
    try:
        parts = job_dir.resolve().relative_to(repo_root.resolve()).parts
    except ValueError:
        return {"canonical": False}
    if len(parts) != 7 or parts[0] != "outputs" or not parts[6].startswith("seed_"):
        return {"canonical": False, "parts": list(parts)}
    algorithm_segment = parts[4]
    algorithm_path, separator, variant = algorithm_segment.partition("__")
    try:
        path_seed = int(parts[6].removeprefix("seed_"))
    except ValueError:
        path_seed = None
    return {
        "canonical": True,
        "phase": parts[1],
        "dataset_model": parts[2],
        "experiment_group": parts[3],
        "algorithm_path": algorithm_path,
        "algorithm_segment": algorithm_segment,
        "variant": variant if separator else "",
        "heterogeneity_path": parts[5],
        "path_seed": path_seed,
    }


def inventory_outputs(outputs_root: Path, repo_root: Path) -> dict[str, Any]:
    by_name: Counter[str] = Counter()
    bytes_by_name: Counter[str] = Counter()
    by_suffix: Counter[str] = Counter()
    bytes_by_suffix: Counter[str] = Counter()
    run_dirs: set[Path] = set()
    sweep_statuses: list[Path] = []
    largest: list[tuple[int, str]] = []
    total_bytes = 0
    total_files = 0

    if not outputs_root.is_dir():
        return {
            "exists": False,
            "total_files": 0,
            "total_bytes": 0,
            "by_name": {},
            "by_suffix": {},
            "largest_files": [],
            "run_dirs": [],
            "sweep_statuses": [],
        }

    for dirpath, _, filenames in os.walk(outputs_root):
        directory = Path(dirpath)
        names = set(filenames)
        if names & RUN_MARKERS:
            run_dirs.add(directory)
        if directory.name == ".hydra" and "config.yaml" in names:
            run_dirs.add(directory.parent)
        if "sweep_status.json" in names:
            sweep_statuses.append(directory / "sweep_status.json")

        for filename in filenames:
            path = directory / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            total_files += 1
            total_bytes += size
            by_name[filename] += 1
            bytes_by_name[filename] += size
            suffix = path.suffix.lower() or "<none>"
            by_suffix[suffix] += 1
            bytes_by_suffix[suffix] += size
            item = (size, relative_text(path, repo_root))
            if len(largest) < 50:
                heapq.heappush(largest, item)
            elif item > largest[0]:
                heapq.heapreplace(largest, item)

    return {
        "exists": True,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "by_name": {
            name: {"count": by_name[name], "bytes": bytes_by_name[name]} for name in sorted(by_name)
        },
        "by_suffix": {
            suffix: {"count": by_suffix[suffix], "bytes": bytes_by_suffix[suffix]}
            for suffix in sorted(by_suffix)
        },
        "largest_files": [
            {"path": path, "bytes": size} for size, path in sorted(largest, reverse=True)
        ],
        "run_dirs": sorted(run_dirs),
        "sweep_statuses": sorted(sweep_statuses),
    }


def scan_csv(path: Path, expected_round: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
        "valid": False,
    }
    if not path.is_file():
        return result

    rounds: list[int] = []
    round_bytes: list[int] = []
    cumulative_bytes: list[int] = []
    cumulative_mb: list[float] = []
    numeric_rows: list[dict[str, float]] = []
    nonfinite_rows: list[int] = []
    parse_errors: list[str] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            result["columns"] = sorted(columns)
            result["missing_columns"] = sorted(REQUIRED_COLUMNS - columns)
            for row_number, row in enumerate(reader, start=2):
                try:
                    round_value = int(float(row["round"]))
                    loss = float(row["test/loss"])
                    accuracy = float(row["test/accuracy"])
                    round_byte_value = int(float(row["communication/round_bytes"]))
                    cumulative_byte_value = int(float(row["communication/cumulative_bytes"]))
                    cumulative = float(row["communication/cumulative_mb"])
                    rounds.append(round_value)
                    round_bytes.append(round_byte_value)
                    cumulative_bytes.append(cumulative_byte_value)
                    cumulative_mb.append(cumulative)
                    numeric_rows.append(
                        {
                            "round": float(round_value),
                            "test/loss": loss,
                            "test/accuracy": accuracy,
                            "communication/round_bytes": float(round_byte_value),
                            "communication/cumulative_bytes": float(cumulative_byte_value),
                            "communication/cumulative_mb": cumulative,
                        }
                    )
                    if not all(math.isfinite(value) for value in (loss, accuracy, cumulative)):
                        nonfinite_rows.append(row_number)
                except (KeyError, TypeError, ValueError) as exc:
                    parse_errors.append(f"row {row_number}: {exc}")
    except (OSError, csv.Error, UnicodeError) as exc:
        result["read_error"] = str(exc)
        return result

    duplicate_rounds = sorted(
        round_value for round_value, count in Counter(rounds).items() if count > 1
    )
    order_violations = [
        {"previous": previous, "current": current}
        for previous, current in zip(rounds, rounds[1:], strict=False)
        if current <= previous
    ]
    cumulative_decreases = [
        {"round": rounds[index], "previous_mb": cumulative_mb[index - 1], "current_mb": value}
        for index, value in enumerate(cumulative_mb[1:], start=1)
        if value + 1e-9 < cumulative_mb[index - 1]
    ]
    byte_recurrence_violations = [
        {
            "round": rounds[index],
            "previous_cumulative_bytes": cumulative_bytes[index - 1],
            "round_bytes": round_bytes[index],
            "current_cumulative_bytes": value,
        }
        for index, value in enumerate(cumulative_bytes[1:], start=1)
        if value != cumulative_bytes[index - 1] + round_bytes[index]
    ]
    mb_conversion_violations = [
        {
            "round": rounds[index],
            "cumulative_bytes": cumulative_bytes[index],
            "cumulative_mb": value,
        }
        for index, value in enumerate(cumulative_mb)
        if not math.isclose(
            value,
            cumulative_bytes[index] / (1024.0 * 1024.0),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]
    range_violations = [
        rounds[index]
        for index, row in enumerate(numeric_rows)
        if not (0.0 <= row["test/accuracy"] <= 1.0)
        or row["communication/round_bytes"] < 0
        or row["communication/cumulative_bytes"] < 0
    ]
    valid_round_sequence = bool(
        rounds
        and expected_round is not None
        and rounds[0] in (0, 1)
        and rounds == list(range(rounds[0], expected_round + 1))
    )
    result.update(
        {
            "row_count": len(rounds),
            "min_round": min(rounds) if rounds else None,
            "max_round": max(rounds) if rounds else None,
            "expected_round": expected_round,
            "has_expected_round": expected_round in rounds if expected_round is not None else None,
            "duplicate_rounds": duplicate_rounds,
            "round_order_violations": order_violations,
            "cumulative_mb_decreases": cumulative_decreases,
            "byte_recurrence_violations": byte_recurrence_violations,
            "mb_conversion_violations": mb_conversion_violations,
            "range_violations": range_violations,
            "valid_round_sequence": valid_round_sequence,
            "nonfinite_rows": nonfinite_rows,
            "parse_errors": parse_errors,
            "final_cumulative_mb": cumulative_mb[-1] if cumulative_mb else None,
        }
    )
    result["valid"] = not any(
        (
            result.get("missing_columns"),
            parse_errors,
            nonfinite_rows,
            duplicate_rounds,
            order_violations,
            cumulative_decreases,
            byte_recurrence_violations,
            mb_conversion_violations,
            range_violations,
            not valid_round_sequence,
            expected_round is not None and expected_round not in rounds,
        )
    )
    result["numeric_rows"] = numeric_rows
    return result


def scan_jsonl(
    path: Path,
    csv_rows: list[dict[str, float]],
    expected_round: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
        "valid": False,
    }
    if not path.is_file():
        return result

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("line is not a JSON object")
                    rows.append(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"line {line_number}: {exc}")
    except (OSError, UnicodeError) as exc:
        result["read_error"] = str(exc)
        return result

    rounds: list[int] = []
    parity_mismatches: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            round_value = int(float(row["round"]))
            rounds.append(round_value)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"record {index + 1}: invalid round: {exc}")
            continue
        if index >= len(csv_rows):
            parity_mismatches.append({"round": round_value, "reason": "missing CSV row"})
            continue
        csv_row = csv_rows[index]
        for key in (
            "round",
            "test/loss",
            "test/accuracy",
            "communication/round_bytes",
            "communication/cumulative_bytes",
            "communication/cumulative_mb",
        ):
            try:
                jsonl_value = float(row[key])
            except (KeyError, TypeError, ValueError):
                parity_mismatches.append(
                    {"round": round_value, "field": key, "reason": "missing or nonnumeric JSONL"}
                )
                continue
            if not math.isclose(jsonl_value, csv_row[key], rel_tol=0.0, abs_tol=1e-9):
                parity_mismatches.append(
                    {
                        "round": round_value,
                        "field": key,
                        "csv": csv_row[key],
                        "jsonl": jsonl_value,
                    }
                )
    if len(rows) != len(csv_rows):
        parity_mismatches.append(
            {"reason": "row-count mismatch", "csv": len(csv_rows), "jsonl": len(rows)}
        )

    duplicate_rounds = sorted(
        round_value for round_value, count in Counter(rounds).items() if count > 1
    )
    result.update(
        {
            "row_count": len(rows),
            "min_round": min(rounds) if rounds else None,
            "max_round": max(rounds) if rounds else None,
            "has_expected_round": expected_round in rounds if expected_round is not None else None,
            "duplicate_rounds": duplicate_rounds,
            "errors": errors,
            "csv_parity_mismatches": parity_mismatches,
        }
    )
    result["valid"] = not any(
        (
            errors,
            duplicate_rounds,
            parity_mismatches,
            expected_round is not None and expected_round not in rounds,
        )
    )
    return result


def scan_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
        "valid_json": False,
    }
    if not path.is_file():
        return result, None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result, None
    if not isinstance(manifest, dict):
        result["error"] = "manifest root is not an object"
        return result, None

    config = manifest.get("config")
    stored_hash = manifest.get("config_sha256")
    computed_hash = config_sha256(config) if isinstance(config, dict) else None
    git = manifest.get("git") if isinstance(manifest.get("git"), dict) else {}
    result.update(
        {
            "valid_json": True,
            "stored_config_sha256": stored_hash,
            "computed_config_sha256": computed_hash,
            "config_hash_matches": bool(stored_hash and stored_hash == computed_hash),
            "git": {
                "commit": git.get("commit"),
                "branch": git.get("branch"),
                "tag": git.get("tag"),
                "dirty": git.get("dirty"),
            },
        }
    )
    run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    algorithm = config.get("algorithm") if isinstance(config, dict) else {}
    dataset = config.get("dataset") if isinstance(config, dict) else {}
    heterogeneity = config.get("heterogeneity") if isinstance(config, dict) else {}
    experiment = config.get("experiment") if isinstance(config, dict) else {}
    comparisons = {
        "algorithm": (run.get("algorithm"), algorithm.get("name")),
        "dataset": (run.get("dataset"), dataset.get("name")),
        "alpha": (run.get("alpha"), heterogeneity.get("alpha")),
        "seed": (run.get("seed"), config.get("seed") if isinstance(config, dict) else None),
        "total_rounds": (run.get("total_rounds"), experiment.get("total_rounds")),
        "num_clients": (run.get("num_clients"), experiment.get("num_clients")),
    }
    result["run_config_mismatches"] = {
        key: {"run": values[0], "config": values[1]}
        for key, values in comparisons.items()
        if values[0] != values[1]
    }
    return result, manifest


def run_identity_from_artifacts(
    manifest: dict[str, Any] | None,
    path_fields: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    manifest = manifest or {}
    run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    algorithm = config.get("algorithm") if isinstance(config.get("algorithm"), dict) else {}
    dataset_cfg = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    heterogeneity = (
        config.get("heterogeneity") if isinstance(config.get("heterogeneity"), dict) else {}
    )

    dataset = run.get("dataset") or dataset_cfg.get("name")
    group = path_fields.get("experiment_group") or config.get("experiment_group")
    algorithm_config = (
        run.get("algorithm_config")
        or config.get("_algorithm_config_name")
        or path_fields.get("algorithm_path")
    )
    variant = path_fields.get("variant", "")
    alpha = run.get("alpha")
    if alpha is None:
        alpha = heterogeneity.get("alpha")
    formulation = algorithm.get("formulation")
    seed = run.get("seed")
    if seed is None:
        seed = config.get("seed", path_fields.get("path_seed"))

    fields = {
        "dataset": dataset,
        "experiment_group": group,
        "algorithm_config": algorithm_config,
        "variant": variant,
        "alpha": alpha,
        "formulation": formulation,
        "seed": seed,
    }
    required = (dataset, group, algorithm_config, alpha, seed)
    if any(value is None for value in required):
        return None, fields
    try:
        return (
            identity_key(
                str(dataset),
                str(group),
                str(algorithm_config),
                str(variant),
                float(alpha),
                int(formulation) if formulation is not None else None,
                int(seed),
            ),
            fields,
        )
    except (TypeError, ValueError):
        return None, fields


def scan_run(
    job_dir: Path,
    repo_root: Path,
    expected_rounds: dict[str, int],
) -> dict[str, Any]:
    path_fields = canonical_path_fields(job_dir, repo_root)
    manifest_result, manifest = scan_manifest(job_dir / "run_manifest.json")
    identity, identity_fields = run_identity_from_artifacts(manifest, path_fields)
    group = identity_fields.get("experiment_group")
    expected_round = expected_rounds.get(str(group)) if group is not None else None
    checkpoint = job_dir / "final_global_model.pt"
    hydra_config = job_dir / ".hydra" / "config.yaml"
    hydra_runtime = job_dir / ".hydra" / "hydra.yaml"
    hydra_overrides = job_dir / ".hydra" / "overrides.yaml"
    csv_result = scan_csv(job_dir / "experiment_log.csv", expected_round)
    jsonl_result = scan_jsonl(
        job_dir / "experiment_log.jsonl",
        csv_result.pop("numeric_rows", []),
        expected_round,
    )

    artifact_sizes = {
        "checkpoint": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "hydra_config": hydra_config.stat().st_size if hydra_config.is_file() else 0,
        "hydra_runtime": hydra_runtime.stat().st_size if hydra_runtime.is_file() else 0,
        "hydra_overrides": hydra_overrides.stat().st_size if hydra_overrides.is_file() else 0,
    }
    identity_mismatches: list[str] = []
    if path_fields.get("path_seed") is not None and identity_fields.get("seed") is not None:
        if int(path_fields["path_seed"]) != int(identity_fields["seed"]):
            identity_mismatches.append("path seed differs from manifest seed")
    if path_fields.get("algorithm_path") and identity_fields.get("algorithm_config"):
        if path_fields["algorithm_path"] != identity_fields["algorithm_config"]:
            identity_mismatches.append("path algorithm differs from manifest algorithm_config")
    manifest_config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    if path_fields.get("experiment_group") and manifest_config.get("experiment_group"):
        if path_fields["experiment_group"] != manifest_config["experiment_group"]:
            identity_mismatches.append("path group differs from manifest config group")

    violations: list[str] = []
    if identity is None:
        violations.append("identity could not be reconstructed")
    if not path_fields.get("canonical"):
        violations.append("non-canonical output path")
    if not manifest_result.get("valid_json"):
        violations.append("manifest missing or invalid")
    elif not manifest_result.get("config_hash_matches"):
        violations.append("manifest config hash mismatch")
    if manifest_result.get("run_config_mismatches"):
        violations.append("manifest run summary differs from resolved config")
    git = manifest_result.get("git", {})
    if not git.get("commit"):
        violations.append("manifest git commit missing")
    if git.get("dirty") is not False:
        violations.append("manifest does not prove a clean tree")
    if not csv_result.get("valid"):
        violations.append("telemetry CSV failed validation")
    if not jsonl_result.get("valid"):
        violations.append("telemetry JSONL failed validation or CSV parity")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        violations.append("final checkpoint missing or empty")
    if not all(path.is_file() for path in (hydra_config, hydra_runtime, hydra_overrides)):
        violations.append("Hydra artifact set is incomplete")
    violations.extend(identity_mismatches)

    return {
        "job_dir": relative_text(job_dir, repo_root),
        "path": path_fields,
        "identity": identity,
        "identity_fields": identity_fields,
        "identity_mismatches": identity_mismatches,
        "manifest": manifest_result,
        "telemetry": csv_result,
        "telemetry_jsonl": jsonl_result,
        "artifacts": {
            "checkpoint_exists": checkpoint.is_file(),
            "hydra_config_exists": hydra_config.is_file(),
            "hydra_runtime_exists": hydra_runtime.is_file(),
            "hydra_overrides_exists": hydra_overrides.is_file(),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "hydra_sha256": {
                "config.yaml": sha256_file(hydra_config) if hydra_config.is_file() else None,
                "hydra.yaml": sha256_file(hydra_runtime) if hydra_runtime.is_file() else None,
                "overrides.yaml": (
                    sha256_file(hydra_overrides) if hydra_overrides.is_file() else None
                ),
            },
            "bytes": artifact_sizes,
        },
        "strict_violations": violations,
        "strict_valid": not violations,
        "patchability": classify_patchability(violations),
    }


def classify_patchability(violations: list[str]) -> str:
    if not violations:
        return "analysis_ready"
    if violations == ["final checkpoint missing or empty"]:
        return "telemetry_usable_checkpoint_missing"
    if any("identity" in violation or "path" in violation for violation in violations):
        return "identity_or_collision_failure"
    if any("telemetry" in violation for violation in violations):
        return "telemetry_failure"
    if any("manifest" in violation or "clean tree" in violation for violation in violations):
        return "provenance_failure"
    return "artifact_failure"


def expected_rounds(groups: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, body in groups.items():
        rounds = {
            int(round_value)
            for regime in body.get("regimes", {}).values()
            for round_value in regime.get("total_rounds", [])
        }
        if len(rounds) == 1:
            result[name] = rounds.pop()
    return result


def closure_report(
    runs: list[dict[str, Any]],
    expected_groups: dict[str, Any],
) -> dict[str, Any]:
    closure: dict[str, Any] = {}
    for group_name in STUDY1_GROUPS:
        body = expected_groups[group_name]
        expected = set(body["runs"])
        members = [
            run for run in runs if run["identity_fields"].get("experiment_group") == group_name
        ]
        observed = Counter(run["identity"] for run in members if run["identity"] is not None)
        duplicates = {
            identity: [run["job_dir"] for run in members if run["identity"] == identity]
            for identity, count in observed.items()
            if count > 1
        }
        invalid = [
            {
                "identity": run["identity"],
                "job_dir": run["job_dir"],
                "violations": run["strict_violations"],
            }
            for run in members
            if not run["strict_valid"]
        ]
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        unidentified = [run["job_dir"] for run in members if run["identity"] is None]
        closed = not (missing or unexpected or duplicates or invalid or unidentified)
        closure[group_name] = {
            "expected": len(expected),
            "physical_run_dirs": len(members),
            "observed_unique_identities": len(observed),
            "missing": missing,
            "unexpected": unexpected,
            "duplicates": duplicates,
            "unidentified": unidentified,
            "invalid": invalid,
            "patchability": dict(sorted(Counter(run["patchability"] for run in members).items())),
            "strict_closed": closed,
        }
    return {
        "groups": closure,
        "strict_all_closed": all(group["strict_closed"] for group in closure.values()),
        "expected_total": sum(group["expected"] for group in closure.values()),
        "physical_run_dirs": sum(group["physical_run_dirs"] for group in closure.values()),
    }


def scan_sweep_status(path: Path, repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative_text(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result
    result["top_level_keys"] = sorted(value) if isinstance(value, dict) else []
    if isinstance(value, dict):
        for key in ("state", "matrix", "total", "completed", "failed", "failed_indices"):
            if key in value:
                result[key] = value[key]
    return result


def build_report(repo_root: Path, expected_path: Path) -> dict[str, Any]:
    expected_document = json.loads(expected_path.read_text(encoding="utf-8"))
    groups = expected_document["groups"]
    missing_groups = [name for name in STUDY1_GROUPS if name not in groups]
    if missing_groups:
        raise ValueError(f"expected-run manifest lacks Study 1 groups: {missing_groups}")

    inventory = inventory_outputs(repo_root / "outputs", repo_root)
    rounds = expected_rounds(groups)
    runs = [scan_run(path, repo_root, rounds) for path in inventory.pop("run_dirs")]
    runs.sort(key=lambda run: (run["identity"] or "", run["job_dir"]))
    sweep_paths = inventory.pop("sweep_statuses")
    closure = closure_report(runs, groups)
    outside_scope = [
        run["job_dir"]
        for run in runs
        if run["identity_fields"].get("experiment_group") not in STUDY1_GROUPS
    ]
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root.resolve()),
        "outputs_root": str((repo_root / "outputs").resolve()),
        "study1_groups": list(STUDY1_GROUPS),
        "expected_runs": {
            "path": relative_text(expected_path, repo_root),
            "sha256": sha256_file(expected_path),
            "expected_rounds": {name: rounds.get(name) for name in STUDY1_GROUPS},
        },
        "inventory": inventory,
        "closure": closure,
        "sweep_statuses": [scan_sweep_status(path, repo_root) for path in sweep_paths],
        "outside_scope_run_dirs": outside_scope,
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="fedmaq-experiments checkout containing outputs/ and docs/freeze/",
    )
    parser.add_argument(
        "--expected-runs",
        type=Path,
        default=None,
        help="expected_runs.json; defaults under --repo-root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("study1_artifact_audit.json"),
        help="JSON report path; may be outside the checkout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    expected_path = (
        args.expected_runs.resolve()
        if args.expected_runs is not None
        else repo_root / "docs" / "freeze" / "expected_runs.json"
    )
    if not expected_path.is_file():
        raise SystemExit(f"expected-run manifest not found: {expected_path}")
    report = build_report(repo_root, expected_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    closure = report["closure"]
    print(f"Report: {args.output.resolve()}")
    print(f"Output files: {report['inventory']['total_files']}")
    print(f"Output bytes: {report['inventory']['total_bytes']}")
    for name, group in closure["groups"].items():
        print(
            f"{name}: expected={group['expected']} "
            f"physical={group['physical_run_dirs']} "
            f"unique={group['observed_unique_identities']} "
            f"missing={len(group['missing'])} "
            f"unexpected={len(group['unexpected'])} "
            f"duplicates={len(group['duplicates'])} "
            f"invalid={len(group['invalid'])} "
            f"closed={group['strict_closed']}"
        )
    print(f"Study 1 strict closure: {closure['strict_all_closed']}")
    return 0 if closure["strict_all_closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
