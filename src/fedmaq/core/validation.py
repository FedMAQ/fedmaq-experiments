"""Atomic run completion and telemetry evidence validation.

Validates that an experiment run has finished completely, atomically, and without
corruption, truncation, missing rounds, duplicate rounds, non-finite values,
non-monotonic cumulative metrics, or identity mismatches.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME
from fedmaq.core.manifest import MANIFEST_FILENAME
from fedmaq.core.partitioning import canonical_partition_digest
from fedmaq.core.run_identity import parse_run_directory

logger = logging.getLogger("fedmaq.validation")

TELEMETRY_CSV_FILENAME = "experiment_log.csv"
TELEMETRY_JSONL_FILENAME = "experiment_log.jsonl"
REQUIRED_METRIC_COLUMNS = ("round", "communication/cumulative_mb")
_JSONL_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "fedavg": ("client/avg_train_loss",),
    "fedprox": ("client/avg_train_loss",),
    "fedpaq": ("client/avg_train_loss",),
    "fedpaq_pipeline": ("client/avg_train_loss",),
    "dadaquant": ("client/avg_train_loss",),
    "feddistill": ("client/avg_task_loss", "client/avg_distill_loss"),
    "fedkd": (
        "client/avg_task_loss_student",
        "client/avg_task_loss_teacher",
        "client/avg_kd_loss_student",
        "client/avg_kd_loss_teacher",
        "client/avg_teacher_acc",
    ),
    "fedmaq": ("client/avg_train_loss", "algorithm/fedmaq/server_kd_loss"),
    "power_mean": ("client/avg_train_loss",),
}
_JSONL_REQUIRED_PREFIXES: dict[str, tuple[str, ...]] = {
    "fedmaq": ("algorithm/fedmaq/q_count_", "algorithm/fedmaq/q_hat_count_"),
}


@dataclass(frozen=True)
class ValidationResult:
    """Detailed evidence validation outcome for an experiment run."""

    is_complete: bool
    output_dir: Path
    errors: tuple[str, ...] = field(default_factory=tuple)
    total_rounds: int | None = None
    max_round: int | None = None
    manifest: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.is_complete


def _infer_repo_root(output_dir: Path) -> Path | None:
    try:
        resolved = output_dir.resolve()
        for parent in resolved.parents:
            if (parent / "conf").is_dir() or (parent / "outputs").is_dir():
                return parent
            if (parent / "pyproject.toml").is_file():
                return parent
    except Exception:
        pass
    return None


def validate_run_evidence(
    output_dir: Path,
    expected_rounds: int | None = None,
    repo_root: Path | None = None,
) -> ValidationResult:
    """Validate all completion invariants for a single experiment run directory.

    Checks:
    1. Checkpoint: ``final_global_model.pt`` exists, non-empty, and loads a valid state_dict.
    2. Manifest: ``run_manifest.json`` exists, valid JSON, matching parsed directory identity.
    3. Telemetry: ``experiment_log.csv`` exists, non-empty, with required metric columns.
    4. Rounds: Contiguous unique sequence 1..R, optionally prefixed by round 0.
    5. Finiteness: Required values and populated optional values are finite.
    6. JSONL: The authoritative log exists and has the same rounds and required keys.
    7. Monotonicity: Cumulative metrics (e.g. cumulative MB) are non-decreasing across rounds.

    Returns a :class:`ValidationResult` detailing whether the run is complete and errors found.
    """
    errors: list[str] = []
    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        return ValidationResult(
            is_complete=False,
            output_dir=output_dir,
            errors=(f"output directory does not exist: {output_dir}",),
        )

    if repo_root is None:
        repo_root = _infer_repo_root(output_dir)

    checkpoint_path = output_dir / FINAL_MODEL_FILENAME
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        errors.append(f"missing or empty checkpoint: {checkpoint_path}")
    else:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict) or not state_dict:
                errors.append(
                    f"checkpoint does not contain a non-empty state_dict: {checkpoint_path}"
                )
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            pickle.UnpicklingError,
            Exception,
        ) as exc:
            errors.append(f"failed to load checkpoint {checkpoint_path}: {exc}")

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_data: dict[str, Any] | None = None
    manifest_rounds: int | None = None

    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        errors.append(f"missing or empty run manifest: {manifest_path}")
    else:
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_data, dict):
                errors.append(f"run manifest is not a JSON object: {manifest_path}")
                manifest_data = None
            else:
                run_info = manifest_data.get("run", {})
                if not isinstance(run_info, dict):
                    errors.append(f"manifest 'run' entry is not a mapping: {manifest_path}")
                else:
                    if run_info.get("total_rounds") is not None:
                        try:
                            manifest_rounds = int(run_info["total_rounds"])
                        except (ValueError, TypeError):
                            errors.append(
                                f"invalid total_rounds in manifest: {run_info.get('total_rounds')}"
                            )

                    # Cross-verify with directory structure if canonical
                    if repo_root is not None:
                        parsed = parse_run_directory(output_dir, repo_root)
                        if parsed is not None:
                            dataset_val = run_info.get("dataset")
                            if dataset_val and not (
                                parsed.dataset_model == dataset_val
                                or parsed.dataset_model.startswith(f"{dataset_val}_")
                            ):
                                errors.append(
                                    f"identity mismatch on dataset: path={parsed.dataset_model} "
                                    f"vs manifest={dataset_val}"
                                )
                            alg_config_val = run_info.get("algorithm_config") or run_info.get(
                                "algorithm"
                            )
                            if alg_config_val and parsed.algorithm_path != alg_config_val:
                                errors.append(
                                    f"identity mismatch on algorithm: path={parsed.algorithm_path} "
                                    f"vs manifest={alg_config_val}"
                                )
                            if run_info.get("seed") is not None and parsed.seed != int(
                                run_info["seed"]
                            ):
                                errors.append(
                                    f"identity mismatch on seed: path={parsed.seed} "
                                    f"vs manifest={run_info.get('seed')}"
                                )
                            if run_info.get("variant") is not None and (
                                parsed.variant or ""
                            ) != run_info.get("variant"):
                                errors.append(
                                    f"identity mismatch on variant: path={parsed.variant!r} "
                                    f"vs manifest={run_info.get('variant')!r}"
                                )
                            alpha_val = run_info.get("alpha")
                            if alpha_val is not None and not (
                                parsed.heterogeneity_path.endswith(str(alpha_val))
                                or f"alpha_{alpha_val}" in parsed.heterogeneity_path
                            ):
                                errors.append(
                                    f"identity mismatch on alpha: path={parsed.heterogeneity_path} "
                                    f"vs manifest={alpha_val}"
                                )

                    # Partition cache validation (Issues #99 and #102)
                    partition_cache = manifest_data.get("partition_cache")
                    protocol_info = manifest_data.get("protocol", {})
                    is_replacement_scientific = (
                        isinstance(protocol_info, dict)
                        and protocol_info.get("name") == "replacement-v1"
                        and protocol_info.get("historical") is False
                        and protocol_info.get("stage") != "assurance"
                    )
                    if partition_cache is not None:
                        if not isinstance(partition_cache, dict):
                            errors.append(
                                f"manifest 'partition_cache' not a mapping: {manifest_path}"
                            )
                        else:
                            cache_rel_path = partition_cache.get("path")
                            expected_sha256 = partition_cache.get("sha256")
                            expected_schema = partition_cache.get("schema_version")
                            if not cache_rel_path or not expected_sha256:
                                errors.append(
                                    f"manifest 'partition_cache' missing path/sha256: "
                                    f"{manifest_path}"
                                )
                            else:
                                root = repo_root or Path.cwd()
                                full_cache_path = Path(root) / cache_rel_path
                                if not full_cache_path.is_file():
                                    errors.append(f"partition cache missing: {cache_rel_path}")
                                else:
                                    try:
                                        with open(full_cache_path, encoding="utf-8") as f:
                                            cache_obj = json.load(f)
                                        actual_sha256 = canonical_partition_digest(cache_obj)
                                        if actual_sha256 != expected_sha256:
                                            errors.append(
                                                "partition cache digest mismatch: "
                                                f"expected={expected_sha256}, "
                                                f"actual={actual_sha256}"
                                            )
                                        if (
                                            expected_schema is not None
                                            and cache_obj.get("schema_version") != expected_schema
                                        ):
                                            errors.append(
                                                f"partition cache schema mismatch: "
                                                f"expected={expected_schema}, "
                                                f"actual={cache_obj.get('schema_version')}"
                                            )
                                    except Exception as exc:
                                        errors.append(
                                            f"failed to read partition cache {full_cache_path}: "
                                            f"{exc}"
                                        )
                    elif is_replacement_scientific:
                        errors.append(
                            f"run manifest missing required partition_cache: {manifest_path}"
                        )
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            errors.append(f"corrupt run manifest {manifest_path}: {exc}")

    # Determine expected total rounds R
    effective_rounds: int | None = expected_rounds
    if effective_rounds is None:
        effective_rounds = manifest_rounds

    csv_path = output_dir / TELEMETRY_CSV_FILENAME
    max_observed_round: int | None = None

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        errors.append(f"missing or empty telemetry log: {csv_path}")
    else:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                errors.append(f"telemetry CSV has no rows: {csv_path}")
            else:
                for col in REQUIRED_METRIC_COLUMNS:
                    if col not in df.columns:
                        errors.append(f"telemetry CSV missing required column {col!r}: {csv_path}")

                if "round" in df.columns:
                    round_series = df["round"]
                    if not pd.api.types.is_numeric_dtype(round_series) or round_series.isna().any():
                        errors.append(
                            f"telemetry round column contains NaN or non-numeric values: {csv_path}"
                        )
                    else:
                        rounds_list = [int(r) for r in round_series]
                        if any(float(r) != int(r) for r in round_series):
                            errors.append(
                                f"telemetry round column contains non-integer values: {csv_path}"
                            )
                        max_observed_round = max(rounds_list) if rounds_list else None

                        if effective_rounds is None:
                            effective_rounds = max_observed_round

                        if effective_rounds is not None and effective_rounds > 0:
                            expected_sequences = (
                                list(range(1, effective_rounds + 1)),
                                list(range(0, effective_rounds + 1)),
                            )
                            if rounds_list not in expected_sequences:
                                if len(rounds_list) != len(set(rounds_list)):
                                    dup_rounds = sorted(
                                        [
                                            r
                                            for r, count in pd.Series(rounds_list)
                                            .value_counts()
                                            .items()
                                            if count > 1
                                        ]
                                    )
                                    errors.append(f"telemetry has duplicate rounds: {dup_rounds}")
                                elif len(rounds_list) < effective_rounds:
                                    errors.append(
                                        f"telemetry is truncated: observed {len(rounds_list)} "
                                        f"rounds, expected {effective_rounds}"
                                    )
                                else:
                                    errors.append(
                                        "telemetry rounds are not contiguous "
                                        f"1..{effective_rounds}: observed {rounds_list[:10]}..."
                                    )

                # Required columns must be populated and finite. Optional columns
                # are commonly blank (for example round-secondary bytes on all
                # non-DAdaQuant arms), so only their populated values are checked.
                numeric_cols = df.select_dtypes(include=["number"]).columns
                for col in numeric_cols:
                    if df[col].isna().any() and not df[col].isna().all():
                        nonzero_nan = df.loc[df["round"] != 0, col].isna().any()
                        if nonzero_nan:
                            errors.append(
                                f"telemetry column {col!r} contains non-finite values (NaN/Inf)"
                            )
                    values = df[col].dropna().to_numpy()
                    if col in REQUIRED_METRIC_COLUMNS:
                        nonzero_rows = df.loc[df["round"] != 0, col].dropna()
                        if len(nonzero_rows) != len(df.loc[df["round"] != 0]):
                            errors.append(
                                f"telemetry required column {col!r} contains NaN: {csv_path}"
                            )
                    if not all(math.isfinite(float(v)) for v in values):
                        errors.append(
                            f"telemetry column {col!r} contains non-finite values (NaN/Inf)"
                        )

                if "communication/cumulative_mb" in df.columns:
                    mb_values = df["communication/cumulative_mb"].to_numpy()
                    for idx in range(1, len(mb_values)):
                        if mb_values[idx] < mb_values[idx - 1] - 1e-9:
                            errors.append(
                                f"non-monotonic cumulative MB at round {df['round'].iloc[idx]}: "
                                f"{mb_values[idx]} < {mb_values[idx - 1]}"
                            )
                            break

                if "communication/cumulative_bytes" in df.columns:
                    byte_values = df["communication/cumulative_bytes"].to_numpy()
                    for idx in range(1, len(byte_values)):
                        if byte_values[idx] < byte_values[idx - 1]:
                            errors.append(
                                f"non-monotonic cumulative bytes at round {df['round'].iloc[idx]}: "
                                f"{byte_values[idx]} < {byte_values[idx - 1]}"
                            )
                            break
        except Exception as exc:
            errors.append(f"failed to read or validate telemetry CSV {csv_path}: {exc}")

    # JSONL is the authoritative superset for dynamic/client metrics. It is
    # required for every complete run and must remain row-aligned with the CSV.
    jsonl_path = output_dir / TELEMETRY_JSONL_FILENAME
    jsonl_rounds: list[int] = []
    if not jsonl_path.is_file() or jsonl_path.stat().st_size == 0:
        errors.append(f"missing or empty telemetry JSONL: {jsonl_path}")
    else:
        try:
            algorithm_config = None
            if manifest_data is not None:
                run_info = manifest_data.get("run", {})
                if isinstance(run_info, dict):
                    algorithm_config = run_info.get("algorithm_config") or run_info.get("algorithm")
            required_jsonl_keys = _JSONL_REQUIRED_KEYS.get(str(algorithm_config))
            required_jsonl_prefixes = _JSONL_REQUIRED_PREFIXES.get(str(algorithm_config), ())
            for line_number, line in enumerate(
                jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    errors.append(f"telemetry JSONL line {line_number} is not a JSON object")
                    continue
                missing = [key for key in REQUIRED_METRIC_COLUMNS if key not in record]
                if missing:
                    errors.append(
                        f"telemetry JSONL line {line_number} missing required keys {missing}"
                    )
                    continue
                round_value = record["round"]
                if isinstance(round_value, bool) or not isinstance(round_value, (int, float)):
                    errors.append(f"telemetry JSONL line {line_number} has a non-numeric round")
                    continue
                if not math.isfinite(float(round_value)) or float(round_value) != int(round_value):
                    errors.append(f"telemetry JSONL line {line_number} has a non-integer round")
                    continue
                required_values_valid = True
                for key in REQUIRED_METRIC_COLUMNS:
                    value = record[key]
                    if round_value == 0 and value is None:
                        continue
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    ):
                        required_values_valid = False
                        break
                if not required_values_valid:
                    errors.append(
                        f"telemetry JSONL line {line_number} has missing/non-finite required values"
                    )
                    continue
                if int(round_value) != 0:
                    missing_analysis_keys = [
                        key for key in required_jsonl_keys or () if key not in record
                    ]
                    missing_analysis_prefixes = [
                        prefix
                        for prefix in required_jsonl_prefixes
                        if not any(key.startswith(prefix) for key in record)
                    ]
                    if missing_analysis_keys or missing_analysis_prefixes:
                        errors.append(
                            f"telemetry JSONL line {line_number} is missing readout keys: "
                            f"keys={missing_analysis_keys}, prefixes={missing_analysis_prefixes}"
                        )
                        continue
                    analysis_values = [record[key] for key in required_jsonl_keys or ()] + [
                        record[key]
                        for prefix in required_jsonl_prefixes
                        for key in record
                        if key.startswith(prefix)
                    ]
                    if any(
                        value is None
                        or isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in analysis_values
                    ):
                        errors.append(
                            f"telemetry JSONL line {line_number} has missing/non-finite "
                            "readout values"
                        )
                        continue
                jsonl_rounds.append(int(round_value))

            if csv_path.is_file() and csv_path.stat().st_size:
                csv_rounds = [int(value) for value in pd.read_csv(csv_path)["round"]]
                if jsonl_rounds != csv_rounds:
                    errors.append(
                        "telemetry JSONL rounds do not match CSV rounds: "
                        f"jsonl={jsonl_rounds[:10]}... csv={csv_rounds[:10]}..."
                    )
        except Exception as exc:
            errors.append(f"failed to read or validate telemetry JSONL {jsonl_path}: {exc}")

    is_complete = len(errors) == 0 and (effective_rounds is not None and effective_rounds > 0)
    return ValidationResult(
        is_complete=is_complete,
        output_dir=output_dir,
        errors=tuple(errors),
        total_rounds=effective_rounds,
        max_round=max_observed_round,
        manifest=manifest_data,
    )


def is_run_evidence_complete(
    output_dir: Path,
    expected_rounds: int | None = None,
    repo_root: Path | None = None,
) -> bool:
    """Convenience boolean check for whether a run directory holds complete valid evidence."""
    return validate_run_evidence(
        output_dir, expected_rounds=expected_rounds, repo_root=repo_root
    ).is_complete


__all__ = [
    "REQUIRED_METRIC_COLUMNS",
    "TELEMETRY_CSV_FILENAME",
    "TELEMETRY_JSONL_FILENAME",
    "ValidationResult",
    "is_run_evidence_complete",
    "validate_run_evidence",
]
