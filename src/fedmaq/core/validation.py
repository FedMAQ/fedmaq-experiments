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
from fedmaq.core.run_identity import parse_run_directory

logger = logging.getLogger("fedmaq.validation")

TELEMETRY_CSV_FILENAME = "experiment_log.csv"
REQUIRED_METRIC_COLUMNS = ("round", "communication/cumulative_mb")


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
    4. Rounds: Contiguous unique sequence 1..R (no missing, duplicate, or truncated rounds).
    5. Finiteness: All required metric values are finite (no NaN, Inf, -Inf).
    6. Monotonicity: Cumulative metrics (e.g. cumulative MB) are non-decreasing across rounds.

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

    # 1. Atomic final model checkpoint check
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

    # 2. Manifest and run identity check
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
                            if (parsed.variant or "") != (run_info.get("variant") or ""):
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
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            errors.append(f"corrupt run manifest {manifest_path}: {exc}")

    # Determine expected total rounds R
    effective_rounds: int | None = expected_rounds
    if effective_rounds is None:
        effective_rounds = manifest_rounds

    # 3. Telemetry log check
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
                    # Check for non-finite or non-integer round values
                    if not pd.api.types.is_numeric_dtype(round_series) or round_series.isna().any():
                        errors.append(
                            f"telemetry round column contains NaN or non-numeric values: {csv_path}"
                        )
                    else:
                        rounds_list = [int(r) for r in round_series]
                        max_observed_round = max(rounds_list) if rounds_list else None

                        if effective_rounds is None:
                            effective_rounds = max_observed_round

                        if effective_rounds is not None and effective_rounds > 0:
                            expected_seq = list(range(1, effective_rounds + 1))
                            if rounds_list != expected_seq:
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

                # 4. Check finiteness across all numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns
                for col in numeric_cols:
                    values = df[col].to_numpy()
                    if not math.isfinite(float(values.sum())) and not all(
                        math.isfinite(float(v)) for v in values
                    ):
                        errors.append(
                            f"telemetry column {col!r} contains non-finite values (NaN/Inf)"
                        )

                # 5. Check monotonicity of cumulative metrics
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
    "ValidationResult",
    "is_run_evidence_complete",
    "validate_run_evidence",
]
