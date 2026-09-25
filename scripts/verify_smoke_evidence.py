"""Verify pre-dispatch smoke gate evidence directories against Issue #100 criteria."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fedmaq.core.manifest import MANIFEST_FILENAME
from fedmaq.core.partitioning import canonical_partition_digest
from fedmaq.core.quantization_planner import DEFAULT_BIT_WIDTHS
from fedmaq.core.validation import (
    TELEMETRY_CSV_FILENAME,
    validate_run_evidence,
)

AlgorithmFamily = Literal["fedmaq", "feddistill", "baseline"]


@dataclass(frozen=True)
class SmokeCellSpec:
    name: str
    relative_path: Path
    expected_split: str
    family: AlgorithmFamily = "baseline"


DEFAULT_CELL_SPECS = (
    SmokeCellSpec("fedmaq", Path("fedmaq"), "test", family="fedmaq"),
    SmokeCellSpec("power_mean", Path("power_mean"), "test", family="fedmaq"),
    SmokeCellSpec("feddistill", Path("feddistill"), "test", family="feddistill"),
    SmokeCellSpec("ablation_no_data", Path("ablation_no_data"), "test", family="fedmaq"),
    SmokeCellSpec("val_split", Path("val_split"), "val", family="fedmaq"),
    # ADR-0016 2026-09-25: the exact V2 candidate, fedmaq_no_kd with post_process=true.
    SmokeCellSpec("no_kd_post", Path("no_kd_post"), "test", family="fedmaq"),
)


def _verify_manifest_contracts(
    cell_dir: Path,
    manifest_data: dict,
    spec: SmokeCellSpec,
) -> list[str]:
    errors: list[str] = []
    run_info = manifest_data.get("run", {})

    actual_split = run_info.get("split")
    actual_loader = run_info.get("loader_used")
    if actual_split != spec.expected_split:
        errors.append(
            f"{cell_dir.name}: expected split {spec.expected_split!r}, got {actual_split!r}"
        )
    if actual_loader != spec.expected_split:
        errors.append(
            f"{cell_dir.name}: expected loader_used {spec.expected_split!r}, got {actual_loader!r}"
        )

    for req_key in ("formulation", "p", "omega"):
        if req_key not in run_info:
            errors.append(f"{cell_dir.name}: run manifest lacks {req_key} key")

    partition_cache = manifest_data.get("partition_cache")
    if not isinstance(partition_cache, dict):
        errors.append(f"{cell_dir.name}: run manifest lacks top-level partition_cache object")
    else:
        cache_rel_path = partition_cache.get("path")
        expected_digest = partition_cache.get("sha256")
        if not expected_digest or not cache_rel_path:
            errors.append(f"{cell_dir.name}: partition_cache lacks path or sha256 digest")
        else:
            repo_root = Path(__file__).resolve().parents[1]
            cache_file = Path(cache_rel_path)
            if not cache_file.is_file():
                cache_file = repo_root / cache_rel_path
            if not cache_file.is_file():
                errors.append(f"{cell_dir.name}: cache file missing on disk: {cache_rel_path}")
            else:
                disk_cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                actual_digest = canonical_partition_digest(disk_cache_data)
                if actual_digest != expected_digest:
                    errors.append(
                        f"{cell_dir.name}: partition cache digest mismatch "
                        f"(manifest={expected_digest}, disk={actual_digest})"
                    )
    return errors


def _verify_fedmaq_telemetry(
    cell_dir: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    manifest_data: dict,
) -> list[str]:
    errors: list[str] = []
    binding_key = "algorithm/fedmaq/tier1_binding_fraction"
    if binding_key not in fieldnames:
        errors.append(f"{cell_dir.name}: CSV missing {binding_key}")
    else:
        binding_values = [
            float(row[binding_key])
            for row in rows
            if row.get(binding_key) and row[binding_key] != ""
        ]
        if not binding_values or all(v == 0.0 for v in binding_values):
            errors.append(f"{cell_dir.name}: tier1_binding_fraction is zero across all rounds")

    for b in DEFAULT_BIT_WIDTHS:
        q_col = f"algorithm/fedmaq/q_count_{b}"
        q_hat_col = f"algorithm/fedmaq/q_hat_count_{b}"
        if q_col not in fieldnames:
            errors.append(f"{cell_dir.name}: CSV missing histogram column {q_col}")
        if q_hat_col not in fieldnames:
            errors.append(f"{cell_dir.name}: CSV missing histogram column {q_hat_col}")

    num_clients = int(manifest_data.get("run", {}).get("num_clients", 2))
    client_fraction = float(
        manifest_data.get("config", {}).get("experiment", {}).get("client_fraction", 1.0)
    )
    expected_sampled = max(int(num_clients * client_fraction), 1)

    for row in rows:
        try:
            round_idx = int(row.get("round", -1))
        except ValueError:
            continue
        if round_idx < 1:
            continue

        try:
            q_total = sum(
                int(row.get(f"algorithm/fedmaq/q_count_{b}", 0) or 0) for b in DEFAULT_BIT_WIDTHS
            )
            q_hat_total = sum(
                int(row.get(f"algorithm/fedmaq/q_hat_count_{b}", 0) or 0)
                for b in DEFAULT_BIT_WIDTHS
            )
            if q_total != expected_sampled:
                errors.append(
                    f"{cell_dir.name} r{round_idx}: sum of q_count_* is {q_total}, "
                    f"expected {expected_sampled}"
                )
            if q_hat_total != expected_sampled:
                errors.append(
                    f"{cell_dir.name} r{round_idx}: sum of q_hat_count_* is {q_hat_total}, "
                    f"expected {expected_sampled}"
                )
        except ValueError as exc:
            errors.append(f"{cell_dir.name} r{round_idx}: invalid histogram integer: {exc}")
    return errors


def _verify_feddistill_telemetry(
    cell_dir: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> list[str]:
    errors: list[str] = []
    norm_key = "algorithm/feddistill/global_logits_l2_norm"
    if norm_key not in fieldnames:
        errors.append(f"{cell_dir.name}: CSV missing {norm_key}")
    else:
        norm_val = rows[-1].get(norm_key)
        if not norm_val or float(norm_val) <= 0.0:
            errors.append(
                f"{cell_dir.name}: global logits norm is non-positive or missing ({norm_val!r})"
            )
    return errors


def verify_cell_evidence(cell_dir: Path, spec: SmokeCellSpec) -> list[str]:
    validation_result = validate_run_evidence(cell_dir)
    if not validation_result.is_complete:
        return [f"{cell_dir.name}: incomplete run evidence"] + [
            f"{cell_dir.name}: {err}" for err in validation_result.errors
        ]

    manifest_data = json.loads((cell_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    errors = _verify_manifest_contracts(cell_dir, manifest_data, spec)

    csv_path = cell_dir / TELEMETRY_CSV_FILENAME
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if len(rows) < 2:
        errors.append(f"{cell_dir.name}: expected at least 2 rounds in CSV, found {len(rows)}")
        return errors

    if spec.family == "fedmaq":
        errors.extend(_verify_fedmaq_telemetry(cell_dir, rows, fieldnames, manifest_data))
    elif spec.family == "feddistill":
        errors.extend(_verify_feddistill_telemetry(cell_dir, rows, fieldnames))

    return errors


def verify_all_smoke_evidence(base_dir: Path) -> list[str]:
    all_errors: list[str] = []
    for spec in DEFAULT_CELL_SPECS:
        cell_dir = base_dir / spec.relative_path
        if not cell_dir.is_dir():
            all_errors.append(f"Missing smoke output directory: {cell_dir}")
            continue
        all_errors.extend(verify_cell_evidence(cell_dir, spec))
    return all_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=Path("outputs/smoke"),
        help="Root directory containing smoke run outputs",
    )
    args = parser.parse_args()

    errors = verify_all_smoke_evidence(args.smoke_dir)
    if errors:
        print("Smoke Gate Evidence Verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"All smoke evidence in {args.smoke_dir} PASSED verification successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
