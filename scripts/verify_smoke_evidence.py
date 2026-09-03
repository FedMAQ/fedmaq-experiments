"""Verify pre-dispatch smoke gate evidence directories against Issue #100 & #102 criteria."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME
from fedmaq.core.manifest import MANIFEST_FILENAME
from fedmaq.core.partitioning import canonical_partition_digest
from fedmaq.core.quantization_planner import DEFAULT_BIT_WIDTHS
from fedmaq.core.validation import validate_run_evidence

EXPECTED_FILES = (
    FINAL_MODEL_FILENAME,
    MANIFEST_FILENAME,
    "experiment_log.csv",
    "experiment_log.jsonl",
)


@dataclass(frozen=True)
class SmokeCellSpec:
    name: str
    relative_path: Path
    expected_split: str
    check_fedmaq_metrics: bool = False
    check_feddistill_metrics: bool = False


DEFAULT_CELL_SPECS = (
    SmokeCellSpec("fedmaq", Path("fedmaq"), "test", check_fedmaq_metrics=True),
    SmokeCellSpec("power_mean", Path("power_mean"), "test", check_fedmaq_metrics=True),
    SmokeCellSpec("feddistill", Path("feddistill"), "test", check_feddistill_metrics=True),
    SmokeCellSpec("ablation_no_data", Path("ablation_no_data"), "test", check_fedmaq_metrics=True),
    SmokeCellSpec("val_split", Path("val_split"), "val", check_fedmaq_metrics=True),
)


def verify_cell_evidence(cell_dir: Path, spec: SmokeCellSpec) -> list[str]:
    errors: list[str] = []

    for filename in EXPECTED_FILES:
        target = cell_dir / filename
        if not target.is_file():
            errors.append(f"{cell_dir.name}: missing required artifact {filename}")

    if errors:
        return errors

    validation_result = validate_run_evidence(cell_dir)
    if not validation_result.is_complete:
        errors.append(f"{cell_dir.name}: validate_run_evidence reported incomplete")
    if validation_result.errors:
        errors.extend(f"{cell_dir.name}: {err}" for err in validation_result.errors)

    manifest_data = json.loads((cell_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
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

    if "formulation" not in run_info:
        errors.append(f"{cell_dir.name}: run manifest lacks formulation key")
    if "p" not in run_info:
        errors.append(f"{cell_dir.name}: run manifest lacks p key")
    if "omega" not in run_info:
        errors.append(f"{cell_dir.name}: run manifest lacks omega key")

    partition_cache = manifest_data.get("partition_cache")
    if not partition_cache or not isinstance(partition_cache, dict):
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
                errors.append(
                    f"{cell_dir.name}: partition cache file not found on disk: {cache_rel_path}"
                )
            else:
                disk_cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                actual_digest = canonical_partition_digest(disk_cache_data)
                if actual_digest != expected_digest:
                    errors.append(
                        f"{cell_dir.name}: partition cache digest mismatch "
                        f"(manifest={expected_digest}, disk={actual_digest})"
                    )

    with (cell_dir / "experiment_log.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if len(rows) < 2:
        errors.append(f"{cell_dir.name}: expected at least 2 rounds in CSV, found {len(rows)}")
        return errors

    expected_clients = int(run_info.get("num_clients", 2))

    if spec.check_fedmaq_metrics:
        binding_key = "algorithm/fedmaq/tier1_binding_fraction"
        if binding_key not in (reader.fieldnames or []):
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
            if q_col not in (reader.fieldnames or []):
                errors.append(f"{cell_dir.name}: CSV missing histogram column {q_col}")
            if q_hat_col not in (reader.fieldnames or []):
                errors.append(f"{cell_dir.name}: CSV missing histogram column {q_hat_col}")

        for row in rows:
            round_idx = row.get("round", "?")
            try:
                q_total = sum(
                    int(row.get(f"algorithm/fedmaq/q_count_{b}", 0) or 0)
                    for b in DEFAULT_BIT_WIDTHS
                )
                if q_total != expected_clients:
                    errors.append(
                        f"{cell_dir.name} r{round_idx}: sum of q_count_* is {q_total}, "
                        f"expected {expected_clients}"
                    )
            except ValueError as exc:
                errors.append(
                    f"{cell_dir.name} round {round_idx}: invalid histogram integer: {exc}"
                )

    if spec.check_feddistill_metrics:
        norm_key = "algorithm/feddistill/global_logits_l2_norm"
        if norm_key not in (reader.fieldnames or []):
            errors.append(f"{cell_dir.name}: CSV missing {norm_key}")
        else:
            norm_val = rows[-1].get(norm_key)
            if not norm_val or float(norm_val) <= 0.0:
                errors.append(
                    f"{cell_dir.name}: global logits norm is non-positive or missing ({norm_val!r})"
                )

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
