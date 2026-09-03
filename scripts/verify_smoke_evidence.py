"""Verify pre-dispatch smoke gate evidence directories against Issue #100 & #102 criteria."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME
from fedmaq.core.manifest import MANIFEST_FILENAME
from fedmaq.core.quantization_planner import DEFAULT_BIT_WIDTHS
from fedmaq.core.validation import validate_run_evidence

EXPECTED_FILES = (
    FINAL_MODEL_FILENAME,
    MANIFEST_FILENAME,
    "experiment_log.csv",
    "experiment_log.jsonl",
)


def verify_cell_evidence(
    cell_dir: Path,
    *,
    expected_split: str,
    is_fedmaq: bool = False,
    is_feddistill: bool = False,
) -> list[str]:
    """Verify one smoke run directory and return any defect descriptions."""
    errors: list[str] = []

    # 1. Structural file presence
    for filename in EXPECTED_FILES:
        target = cell_dir / filename
        if not target.is_file():
            errors.append(f"{cell_dir.name}: missing required artifact {filename}")

    if errors:
        return errors

    # 2. Evidence validation via core validator (#96 regression check)
    validation_result = validate_run_evidence(cell_dir)
    if not validation_result.is_complete:
        errors.append(f"{cell_dir.name}: validate_run_evidence reported incomplete")
    if validation_result.errors:
        errors.extend(f"{cell_dir.name}: {err}" for err in validation_result.errors)

    # 3. Manifest contract check (#99 & #102)
    manifest_data = json.loads((cell_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    run_info = manifest_data.get("run", {})

    actual_split = run_info.get("split")
    actual_loader = run_info.get("loader_used")
    if actual_split != expected_split:
        errors.append(f"{cell_dir.name}: expected split {expected_split!r}, got {actual_split!r}")
    if actual_loader != expected_split:
        errors.append(
            f"{cell_dir.name}: expected loader_used {expected_split!r}, got {actual_loader!r}"
        )

    # Partition cache digest provenance (#99 & #102)
    partition_cache = run_info.get("partition_cache")
    if not partition_cache or not isinstance(partition_cache, dict):
        errors.append(f"{cell_dir.name}: run manifest lacks partition_cache object")
    elif not partition_cache.get("sha256"):
        errors.append(f"{cell_dir.name}: run manifest partition_cache lacks sha256 digest")

    # Formulation metadata
    if "formulation" not in run_info:
        errors.append(f"{cell_dir.name}: run manifest lacks formulation key")

    # 4. Telemetry columns
    with (cell_dir / "experiment_log.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if len(rows) < 2:
        errors.append(f"{cell_dir.name}: expected at least 2 rounds in CSV, found {len(rows)}")
        return errors

    # Check FedMAQ metrics (#97 & #102)
    if is_fedmaq:
        # Check Tier-1 binding fraction
        binding_key = "algorithm/fedmaq/tier1_binding_fraction"
        if binding_key not in reader.fieldnames:
            errors.append(f"{cell_dir.name}: CSV missing {binding_key}")

        # Check bit-width histograms
        for b in DEFAULT_BIT_WIDTHS:
            q_col = f"algorithm/fedmaq/q_count_{b}"
            q_hat_col = f"algorithm/fedmaq/q_hat_count_{b}"
            if q_col not in reader.fieldnames:
                errors.append(f"{cell_dir.name}: CSV missing histogram column {q_col}")
            if q_hat_col not in reader.fieldnames:
                errors.append(f"{cell_dir.name}: CSV missing histogram column {q_hat_col}")

        # Check histogram client sum per round
        for row in rows:
            round_idx = row.get("round", "?")
            try:
                q_total = sum(
                    int(row.get(f"algorithm/fedmaq/q_count_{b}", 0) or 0)
                    for b in DEFAULT_BIT_WIDTHS
                )
                if q_total != 2:
                    errors.append(
                        f"{cell_dir.name} r{round_idx}: sum of q_count_* is {q_total}, expected 2"
                    )
            except ValueError as exc:
                errors.append(
                    f"{cell_dir.name} round {round_idx}: invalid histogram integer: {exc}"
                )

    # Check FedDistill+ metrics (#98)
    if is_feddistill:
        norm_key = "algorithm/feddistill/global_logits_l2_norm"
        if norm_key not in reader.fieldnames:
            errors.append(f"{cell_dir.name}: CSV missing {norm_key}")
        else:
            # Row 1 is round 1, row 2 is round 2
            norm_val = rows[-1].get(norm_key)
            if not norm_val or float(norm_val) <= 0.0:
                errors.append(
                    f"{cell_dir.name}: global logits norm is non-positive or missing ({norm_val!r})"
                )

    return errors


def verify_all_smoke_evidence(base_dir: Path) -> list[str]:
    """Verify the standard five smoke directories."""
    all_errors: list[str] = []

    fedmaq_dir = base_dir / "fedmaq"
    power_mean_dir = base_dir / "power_mean"
    feddistill_dir = base_dir / "feddistill"
    ablation_dir = base_dir / "ablation_no_data"
    val_split_dir = base_dir / "val_split"

    cells = (
        (fedmaq_dir, "test", True, False),
        (power_mean_dir, "test", True, False),
        (feddistill_dir, "test", False, True),
        (ablation_dir, "test", True, False),
        (val_split_dir, "val", True, False),
    )

    for cell_dir, split, is_fedmaq, is_feddistill in cells:
        if not cell_dir.is_dir():
            all_errors.append(f"Missing smoke output directory: {cell_dir}")
            continue
        errs = verify_cell_evidence(
            cell_dir,
            expected_split=split,
            is_fedmaq=is_fedmaq,
            is_feddistill=is_feddistill,
        )
        all_errors.extend(errs)

    return all_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=Path("outputs/smoke"),
        help="Root directory containing the smoke run outputs",
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
