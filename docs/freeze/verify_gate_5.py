"""Verify Gate 5's narrow telemetry producer-consumer compatibility contract.

Envelope: fedmaq-experiments:docs/freeze/assurance-envelope-2026-08-29.json
Gate: 5
Producer: assurance orchestrator
Created: 2026-08-30
Bound revisions: fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5;
fedmaq-analyses@d68a3c44ebd7b9b9301b1f5912190bfdf7ea65fd.
Content digest: recorded in the envelope's evidence_sha256 entry.

Usage:
    uv run python docs/freeze/verify_gate_5.py --analyses-root ..\\fedmaq-analyses \\
        --require-revision d68a3c44ebd7b9b9301b1f5912190bfdf7ea65fd --self-test

The verifier intentionally reads only analysis manifests and their CSV telemetry.
It does not import notebooks, figures, result interpretations, or experiment code.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REQUIRED_MANIFEST_FIELDS = (
    "algorithm",
    "dataset",
    "experiment_group",
    "algorithm_config",
    "variant",
    "phase",
    "alpha",
    "formulation",
    "seed",
    "hydra_output",
    "integrity_status",
    "source_config_sha256",
)
REQUIRED_CSV_FIELDS = (
    "round",
    "test/accuracy",
    "communication/round_bytes",
    "communication/cumulative_bytes",
    "communication/cumulative_mb",
)
ALLOWED_INTEGRITY_STATUSES = {"verified", "derived_exact_dedupe", "pending_rerun"}
SECONDARY_FIELD = "communication/round_secondary_bytes"


@dataclass(frozen=True)
class VerificationResult:
    checked_manifests: int
    secondary_column_present: int
    errors: tuple[str, ...]


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError("manifest must contain a mapping")
    return loaded


def _resolve_manifest_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("path value must be a non-empty string")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("path must be relative to its manifest")
    return manifest_path.parent / candidate


def _finite_nonnegative(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, found {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative, found {value!r}")
    return number


def _require_fields(values: Mapping[str, Any], fields: Iterable[str], kind: str) -> None:
    missing = [field for field in fields if field not in values]
    if missing:
        raise ValueError(f"missing required {kind} field(s): {', '.join(missing)}")


def _verify_csv(csv_path: Path, manifest: Mapping[str, Any]) -> bool:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers:
            raise ValueError("CSV has no header")
        if len(headers) != len(set(headers)):
            raise ValueError("CSV has duplicate headers")
        _require_fields({header: None for header in headers}, REQUIRED_CSV_FIELDS, "CSV")
        has_secondary = SECONDARY_FIELD in headers
        is_dadaquant = str(manifest["algorithm_config"]).lower() == "dadaquant"

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CSV row {line_number} has more values than headers")
            round_value = _finite_nonnegative(row["round"], f"row {line_number} round")
            if not round_value.is_integer():
                raise ValueError(f"row {line_number} round must be an integer")
            accuracy = _finite_nonnegative(row["test/accuracy"], f"row {line_number} test/accuracy")
            if accuracy > 1:
                raise ValueError(f"row {line_number} test/accuracy must be within [0, 1]")

            for field, value in row.items():
                if value in (None, ""):
                    continue
                if field.endswith("_bytes") or field.endswith("_time_sec"):
                    _finite_nonnegative(value, f"row {line_number} {field}")

            cumulative_bytes = _finite_nonnegative(
                row["communication/cumulative_bytes"],
                f"row {line_number} communication/cumulative_bytes",
            )
            cumulative_mb = _finite_nonnegative(
                row["communication/cumulative_mb"],
                f"row {line_number} communication/cumulative_mb",
            )
            expected_mb = cumulative_bytes / (1024**2)
            if not math.isclose(cumulative_mb, expected_mb, rel_tol=0, abs_tol=1e-9):
                raise ValueError(
                    f"row {line_number} communication/cumulative_mb must equal "
                    "communication/cumulative_bytes / 1024^2"
                )

            secondary = row.get(SECONDARY_FIELD, "")
            if secondary not in (None, ""):
                _finite_nonnegative(secondary, f"row {line_number} {SECONDARY_FIELD}")
                if not is_dadaquant:
                    raise ValueError(
                        f"row {line_number} reports {SECONDARY_FIELD} for non-DAdaQuant run"
                    )
    return has_secondary


def _manifest_csv_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    hydra_output = _resolve_manifest_path(manifest_path, manifest["hydra_output"])
    telemetry_path = manifest.get("telemetry_path")
    return (
        _resolve_manifest_path(manifest_path, telemetry_path)
        if telemetry_path is not None
        else hydra_output / "experiment_log.csv"
    )


def verify(analyses_root: Path) -> VerificationResult:
    manifests_dir = analyses_root / "data" / "manifests"
    errors: list[str] = []
    secondary_column_present = 0
    manifest_paths = sorted(manifests_dir.glob("*.yaml"))
    if not manifest_paths:
        return VerificationResult(0, 0, (f"no manifests found at {manifests_dir}",))

    for manifest_path in manifest_paths:
        try:
            manifest = _load_manifest(manifest_path)
            _require_fields(manifest, REQUIRED_MANIFEST_FIELDS, "manifest")
            if manifest["integrity_status"] not in ALLOWED_INTEGRITY_STATUSES:
                raise ValueError(f"unrecognized integrity_status {manifest['integrity_status']!r}")
            csv_path = _manifest_csv_path(manifest_path, manifest)
            if not csv_path.is_file():
                raise ValueError(f"telemetry CSV does not exist: {csv_path}")
            secondary_column_present += _verify_csv(csv_path, manifest)
        except (OSError, ValueError) as exc:
            errors.append(f"{manifest_path.name}: {exc}")
    return VerificationResult(len(manifest_paths), secondary_column_present, tuple(errors))


def _current_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _first_manifest_and_csv(analyses_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    for manifest_path in sorted((analyses_root / "data" / "manifests").glob("*.yaml")):
        manifest = _load_manifest(manifest_path)
        csv_path = _manifest_csv_path(manifest_path, manifest)
        if csv_path.is_file():
            return manifest_path, csv_path, manifest
    raise ValueError("no manifest with a telemetry CSV found")


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _expect_rejection(csv_path: Path, manifest: Mapping[str, Any]) -> None:
    try:
        _verify_csv(csv_path, manifest)
    except ValueError:
        return
    raise AssertionError("mutation was accepted")


def self_test(analyses_root: Path) -> None:
    _, source_csv, manifest = _first_manifest_and_csv(analyses_root)
    with tempfile.TemporaryDirectory(prefix="fedmaq-gate-5-") as temporary:
        temporary_path = Path(temporary)
        copied_csv = temporary_path / "experiment_log.csv"
        shutil.copyfile(source_csv, copied_csv)
        _verify_csv(copied_csv, manifest)
        try:
            _resolve_manifest_path(temporary_path / "manifest.yaml", str(copied_csv.resolve()))
        except ValueError:
            pass
        else:
            raise AssertionError("absolute telemetry path was accepted")

        with copied_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            rows = list(reader)

        renamed_headers = ["round_renamed" if header == "round" else header for header in headers]
        renamed_rows = []
        for row in rows:
            renamed_row = dict(row)
            renamed_row["round_renamed"] = renamed_row.pop("round")
            renamed_rows.append(renamed_row)
        renamed_csv = temporary_path / "renamed.csv"
        _write_csv(renamed_csv, renamed_headers, renamed_rows)
        _expect_rejection(renamed_csv, manifest)

        wrong_mb_rows = [dict(row) for row in rows]
        wrong_mb_rows[0]["communication/cumulative_mb"] = "1"
        wrong_mb_csv = temporary_path / "wrong_mb.csv"
        _write_csv(wrong_mb_csv, headers, wrong_mb_rows)
        _expect_rejection(wrong_mb_csv, manifest)

        secondary_rows = [dict(row) for row in rows]
        for row in secondary_rows:
            row[SECONDARY_FIELD] = "0"
        secondary_csv = temporary_path / "secondary_zero.csv"
        _write_csv(secondary_csv, [*headers, SECONDARY_FIELD], secondary_rows)
        _expect_rejection(secondary_csv, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses-root", type=Path, required=True)
    parser.add_argument("--require-revision")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    analyses_root = args.analyses_root.resolve()

    if args.require_revision:
        try:
            actual_revision = _current_revision(analyses_root)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"cannot resolve analysis revision: {exc}", file=sys.stderr)
            return 2
        if actual_revision != args.require_revision:
            print(
                "analysis revision mismatch: "
                f"expected {args.require_revision}, found {actual_revision}",
                file=sys.stderr,
            )
            return 2

    result = verify(analyses_root)
    print(f"checked_manifests={result.checked_manifests}")
    print(f"secondary_column_present={result.secondary_column_present}")
    print(f"errors={len(result.errors)}")
    for error in result.errors:
        print(f"  {error}", file=sys.stderr)
    if result.errors:
        return 1
    if args.self_test:
        try:
            self_test(analyses_root)
        except (AssertionError, OSError, ValueError) as exc:
            print(f"self_test_failed: {exc}", file=sys.stderr)
            return 1
        print("positive_current_record=yes")
        print("negative_renamed_required_field=rejected")
        print("negative_wrong_mb_unit=rejected")
        print("negative_absent_secondary_to_zero=rejected")
        print("negative_absolute_telemetry_path=rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
