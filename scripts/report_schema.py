"""Canonical schemas for analysis reports and seed summaries.

Analysis functions keep returning mappings so existing notebooks and scripts can
read them without an adapter.  This module owns the shape of shared summaries and
the on-disk envelope used by newly written artifacts.  ``load_report`` is the
explicit compatibility boundary for the bare JSON documents written before the
envelope existed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 1


class ReportSchemaError(ValueError):
    """Raised when a report or shared summary cannot be interpreted safely."""


class SummaryDict(dict[str, Any]):
    """Canonical summary mapping with read-only aliases for old callers.

    New serialization exposes only ``dispersion``.  ``sd`` and ``sigma`` remain
    readable in memory so notebooks built against pre-schema reports continue to
    work while callers migrate.  They are intentionally not materialized as
    duplicate keys in new JSON.
    """

    _ALIASES = {"sd": "dispersion", "sigma": "dispersion"}

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(self._ALIASES.get(key, key))

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(self._ALIASES.get(key, key), default)


@dataclass(frozen=True)
class Summary:
    """Mean, sample dispersion, and sample count for one reported quantity."""

    mean: float | None
    dispersion: float | None
    n: int

    @classmethod
    def from_values(cls, values: list[float]) -> Summary:
        if not values:
            return cls(mean=None, dispersion=None, n=0)
        mean = sum(values) / len(values)
        dispersion = None
        if len(values) >= 2:
            centered = [(value - mean) ** 2 for value in values]
            dispersion = math.sqrt(sum(centered) / (len(values) - 1))
        return cls(mean=mean, dispersion=dispersion, n=len(values))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str = "summary") -> Summary:
        if "mean" not in value or "n" not in value:
            raise ReportSchemaError(f"{path} must contain mean and n")

        aliases = [key for key in ("dispersion", "sd", "sigma") if key in value]
        if len(aliases) > 1:
            values = {value[key] for key in aliases}
            if len(values) > 1:
                raise ReportSchemaError(
                    f"{path} contains conflicting dispersion fields: {aliases}"
                )
        dispersion_key = aliases[0] if aliases else None
        mean = _number_or_none(value["mean"], f"{path}.mean")
        dispersion = (
            _number_or_none(value[dispersion_key], f"{path}.{dispersion_key}")
            if dispersion_key
            else None
        )
        n = value["n"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ReportSchemaError(f"{path}.n must be a non-negative integer")
        if n == 0 and (mean is not None or dispersion is not None):
            raise ReportSchemaError(f"{path} with n=0 must have null mean and dispersion")
        if n > 0 and mean is None:
            raise ReportSchemaError(f"{path} with observations must have a mean")
        if n >= 2 and dispersion_key is None:
            raise ReportSchemaError(f"{path} with at least two observations requires dispersion")
        if n < 2 and dispersion is not None:
            raise ReportSchemaError(f"{path}.dispersion requires at least two observations")
        return cls(mean=mean, dispersion=dispersion, n=n)

    def to_dict(self) -> SummaryDict:
        return SummaryDict(
            mean=self.mean,
            dispersion=self.dispersion,
            n=self.n,
        )


def _number_or_none(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportSchemaError(f"{path} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ReportSchemaError(f"{path} must be a finite number or null")
    return number


def summary(values: list[float]) -> SummaryDict:
    """Build the canonical JSON-ready summary for a sequence of observations."""

    return Summary.from_values(values).to_dict()


def summary_from_mapping(value: Mapping[str, Any], *, path: str = "summary") -> Summary:
    """Validate either a canonical or pre-schema summary mapping."""

    return Summary.from_mapping(value, path=path)


def validate_report_document(
    document: Mapping[str, Any], *, expected_type: str | None = None
) -> dict[str, Any]:
    """Validate a canonical envelope and return its payload.

    A mapping without ``schema_version`` is a legacy bare report and is returned
    unchanged.  This is deliberately the only compatibility concession; malformed
    canonical envelopes fail instead of being treated as an older artifact.
    """

    if "schema_version" not in document:
        if not document:
            raise ReportSchemaError("legacy report must not be empty")
        return _normalize_payload(document)
    if document.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReportSchemaError(
            f"unsupported report schema version {document.get('schema_version')!r}"
        )
    report_type = document.get("report_type")
    if not isinstance(report_type, str) or not report_type:
        raise ReportSchemaError("canonical report requires a non-empty report_type")
    if expected_type is not None and report_type != expected_type:
        raise ReportSchemaError(
            f"expected report_type {expected_type!r}, got {report_type!r}"
        )
    payload = document.get("data")
    if not isinstance(payload, Mapping) or not payload:
        raise ReportSchemaError("canonical report requires a non-empty data mapping")
    return _normalize_payload(payload)


def _normalize_payload(value: Any, *, path: str = "data") -> Any:
    if isinstance(value, Mapping):
        summary_keys = {"mean", "n", "dispersion", "sd", "sigma"}
        leaf = path.rsplit(".", maxsplit=1)[-1]
        is_summary = set(value).issubset(summary_keys) and "mean" in value and "n" in value
        is_named_summary = leaf in {
            "summary",
            "reference",
            "accuracy_r100",
            "accuracy_at_budget",
            "cumulative_mb",
            "crossing_cumulative_mb",
            "delta",
        }
        if is_summary or (is_named_summary and "mean" in value and "n" in value):
            return summary_from_mapping(value, path=path).to_dict()
        return {
            str(key): _normalize_payload(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_payload(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def write_report(path: Path, report_type: str, data: Mapping[str, Any]) -> None:
    """Write a versioned report envelope after validating its outer shape."""

    if not report_type:
        raise ReportSchemaError("report_type must not be empty")
    normalized = _normalize_payload(data)
    document = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": report_type,
        "data": normalized,
    }
    validate_report_document(document, expected_type=report_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_legacy_report(path: Path, data: Mapping[str, Any]) -> None:
    """Preserve the bare artifact envelope used by the v1 analysis command."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_report(path: Path, *, expected_type: str | None = None) -> dict[str, Any]:
    """Load a canonical report or an older bare JSON artifact."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportSchemaError(f"could not read report {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ReportSchemaError(f"report {path} must contain a JSON object")
    return validate_report_document(document, expected_type=expected_type)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportSchemaError",
    "Summary",
    "SummaryDict",
    "load_report",
    "summary",
    "summary_from_mapping",
    "validate_report_document",
    "write_legacy_report",
    "write_report",
]
