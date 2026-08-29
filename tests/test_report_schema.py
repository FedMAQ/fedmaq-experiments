from __future__ import annotations

import json

import pytest

from scripts.report_schema import (
    REPORT_SCHEMA_VERSION,
    ReportSchemaError,
    load_report,
    summary,
    summary_from_mapping,
    write_report,
)


def test_summary_has_one_canonical_dispersion_field_and_legacy_read_aliases():
    value = summary([1.0, 2.0, 3.0])

    assert set(value) == {"mean", "dispersion", "n"}
    assert value["mean"] == pytest.approx(2.0)
    assert value["dispersion"] == pytest.approx(1.0)
    assert value["sd"] == pytest.approx(1.0)
    assert value["sigma"] == pytest.approx(1.0)


@pytest.mark.parametrize("field", ["sd", "sigma"])
def test_summary_accepts_one_legacy_dispersion_name(field):
    parsed = summary_from_mapping({"mean": 0.5, field: 0.1, "n": 3})

    assert parsed.dispersion == pytest.approx(0.1)
    assert parsed.to_dict()["dispersion"] == pytest.approx(0.1)


def test_summary_rejects_conflicting_or_incomplete_fields():
    with pytest.raises(ReportSchemaError, match="conflicting"):
        summary_from_mapping({"mean": 0.5, "dispersion": 0.1, "sigma": 0.2, "n": 3})
    with pytest.raises(ReportSchemaError, match="mean and n"):
        summary_from_mapping({"mean": 0.5, "dispersion": 0.1})
    with pytest.raises(ReportSchemaError, match="at least two"):
        summary_from_mapping({"mean": 0.5, "dispersion": 0.1, "n": 1})


def test_report_envelope_round_trips_and_legacy_artifacts_are_explicitly_supported(tmp_path):
    path = tmp_path / "winner.json"
    write_report(path, "formulation_winner", {"cell": {"score": summary([1.0, 2.0])}})

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == REPORT_SCHEMA_VERSION
    assert document["report_type"] == "formulation_winner"
    assert load_report(path, expected_type="formulation_winner")["cell"]["score"]["dispersion"]

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"cell": {"score": {"mean": 1.0, "sd": 0.2, "n": 2}}}),
        encoding="utf-8",
    )
    loaded = load_report(legacy)
    assert "dispersion" in loaded["cell"]["score"]
    assert loaded["cell"]["score"]["sigma"] == pytest.approx(0.2)
    assert summary_from_mapping(loaded["cell"]["score"]).dispersion == pytest.approx(0.2)


def test_report_loader_rejects_malformed_canonical_documents(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "data": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ReportSchemaError, match="report_type"):
        load_report(path)

    with pytest.raises(ReportSchemaError, match="requires dispersion"):
        write_report(tmp_path / "incomplete.json", "test", {"summary": {"mean": 1.0, "n": 2}})
