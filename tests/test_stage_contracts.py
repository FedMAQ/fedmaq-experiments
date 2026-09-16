"""Acceptance tests for Stage-A registration, ledgers, and assurance semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.golden_diff import repeatability_report, transition_diagnostic
from scripts.run_assurance_fixtures import run_analysis_fixture
from scripts.stage_ledgers import build_ledgers
from tests.run_fixtures import run_fedmaq_q_transition_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest(name: str) -> dict[str, object]:
    return json.loads((REPO_ROOT / "docs" / "recut" / name).read_text(encoding="utf-8"))


def _write_capture(
    directory: Path,
    *,
    commit: str = "abc123",
    dirty: bool = False,
    config: str = "config-hash",
    environment: dict[str, object] | None = None,
    accuracy: float = 0.5,
    wall_time: float = 1.0,
) -> None:
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "round": [1],
            "test/accuracy": [accuracy],
            "system/wall_time_sec": [wall_time],
        }
    ).to_csv(directory / "experiment_log.csv", index=False)
    manifest = {
        "config_sha256": config,
        "git": {"commit": commit, "dirty": dirty},
        "environment": environment or {"python": "3.13", "torch": "2.8"},
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_stage_a_manifest_is_exact_and_registered() -> None:
    manifest = _manifest("stage_a_manifest.json")
    cells = manifest["cells"]
    assert manifest["cell_count"] == 145
    assert len(cells) == 145
    assert len({cell["cell_id"] for cell in cells}) == 145
    assert manifest["registered_seeds"] == [0, 42, 123, 7, 21]
    assert manifest["registered_algorithms"] == [
        "dadaquant",
        "feddistill",
        "fedkd",
        "fedmaq",
        "fedpaq",
        "fedprox",
    ]
    assert {cell["split"] for cell in cells} == {"val"}
    assert {cell["ledger"] for cell in cells} == {"scientific"}
    for cell in cells:
        assert cell["hyperparameter"]["variant"] == cell["variant"]
        assert cell["hyperparameter"]["value"] is not None


def test_ledgers_have_required_counts_and_are_disjoint() -> None:
    ledgers = _manifest("stage_ledgers.json")
    scientific = ledgers["scientific"]
    stages = scientific["stages"]
    assert scientific["total_cell_count"] == 499
    assert {name: stages[name]["cell_count"] for name in stages} == {
        "matched_tuning": 145,
        "stage_1a": 168,
        "stage_1b": 12,
        "downstream": 174,
    }
    assurance = ledgers["assurance"]
    assert assurance["executions"] == [
        "golden_transition",
        "golden_repeatability",
        "concurrency",
    ]
    for name in assurance["executions"]:
        execution = assurance["execution_records"][name]
        assert execution["ledger"] == "assurance"
        assert execution["cell_count"] == assurance["total_matrix_cell_count"]
        assert execution["output_root"]
        assert execution["provenance"] == "run_manifest.json"
    scientific_ids = {cell_id for stage in stages.values() for cell_id in stage["cell_ids"]}
    assurance_ids = {
        cell_id for matrix in assurance["matrices"].values() for cell_id in matrix["cell_ids"]
    }
    assert scientific_ids.isdisjoint(assurance_ids)


def test_downstream_ledger_carries_the_selected_power_mean_identity() -> None:
    """Gate 2 must not label downstream FedMAQ rows as the retired formulation."""
    ledgers = build_ledgers()
    matched_tuning_ids = ledgers["scientific"]["stages"]["matched_tuning"]["cell_ids"]
    downstream_ids = ledgers["scientific"]["stages"]["downstream"]["cell_ids"]
    historical_fedmaq_ids = [cell_id for cell_id in matched_tuning_ids if "|fedmaq|" in cell_id]
    fedmaq_ids = [cell_id for cell_id in downstream_ids if "|fedmaq" in cell_id]

    assert historical_fedmaq_ids
    assert all("|f2|" in cell_id for cell_id in historical_fedmaq_ids)
    assert fedmaq_ids
    assert all("|fpower_mean|" in cell_id for cell_id in fedmaq_ids)


def test_transition_is_diagnostic_and_never_pass_eligible(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_capture(old, accuracy=0.5)
    _write_capture(new, accuracy=0.6, wall_time=2.0)
    report = transition_diagnostic(old, new, "fixture")
    assert report["status"] == "RECORDED"
    assert report["pass"] is False
    assert report["pass_eligible"] is False
    assert report["classified_differential"]["semantic_value_changes"]
    assert report["classified_differential"]["ignored_runtime_changes"]


def test_repeatability_requires_clean_provenance_and_exact_output(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_capture(first, wall_time=1.0)
    _write_capture(second, wall_time=2.0)
    report = repeatability_report(first, second, "fixture")
    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["classified_differential"]["semantic_value_changes"] == []
    assert report["classified_differential"]["ignored_runtime_changes"]

    (second / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "different",
                "git": {"commit": "abc123", "dirty": False},
                "environment": {"python": "3.13", "torch": "2.8"},
            }
        ),
        encoding="utf-8",
    )
    failed = repeatability_report(first, second, "fixture")
    assert failed["status"] == "FAIL"
    assert failed["pass"] is False
    assert "config_sha256 differs or is missing" in failed["reasons"]


def test_assurance_fixture_records_real_q_transition(tmp_path: Path) -> None:
    fixture = run_fedmaq_q_transition_fixture(tmp_path)
    records = fixture["records"]
    assert [record["q"] for record in records] == [8, 8, 4]
    assert [record["is_diff"] for record in records] == [False, True, False]
    assert records[0]["bit_width"] == 8
    assert records[1]["bit_width"] == 9
    assert records[2]["bit_width"] == 4


def test_analysis_fixture_uses_canonical_report_schema(tmp_path: Path) -> None:
    report = run_analysis_fixture(tmp_path)
    assert report["accuracy_at_budget"]["mean"] == 0.5
    assert report["accuracy_at_budget"]["n"] == 1
