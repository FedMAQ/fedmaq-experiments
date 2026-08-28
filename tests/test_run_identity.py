from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fedmaq.core.manifest import config_sha256 as manifest_config_sha256
from fedmaq.core.run_identity import (
    config_sha256,
    get_canonical_output_dir,
    identity_key,
    parse_run_directory,
)
from scripts.analysis import GRID_GROUP, RunRecord, closure_certificate
from scripts.audit_study1_artifacts import run_identity_from_artifacts
from scripts.common import expand_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "matrix_path",
    sorted((REPO_ROOT / "conf" / "matrix").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_every_matrix_task_round_trips_through_the_canonical_run_parser(matrix_path: Path):
    matrix = OmegaConf.to_container(OmegaConf.load(matrix_path), resolve=True)

    for task in expand_matrix(matrix, matrix_path.stem):
        built = get_canonical_output_dir(
            phase=task["phase"],
            dataset=task["dataset"],
            model=task["model"],
            exp_group=task["experiment_group"],
            algorithm=task["algorithm_config"],
            heterogeneity=task["heterogeneity"],
            seed=task["seed"],
            variant=task["variant"],
        )

        parsed = parse_run_directory(REPO_ROOT / built, REPO_ROOT)

        assert parsed is not None
        assert parsed.phase == task["phase"]
        assert parsed.dataset_model == f"{task['dataset']}_{task['model']}"
        assert parsed.experiment_group == task["experiment_group"]
        assert parsed.algorithm_path == task["algorithm_config"]
        assert parsed.variant == task["variant"]
        assert parsed.heterogeneity_path == task["heterogeneity"]
        assert parsed.seed == task["seed"]


def test_noncanonical_path_returns_no_run_directory():
    assert parse_run_directory(REPO_ROOT / "outputs" / "not-a-run", REPO_ROOT) is None


@pytest.mark.parametrize(
    "relative_path",
    [
        "outputs/explore/cifar10_mobilenetv2/group/algorithm__/dirichlet_alpha_0.1/seed_0",
        "outputs/explore/cifar10_mobilenetv2/group/algorithm/dirichlet_alpha_0.1/seed_01",
    ],
)
def test_parser_rejects_paths_the_builder_cannot_emit(relative_path: str):
    assert parse_run_directory(REPO_ROOT / relative_path, REPO_ROOT) is None


def test_auditor_and_canonical_identity_agree_on_string_formulation():
    path_fields = {
        "canonical": True,
        "phase": "explore",
        "dataset_model": "cifar10_mobilenetv2",
        "experiment_group": "power_mean_design",
        "algorithm_path": "power_mean",
        "algorithm_segment": "power_mean__p0",
        "variant": "p0",
        "heterogeneity_path": "dirichlet_alpha_0.1",
        "path_seed": 0,
    }
    config = {
        "algorithm": {"formulation": "power_mean"},
        "dataset": {"name": "cifar10"},
        "heterogeneity": {"alpha": 0.1},
        "seed": 0,
    }
    manifest = {
        "config": config,
        "run": {
            "dataset": "cifar10",
            "algorithm_config": "power_mean",
            "alpha": 0.1,
            "seed": 0,
        },
    }

    actual, fields = run_identity_from_artifacts(manifest, path_fields)
    expected = identity_key(
        "cifar10", "power_mean_design", "power_mean", "p0", 0.1, "power_mean", 0
    )

    assert actual == expected
    assert fields["formulation"] == "power_mean"
    assert manifest_config_sha256 is config_sha256


def _run_record(tmp_path: Path) -> RunRecord:
    csv_path = tmp_path / "experiment_log.csv"
    csv_path.write_text(
        "round,test/accuracy,communication/cumulative_mb\n1,0.5,1\n2,0.5,2\n",
        encoding="utf-8",
    )
    return RunRecord(
        job_dir=tmp_path / "outputs" / "malformed",
        dataset="cifar10",
        alpha=0.1,
        algorithm="fedmaq",
        formulation=2,
        seed=0,
        csv_path=csv_path,
        experiment_group=None,
    )


def test_noncanonical_output_run_closes_no_group_and_marks_certificate_open(tmp_path: Path):
    run = _run_record(tmp_path)
    expected_identity = identity_key("cifar10", GRID_GROUP, "fedmaq", "", 0.1, 2, 0)
    manifest = {
        GRID_GROUP: {
            "runs": [expected_identity],
            "regimes": {"fedmaq": {"total_rounds": [2]}},
        }
    }

    certificate = closure_certificate([run], manifest, groups=[GRID_GROUP])
    grid = certificate["groups"][GRID_GROUP]

    assert grid["non_canonical"] == [str(run.job_dir)]
    assert grid["closed"] is False
    assert certificate["all_closed"] is False


def test_legacy_multirun_run_is_not_a_noncanonical_output_defect(tmp_path: Path):
    run = _run_record(tmp_path)
    run.job_dir = tmp_path / "multirun" / "2026-08-28" / "12-00-00" / "0"
    manifest = {
        GRID_GROUP: {
            "runs": [],
            "regimes": {"fedmaq": {"total_rounds": [2]}},
        }
    }

    certificate = closure_certificate([run], manifest, groups=[GRID_GROUP])

    assert certificate["groups"][GRID_GROUP]["non_canonical"] == []


def test_nested_multirun_directory_under_outputs_is_not_legacy(tmp_path: Path):
    run = _run_record(tmp_path)
    run.job_dir = tmp_path / "outputs" / "nested" / "multirun" / "0"
    manifest = {
        GRID_GROUP: {
            "runs": [],
            "regimes": {"fedmaq": {"total_rounds": [2]}},
        }
    }

    certificate = closure_certificate([run], manifest, groups=[GRID_GROUP])

    assert certificate["groups"][GRID_GROUP]["non_canonical"] == [str(run.job_dir)]
