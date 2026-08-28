from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_study1_artifacts import build_report, config_sha256

GROUPS = {
    "benchmark_grid": "cifar10|benchmark_grid|fedmaq||a0.1|f2|s0",
    "formulation_study": "cifar10|formulation_study|fedmaq|f2|a0.1|f2|s0",
    "ablation": "cifar10|ablation|fedmaq_no_kd||a0.1|f2|s0",
    "uniform_memory_control": "cifar10|uniform_memory_control|fedmaq||a0.1|f2|s0",
}


def write_expected(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "groups": {
                    group: {
                        "runs": [identity],
                        "count": 1,
                        "regimes": {"fedmaq": {"total_rounds": [100]}},
                    }
                    for group, identity in GROUPS.items()
                }
            }
        ),
        encoding="utf-8",
    )


def write_run(root: Path, group: str, algorithm: str, variant: str = "") -> None:
    algorithm_segment = f"{algorithm}__{variant}" if variant else algorithm
    job_dir = (
        root
        / "outputs"
        / ("explore" if group == "formulation_study" else "formal")
        / "cifar10_mobilenetv2"
        / group
        / algorithm_segment
        / "dirichlet_alpha_0.1"
        / "seed_0"
    )
    job_dir.mkdir(parents=True)
    config = {
        "_algorithm_config_name": algorithm,
        "algorithm": {"name": "fedmaq", "formulation": 2},
        "dataset": {"name": "cifar10"},
        "heterogeneity": {"alpha": 0.1},
        "experiment_group": group,
        "experiment": {"total_rounds": 100},
        "seed": 0,
    }
    manifest = {
        "config_sha256": config_sha256(config),
        "run": {
            "algorithm": "fedmaq",
            "algorithm_config": algorithm,
            "dataset": "cifar10",
            "alpha": 0.1,
            "seed": 0,
            "total_rounds": 100,
        },
        "git": {"commit": "a" * 40, "branch": "main", "tag": None, "dirty": False},
        "config": config,
    }
    (job_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (job_dir / "final_global_model.pt").write_bytes(b"checkpoint")
    hydra_dir = job_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (hydra_dir / "hydra.yaml").write_text("hydra: {}\n", encoding="utf-8")
    (hydra_dir / "overrides.yaml").write_text("[]\n", encoding="utf-8")
    fieldnames = [
        "round",
        "test/loss",
        "test/accuracy",
        "communication/round_bytes",
        "communication/cumulative_bytes",
        "communication/cumulative_mb",
        "system/round_time_sec",
        "system/cumulative_time_sec",
        "system/client_sim_time_sec",
        "system/cumulative_client_time_sec",
        "system/server_sim_time_sec",
        "system/cumulative_server_time_sec",
        "system/wall_time_sec",
        "system/cumulative_wall_time_sec",
    ]
    jsonl_rows = []
    with (job_dir / "experiment_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for round_value in range(101):
            cumulative_bytes = round_value * 1024
            row = {
                "round": round_value,
                "test/loss": 1.0,
                "test/accuracy": 0.5,
                "communication/round_bytes": 0 if round_value == 0 else 1024,
                "communication/cumulative_bytes": cumulative_bytes,
                "communication/cumulative_mb": cumulative_bytes / (1024 * 1024),
                "system/round_time_sec": 1.0,
                "system/cumulative_time_sec": float(round_value),
                "system/client_sim_time_sec": 1.0,
                "system/cumulative_client_time_sec": float(round_value),
                "system/server_sim_time_sec": 1.0,
                "system/cumulative_server_time_sec": float(round_value),
                "system/wall_time_sec": 1.0,
                "system/cumulative_wall_time_sec": float(round_value),
            }
            writer.writerow(row)
            jsonl_rows.append(json.dumps(row))
    (job_dir / "experiment_log.jsonl").write_text("\n".join(jsonl_rows) + "\n", encoding="utf-8")


def test_build_report_closes_a_complete_study1_fixture(tmp_path: Path) -> None:
    expected = tmp_path / "docs" / "freeze" / "expected_runs.json"
    write_expected(expected)
    for group in GROUPS:
        if group == "formulation_study":
            write_run(tmp_path, group, "fedmaq", "f2")
        elif group == "ablation":
            write_run(tmp_path, group, "fedmaq_no_kd")
        else:
            write_run(tmp_path, group, "fedmaq")

    report = build_report(tmp_path, expected)

    assert report["closure"]["expected_total"] == 4
    assert report["closure"]["strict_all_closed"] is True
    assert all(run["strict_valid"] for run in report["runs"])


def test_build_report_rejects_missing_checkpoint_and_short_csv(tmp_path: Path) -> None:
    expected = tmp_path / "docs" / "freeze" / "expected_runs.json"
    write_expected(expected)
    for group in GROUPS:
        if group == "formulation_study":
            write_run(tmp_path, group, "fedmaq", "f2")
        elif group == "ablation":
            write_run(tmp_path, group, "fedmaq_no_kd")
        else:
            write_run(tmp_path, group, "fedmaq")

    job_dir = next((tmp_path / "outputs").glob("**/benchmark_grid/**/seed_0"))
    (job_dir / "final_global_model.pt").unlink()
    rows = (job_dir / "experiment_log.csv").read_text(encoding="utf-8").splitlines()
    (job_dir / "experiment_log.csv").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    report = build_report(tmp_path, expected)
    benchmark = report["closure"]["groups"]["benchmark_grid"]

    assert report["closure"]["strict_all_closed"] is False
    assert benchmark["strict_closed"] is False
    violations = benchmark["invalid"][0]["violations"]
    assert "telemetry CSV failed validation" in violations
    assert "telemetry JSONL failed validation or CSV parity" in violations
    assert "final checkpoint missing or empty" in violations
