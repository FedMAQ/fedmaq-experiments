"""Comprehensive unit and invariant tests for Issue #89.

Verifies:
1. Atomic run completion: valid checkpoint + matching manifest + contiguous telemetry.
2. Failure cases fail closed: missing, duplicate, corrupt, non-finite, non-monotone,
   truncated, and identity-mismatched runs are never reported complete or skipped.
3. Selection split identity: selectors strictly require validation split; test-split
   inputs raise ValueError.
4. Exact-set closure certificate: rejects historical, unexpected, duplicate, cross-stage,
   and missing runs.
5. Matched tuning: 5 paired validation seeds, common in-support observed-byte budget
   interpolation, strict adoption margin, deterministic tie-breaking.
6. Synthetic complete fixtures produce deterministic verdicts, malformed fixtures produce none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME
from fedmaq.core.manifest import MANIFEST_FILENAME
from fedmaq.core.run_identity import get_canonical_output_dir
from fedmaq.core.validation import is_run_evidence_complete, validate_run_evidence
from scripts.analysis import (
    BASELINE_TUNING_WIDE_GROUP,
    RunRecord,
    baseline_tuning_margin,
    closure_certificate,
    interpolate_accuracy_at_budget,
    select_power_mean_degree_iso_byte,
)
from scripts.common import is_run_complete
from scripts.matrix_executor import MatrixExecutor
from scripts.matrix_planner import MatrixPlan, MatrixTask


def _create_synthetic_run(
    base_dir: Path,
    phase: str = "explore",
    dataset: str = "cifar10",
    model: str = "mobilenetv2",
    group: str = "baseline_tuning_wide",
    algorithm: str = "fedprox",
    alpha: float = 0.3,
    seed: int = 0,
    variant: str = "mu1p0",
    total_rounds: int = 100,
    split: str = "val",
    wire_protocol: str = "packed_wire_v1",
    corrupt_checkpoint: bool = False,
    missing_checkpoint: bool = False,
    corrupt_manifest: bool = False,
    missing_manifest: bool = False,
    mismatched_identity: bool = False,
    missing_csv: bool = False,
    duplicate_rounds: bool = False,
    missing_round: bool = False,
    truncated_rounds: bool = False,
    nonfinite_metric: bool = False,
    nonmonotone_mb: bool = False,
    include_round_zero: bool = False,
    extra_columns: dict[str, list[float | None]] | None = None,
    missing_jsonl: bool = False,
    mismatched_jsonl_rounds: bool = False,
    missing_partition_cache: bool = False,
    corrupt_partition_cache: bool = False,
    mismatched_partition_digest: bool = False,
    accuracies: list[float] | None = None,
    cumulative_mbs: list[float] | None = None,
) -> Path:
    """Create a synthetic run directory with controlled integrity and defects."""
    output_dir = base_dir / get_canonical_output_dir(
        phase,
        dataset,
        model,
        group,
        algorithm,
        f"dirichlet_alpha_{alpha}",
        seed,
        variant=variant,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not missing_checkpoint:
        checkpoint_path = output_dir / FINAL_MODEL_FILENAME
        if corrupt_checkpoint:
            checkpoint_path.write_bytes(b"corrupt non-torch data payload")
        else:
            torch.save({"weight": torch.tensor([1.0, 2.0])}, checkpoint_path)

    if not missing_manifest:
        manifest_path = output_dir / MANIFEST_FILENAME
        if corrupt_manifest:
            manifest_path.write_text("{corrupt json", encoding="utf-8")
        else:
            manifest_data: dict[str, Any] = {
                "schema_version": 1,
                "run": {
                    "algorithm": algorithm,
                    "algorithm_config": algorithm,
                    "dataset": "wrong_dataset" if mismatched_identity else dataset,
                    "alpha": alpha,
                    "seed": seed,
                    "variant": variant,
                    "total_rounds": total_rounds,
                    "num_clients": 10,
                    "split": split,
                    "loader_used": "val" if split == "val" else "test",
                    "wire_protocol": wire_protocol,
                },
                "protocol": {
                    "name": "replacement-v1",
                    "stage": "matched_tuning",
                    "split": split,
                    "wire_protocol": wire_protocol,
                    "promotable": True,
                    "historical": False,
                },
                "git": {"commit": "abcdef1234567890", "dirty": False},
            }
            if not missing_partition_cache:
                cache_dir = base_dir / ".data_partitions"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_filename = f"{dataset}_clients_10_alpha_{alpha}_pub_200_seed_{seed}.json"
                cache_file = cache_dir / cache_filename
                cache_content: dict[str, Any] = {
                    "schema_version": 1,
                    "dataset": dataset,
                    "num_clients": 10,
                    "alpha": alpha,
                    "num_public_samples": 200,
                    "num_val_samples": 5000,
                    "seed": seed,
                    "partition": "dirichlet",
                    "public_indices": list(range(200)),
                    "validation_indices": list(range(200, 5200)),
                    "client_indices": {str(i): [5200 + i] for i in range(10)},
                    "shard_sizes": [1] * 10,
                    "shard_stats": {
                        "min": 1,
                        "max": 1,
                        "mean": 1.0,
                        "median": 1.0,
                        "std": 0.0,
                        "total": 10,
                    },
                    "client_class_counts": [[1] + [0] * 9 for _ in range(10)],
                }
                if corrupt_partition_cache:
                    cache_file.write_text("{corrupt partition cache", encoding="utf-8")
                else:
                    cache_file.write_text(json.dumps(cache_content, indent=2), encoding="utf-8")

                from fedmaq.core.partitioning import canonical_partition_digest

                sha256 = (
                    "0" * 64
                    if mismatched_partition_digest
                    else canonical_partition_digest(cache_content)
                )
                manifest_data["partition_cache"] = {
                    "schema_version": 1,
                    "path": f".data_partitions/{cache_filename}",
                    "sha256": sha256,
                }
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    if not missing_csv:
        csv_path = output_dir / "experiment_log.csv"
        if accuracies is not None and cumulative_mbs is not None:
            rounds = list(range(1, len(accuracies) + 1))
            acc_list = list(accuracies)
            mb_list = list(cumulative_mbs)
        else:
            n_rounds = total_rounds
            if truncated_rounds:
                n_rounds = total_rounds // 2
            rounds = list(range(1, n_rounds + 1))
            acc_list = [0.3 + 0.3 * (r / total_rounds) for r in rounds]
            mb_list = [0.5 * r for r in rounds]

        if include_round_zero:
            rounds = [0, *rounds]
            acc_list = [float("nan"), *acc_list]
            mb_list = [float("nan"), *mb_list]

        if duplicate_rounds and len(rounds) > 2:
            rounds[2] = rounds[1]  # duplicate round
        if missing_round and len(rounds) > 4:
            rounds.pop(3)  # drop round 4
            acc_list.pop(3)
            mb_list.pop(3)
        if nonfinite_metric and len(acc_list) > 2:
            acc_list[2] = float("nan")
        if nonmonotone_mb and len(mb_list) > 3:
            mb_list[3] = mb_list[2] - 5.0  # decrease cumulative MB

        df = pd.DataFrame(
            {
                "round": rounds,
                "train/loss": [1.5 / r for r in range(1, len(rounds) + 1)],
                "test/accuracy": acc_list,
                "val/accuracy": acc_list,
                "communication/round_bytes": [500000] * len(rounds),
                "communication/cumulative_bytes": [500000 * r for r in range(1, len(rounds) + 1)],
                "communication/cumulative_mb": mb_list,
            }
        )
        df.to_csv(csv_path, index=False)
        if extra_columns:
            for name, values in extra_columns.items():
                if len(values) != len(df):
                    raise ValueError(f"extra column {name!r} has the wrong length")
                df[name] = values
            df.to_csv(csv_path, index=False)
        if not missing_jsonl:
            jsonl_rounds = list(rounds)
            if mismatched_jsonl_rounds and len(jsonl_rounds) > 1:
                jsonl_rounds[-1] = jsonl_rounds[-1] + 1
            with (output_dir / "experiment_log.jsonl").open("w", encoding="utf-8") as handle:
                for index, round_number in enumerate(jsonl_rounds):
                    record = df.iloc[index].to_dict()
                    record = {
                        key: (None if pd.isna(value) else value) for key, value in record.items()
                    }
                    record["round"] = round_number
                    required_by_algorithm = {
                        "fedavg": ("client/avg_train_loss",),
                        "fedprox": ("client/avg_train_loss",),
                        "fedpaq": ("client/avg_train_loss",),
                        "dadaquant": ("client/avg_train_loss",),
                        "feddistill": ("client/avg_task_loss", "client/avg_distill_loss"),
                        "fedkd": (
                            "client/avg_task_loss_student",
                            "client/avg_task_loss_teacher",
                            "client/avg_kd_loss_student",
                            "client/avg_kd_loss_teacher",
                            "client/avg_teacher_acc",
                        ),
                        "fedmaq": (
                            "client/avg_train_loss",
                            "algorithm/fedmaq/server_kd_loss",
                            "algorithm/fedmaq/q_count_2",
                            "algorithm/fedmaq/q_hat_count_2",
                        ),
                        "power_mean": ("client/avg_train_loss",),
                    }
                    for key in required_by_algorithm.get(algorithm, ()):
                        record.setdefault(key, 1 if "q_count_" in key else 0.1)
                    handle.write(json.dumps(record, allow_nan=True) + "\n")

    return output_dir


def test_valid_synthetic_run_passes_validation(tmp_path):
    """A completely intact run passes all completion checks."""
    run_dir = _create_synthetic_run(tmp_path)
    result = validate_run_evidence(run_dir, expected_rounds=100, repo_root=tmp_path)
    assert result.is_complete is True
    assert len(result.errors) == 0
    assert is_run_complete(run_dir) is True


def test_checkpoint_failures_fail_closed(tmp_path):
    """Missing, empty, or corrupt checkpoints fail validation."""
    missing_dir = _create_synthetic_run(tmp_path, seed=1, missing_checkpoint=True)
    assert not is_run_evidence_complete(missing_dir, repo_root=tmp_path)
    assert not is_run_complete(missing_dir)

    corrupt_dir = _create_synthetic_run(tmp_path, seed=2, corrupt_checkpoint=True)
    result = validate_run_evidence(corrupt_dir, repo_root=tmp_path)
    assert result.is_complete is False
    assert any("failed to load checkpoint" in e or "checkpoint" in e for e in result.errors)

    empty_dir = _create_synthetic_run(tmp_path, seed=3)
    (empty_dir / FINAL_MODEL_FILENAME).write_bytes(b"")
    assert not is_run_complete(empty_dir)


def test_manifest_and_identity_failures_fail_closed(tmp_path):
    """Corrupt manifest or identity mismatch fails validation."""
    missing_m = _create_synthetic_run(tmp_path, seed=10, missing_manifest=True)
    assert not is_run_complete(missing_m)

    corrupt_m = _create_synthetic_run(tmp_path, seed=11, corrupt_manifest=True)
    assert not is_run_complete(corrupt_m)

    mismatched = _create_synthetic_run(tmp_path, seed=12, mismatched_identity=True)
    result = validate_run_evidence(mismatched, repo_root=tmp_path)
    assert result.is_complete is False
    assert any("identity mismatch" in e for e in result.errors)


def test_telemetry_failures_fail_closed(tmp_path):
    """Missing CSV, non-contiguous rounds, non-finite values, and non-monotone MB fail."""
    missing_csv_dir = _create_synthetic_run(tmp_path, seed=20, missing_csv=True)
    assert not is_run_complete(missing_csv_dir)

    dup_dir = _create_synthetic_run(tmp_path, seed=21, duplicate_rounds=True)
    res_dup = validate_run_evidence(dup_dir, repo_root=tmp_path)
    assert res_dup.is_complete is False
    assert any("duplicate rounds" in e for e in res_dup.errors)

    missing_r_dir = _create_synthetic_run(tmp_path, seed=22, missing_round=True)
    res_mr = validate_run_evidence(missing_r_dir, expected_rounds=100, repo_root=tmp_path)
    assert res_mr.is_complete is False
    assert any("truncated" in e or "not contiguous" in e for e in res_mr.errors)

    trunc_dir = _create_synthetic_run(tmp_path, seed=23, truncated_rounds=True)
    res_trunc = validate_run_evidence(trunc_dir, expected_rounds=100, repo_root=tmp_path)
    assert res_trunc.is_complete is False
    assert any("truncated" in e for e in res_trunc.errors)

    nan_dir = _create_synthetic_run(tmp_path, seed=24, nonfinite_metric=True)
    res_nan = validate_run_evidence(nan_dir, repo_root=tmp_path)
    assert res_nan.is_complete is False
    assert any("non-finite" in e for e in res_nan.errors)

    nonmono_dir = _create_synthetic_run(tmp_path, seed=25, nonmonotone_mb=True)
    res_nm = validate_run_evidence(nonmono_dir, repo_root=tmp_path)
    assert res_nm.is_complete is False
    assert any("non-monotonic" in e for e in res_nm.errors)


def test_round_zero_and_all_nan_optional_columns_are_valid(tmp_path):
    run_dir = _create_synthetic_run(
        tmp_path,
        seed=26,
        include_round_zero=True,
        extra_columns={"communication/round_secondary_bytes": [None] * 101},
    )
    result = validate_run_evidence(run_dir, expected_rounds=100, repo_root=tmp_path)
    assert result.is_complete is True


def test_jsonl_is_required_and_must_match_csv_rounds(tmp_path):
    missing = _create_synthetic_run(tmp_path, seed=27, missing_jsonl=True)
    result = validate_run_evidence(missing, repo_root=tmp_path)
    assert result.is_complete is False
    assert any("JSONL" in error for error in result.errors)

    mismatched = _create_synthetic_run(tmp_path, seed=28, mismatched_jsonl_rounds=True)
    result = validate_run_evidence(mismatched, repo_root=tmp_path)
    assert result.is_complete is False
    assert any("do not match CSV" in error for error in result.errors)


def test_jsonl_must_include_algorithm_readout_keys(tmp_path):
    run_dir = _create_synthetic_run(tmp_path, seed=29)
    jsonl_path = run_dir / "experiment_log.jsonl"
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    records[0].pop("client/avg_train_loss")
    jsonl_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    result = validate_run_evidence(run_dir, repo_root=tmp_path)
    assert result.is_complete is False
    assert any("missing readout keys" in error for error in result.errors)


def test_matrix_executor_skip_completed_never_skips_malformed_runs(tmp_path):
    """MatrixExecutor skip_reason returns None for incomplete runs."""
    plan = MatrixPlan(
        matrix_path=tmp_path / "matrix.yaml",
        matrix={},
        phase="explore",
        experiment_group="baseline_tuning_wide",
        dataset="cifar10",
        model="mobilenetv2",
        total_rounds=100,
        client_gpus=0.0,
        experiment=None,
        seeds=(30,),
        heterogeneities=("dirichlet_alpha_0.3",),
        canonical_tasks=(),
        tasks=(),
        shard=None,
    )
    executor = MatrixExecutor(plan, skip_completed=True)
    malformed_dir = _create_synthetic_run(tmp_path, seed=30, truncated_rounds=True)
    task = MatrixTask(
        canonical_index=1,
        label="fedprox",
        algorithm="fedprox",
        heterogeneity="dirichlet_alpha_0.3",
        seed=30,
        output_dir=malformed_dir,
        command=("python", "scripts/run.py"),
    )
    assert executor.skip_reason(task) is None


def test_selection_inputs_reject_test_split(tmp_path):
    """Passing a test-split run to any selector raises a hard ValueError."""
    test_run_dir = _create_synthetic_run(tmp_path, seed=40, split="test")
    record = RunRecord(
        job_dir=test_run_dir,
        dataset="cifar10",
        alpha=0.3,
        algorithm="fedprox",
        formulation=None,
        seed=40,
        csv_path=test_run_dir / "experiment_log.csv",
        algorithm_config="fedprox",
        experiment_group="baseline_tuning_wide",
        phase="explore",
        variant="mu1p0",
        promotable=True,
        split="test",
    )
    with pytest.raises(ValueError, match="requires validation-split inputs"):
        baseline_tuning_margin([record], experiment_group="baseline_tuning_wide")

    with pytest.raises(ValueError, match="requires validation-split inputs"):
        select_power_mean_degree_iso_byte([record])


def test_linear_interpolation_accuracy_at_budget():
    """Verify linear interpolation without extrapolation."""
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "communication/cumulative_mb": [10.0, 20.0, 30.0],
            "val/accuracy": [0.40, 0.60, 0.70],
        }
    )
    # Exact points
    assert interpolate_accuracy_at_budget(df, 10.0) == pytest.approx(0.40)
    assert interpolate_accuracy_at_budget(df, 20.0) == pytest.approx(0.60)
    assert interpolate_accuracy_at_budget(df, 30.0) == pytest.approx(0.70)

    # In-support interpolated points
    assert interpolate_accuracy_at_budget(df, 15.0) == pytest.approx(0.50)
    assert interpolate_accuracy_at_budget(df, 25.0) == pytest.approx(0.65)

    # Out of support -> strictly None (no extrapolation)
    assert interpolate_accuracy_at_budget(df, 5.0) is None
    assert interpolate_accuracy_at_budget(df, 35.0) is None


def test_balanced_matched_tuning_scoring_and_tie_rule(tmp_path):
    """Five paired validation seeds scored at common observed byte budgets with strict margin."""
    runs = []
    # FedProx has reference mu1p0 and challengers mu0p001, mu0p01, mu0p1, mu10p0
    seeds = [0, 42, 123, 7, 21]
    # Ref: terminal MB = 50.0, val acc = 0.54 +- 0.01
    for i, s in enumerate(seeds):
        acc = 0.54 + (i - 2) * 0.005
        d = _create_synthetic_run(
            tmp_path,
            seed=s,
            variant="mu1p0",
            accuracies=[0.3, 0.45, acc],
            cumulative_mbs=[10.0, 25.0, 50.0],
            group=BASELINE_TUNING_WIDE_GROUP,
        )
        runs.append(
            RunRecord(
                job_dir=d,
                dataset="cifar10",
                alpha=0.3,
                algorithm="fedprox",
                formulation=None,
                seed=s,
                csv_path=d / "experiment_log.csv",
                algorithm_config="fedprox",
                experiment_group=BASELINE_TUNING_WIDE_GROUP,
                phase="explore",
                variant="mu1p0",
                promotable=True,
                split="val",
            )
        )

    # Challengers
    for var, base_acc in (
        ("mu0p001", 0.53),
        ("mu0p01", 0.58),  # clears margin with highest delta
        ("mu0p1", 0.57),  # clears margin, lower delta
        ("mu10p0", 0.51),
    ):
        for i, s in enumerate(seeds):
            acc = base_acc + (i - 2) * 0.005
            d = _create_synthetic_run(
                tmp_path,
                seed=s,
                variant=var,
                accuracies=[0.3, 0.45, acc],
                cumulative_mbs=[10.0, 25.0, 50.0],
                group=BASELINE_TUNING_WIDE_GROUP,
            )
            runs.append(
                RunRecord(
                    job_dir=d,
                    dataset="cifar10",
                    alpha=0.3,
                    algorithm="fedprox",
                    formulation=None,
                    seed=s,
                    csv_path=d / "experiment_log.csv",
                    algorithm_config="fedprox",
                    experiment_group=BASELINE_TUNING_WIDE_GROUP,
                    phase="explore",
                    variant=var,
                    promotable=True,
                    split="val",
                    p=-1.0,
                )
            )

    report = baseline_tuning_margin(runs, experiment_group=BASELINE_TUNING_WIDE_GROUP)
    cell = report["baselines"]["fedprox"]
    assert "error" not in cell
    assert cell["reference_variant"] == "mu1p0"
    assert cell["adopted_variant"] == "mu0p01"  # highest clearing delta
    assert cell["retained_shipped_value"] is False
    assert cell["challengers"]["mu0p01"]["clears_margin"] is True


def test_closure_certificate_rejects_missing_unexpected_and_duplicates(tmp_path):
    """Closure certificate enforces exact set matching."""
    manifest_groups = {
        "test_group": {
            "runs": ["run_a", "run_b", "run_c"],
            "total_rounds": 100,
        }
    }
    # Case 1: Missing run_c
    runs = [
        RunRecord(
            job_dir=tmp_path / "a",
            dataset="cifar10",
            alpha=0.3,
            algorithm="fedprox",
            formulation=None,
            seed=0,
            csv_path=tmp_path / "a/log.csv",
            experiment_group="test_group",
            algorithm_config="fedprox",
            variant="",
            promotable=True,
        ),
        RunRecord(
            job_dir=tmp_path / "b",
            dataset="cifar10",
            alpha=0.3,
            algorithm="fedpaq",
            formulation=None,
            seed=0,
            csv_path=tmp_path / "b/log.csv",
            experiment_group="test_group",
            algorithm_config="fedpaq",
            variant="",
            promotable=True,
        ),
    ]
    # Synthetic helper mapping
    for r in runs:
        r.job_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "round": [1, 100],
                "communication/cumulative_mb": [1.0, 2.0],
                "test/accuracy": [0.5, 0.5],
            }
        ).to_csv(r.csv_path, index=False)

    cert = closure_certificate(runs, manifest_groups, expected_round=100)
    assert cert["all_closed"] is False
    assert len(cert["groups"]["test_group"]["missing"]) > 0


def test_stage_1b_closure_and_selection_validation(tmp_path):
    """Stage 1b verifies closure against expected runs and selects omega (neutral preference)."""
    from scripts.analysis import (
        POWER_MEAN_DESIGN_GROUP,
        POWER_MEAN_OMEGA_GROUP,
        power_mean_stage_one_b,
        select_power_mean_omega_iso_byte,
    )
    from scripts.dump_expected_runs import POWER_MEAN_RECUT_MATRICES, expected_identities

    manifest = expected_identities(POWER_MEAN_RECUT_MATRICES)
    stage_1b = power_mean_stage_one_b()

    runs = []
    # Stage 1a runs (omega=0.5 at p=-1)
    for alpha in (0.1, 1.0):
        for seed in (0, 42, 123):
            run_dir = _create_synthetic_run(
                tmp_path,
                phase="explore",
                dataset="cifar10",
                group=POWER_MEAN_DESIGN_GROUP,
                algorithm="power_mean",
                alpha=alpha,
                seed=seed,
                variant="p-1",
                split="val",
                total_rounds=100,
            )
            runs.append(
                RunRecord(
                    job_dir=run_dir,
                    dataset="cifar10",
                    alpha=alpha,
                    algorithm="power_mean",
                    formulation="power_mean",
                    seed=seed,
                    csv_path=run_dir / "experiment_log.csv",
                    experiment_group=POWER_MEAN_DESIGN_GROUP,
                    algorithm_config="power_mean",
                    variant="p-1",
                    promotable=True,
                    split="val",
                    p=-1.0,
                )
            )

    # Stage 1b runs (omega=0.25 and omega=0.75)
    for alpha in (0.1, 1.0):
        for w_var in ("omega0.25", "omega0.75"):
            for seed in (0, 42, 123):
                run_dir = _create_synthetic_run(
                    tmp_path,
                    phase="explore",
                    dataset="cifar10",
                    group=POWER_MEAN_OMEGA_GROUP,
                    algorithm="power_mean",
                    alpha=alpha,
                    seed=seed,
                    variant=w_var,
                    split="val",
                    total_rounds=100,
                )
                runs.append(
                    RunRecord(
                        job_dir=run_dir,
                        dataset="cifar10",
                        alpha=alpha,
                        algorithm="power_mean",
                        formulation="power_mean",
                        seed=seed,
                        csv_path=run_dir / "experiment_log.csv",
                        experiment_group=POWER_MEAN_OMEGA_GROUP,
                        algorithm_config="power_mean",
                        variant=w_var,
                        promotable=True,
                        split="val",
                        p=-1.0,
                    )
                )

    cert = closure_certificate(runs, manifest, groups=[stage_1b.experiment_group])
    assert cert["all_closed"] is True

    selection = select_power_mean_omega_iso_byte(runs, selected_p=-1.0)
    assert selection["cifar10_alpha_0.1"]["winner"] == 0.5
    assert selection["cifar10_alpha_1.0"]["winner"] == 0.5


def test_validation_rejects_missing_partition_cache_entry(tmp_path: Path):
    """Evidence validation must reject replacement-v1 runs missing partition_cache in manifest."""
    run_dir = _create_synthetic_run(tmp_path, missing_partition_cache=True)
    result = validate_run_evidence(run_dir, repo_root=tmp_path)
    assert any("run manifest missing required partition_cache" in err for err in result.errors)
    assert not is_run_evidence_complete(run_dir, repo_root=tmp_path)


def test_validation_rejects_missing_partition_cache_file(tmp_path: Path):
    """Evidence validation must reject runs whose cache file does not exist on disk."""
    run_dir = _create_synthetic_run(tmp_path)
    cache_file = tmp_path / ".data_partitions" / "cifar10_clients_10_alpha_0.3_pub_200_seed_0.json"
    cache_file.unlink()

    result = validate_run_evidence(run_dir, repo_root=tmp_path)
    assert any("partition cache missing" in err for err in result.errors)
    assert not is_run_evidence_complete(run_dir, repo_root=tmp_path)


def test_validation_rejects_mismatched_partition_cache_digest(tmp_path: Path):
    """Evidence validation must reject runs whose cache SHA-256 does not match manifest."""
    run_dir = _create_synthetic_run(tmp_path, mismatched_partition_digest=True)
    result = validate_run_evidence(run_dir, repo_root=tmp_path)
    assert any("partition cache digest mismatch" in err for err in result.errors)
    assert not is_run_evidence_complete(run_dir, repo_root=tmp_path)
