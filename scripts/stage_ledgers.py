"""Generate disjoint scientific and assurance ledgers for the replacement pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedmaq.core.run_identity import identity_key
from scripts.common import expand_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = REPO_ROOT / "conf" / "matrix"
LEDGER_PATH = REPO_ROOT / "docs" / "recut" / "stage_ledgers.json"
SCHEMA_VERSION = 1

SCIENTIFIC_STAGES: dict[str, tuple[str, ...]] = {
    "matched_tuning": ("baseline_tuning_wide",),
    "stage_1a": ("power_mean_design",),
    "stage_1b": ("power_mean_omega",),
    "downstream": (
        "benchmark_grid",
        "benchmark_grid_cifar100",
        "benchmark_grid_femnist",
        "ablation",
        "fedpaq_pipeline",
        "fedpaq_pipeline_cifar100",
        "fedpaq_pipeline_femnist",
        "memory_sensitivity",
        "uniform_memory_control",
    ),
}
ASSURANCE_MATRICES = ("ci_test", "mobilenetv2_smoke_50r")
EXPECTED_COUNTS = {"matched_tuning": 145, "stage_1a": 126, "stage_1b": 12, "downstream": 174}
EXECUTION_OUTPUTS = {
    "golden_transition": "outputs/golden/step2_transition",
    "golden_repeatability": "outputs/golden/step2_repeatability",
    "concurrency": "outputs/assurance/concurrency",
}


def _load(name: str) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(MATRIX_DIR / f"{name}.yaml"), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _ids(matrix: dict[str, Any], name: str, stage: str) -> list[str]:
    tasks = expand_matrix(matrix, name)
    return [
        identity_key(
            dataset=str(task["dataset"]),
            experiment_group=str(task["experiment_group"]),
            algorithm_config=str(task["algorithm_config"]),
            variant=str(task["variant"]),
            alpha=_alpha(task),
            formulation=2 if task["algorithm_config"] == "fedmaq" else None,
            seed=int(task["seed"]),
        )
        + f"|stage={stage}|split={task['split']}"
        for task in tasks
    ]


def _alpha(task: dict[str, Any]) -> float:
    """Recover the alpha encoded in a matrix task's heterogeneity label."""
    label = str(task["heterogeneity"])
    try:
        return float(label.rsplit("_", 1)[-1])
    except ValueError:
        return 1.0


def build_ledgers() -> dict[str, Any]:
    scientific: dict[str, Any] = {}
    all_scientific_ids: list[str] = []
    for stage, matrix_names in SCIENTIFIC_STAGES.items():
        stage_ids: list[str] = []
        for name in matrix_names:
            matrix = _load(name)
            if matrix.get("stage") != stage or matrix.get("ledger") != "scientific":
                raise ValueError(f"{name} is not registered to scientific stage {stage}")
            stage_ids.extend(_ids(matrix, name, stage))
        expected = EXPECTED_COUNTS[stage]
        if len(stage_ids) != expected or len(set(stage_ids)) != expected:
            raise ValueError(f"{stage} expands to a non-unique {len(stage_ids)} cells")
        scientific[stage] = {
            "ledger": "scientific",
            "matrices": list(matrix_names),
            "cell_count": len(stage_ids),
            "cell_ids": sorted(stage_ids),
        }
        all_scientific_ids.extend(stage_ids)

    assurance: dict[str, Any] = {}
    assurance_ids: list[str] = []
    for name in ASSURANCE_MATRICES:
        matrix = _load(name)
        if matrix.get("stage") != "assurance" or matrix.get("ledger") != "assurance":
            raise ValueError(f"{name} is not registered to the assurance ledger")
        ids = _ids(matrix, name, "assurance")
        assurance[name] = {"ledger": "assurance", "cell_count": len(ids), "cell_ids": sorted(ids)}
        assurance_ids.extend(ids)

    if set(all_scientific_ids) & set(assurance_ids):
        raise ValueError("scientific and assurance ledgers overlap")
    execution_records = {
        name: {
            "ledger": "assurance",
            "cell_count": len(assurance_ids),
            "cell_ids": sorted(f"{cell_id}|execution={name}" for cell_id in assurance_ids),
            "output_root": output_root,
            "provenance": "run_manifest.json",
        }
        for name, output_root in EXECUTION_OUTPUTS.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific": {
            "stages": scientific,
            "total_cell_count": sum(EXPECTED_COUNTS.values()),
        },
        "assurance": {
            "matrices": assurance,
            "executions": list(EXECUTION_OUTPUTS),
            "execution_records": execution_records,
            "total_matrix_cell_count": len(assurance_ids),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_ledgers(), indent=2) + "\n"
    if args.check:
        if not LEDGER_PATH.is_file() or LEDGER_PATH.read_text(encoding="utf-8") != rendered:
            print(f"{LEDGER_PATH.relative_to(REPO_ROOT)} is stale or missing", file=sys.stderr)
            return 1
        print(f"{LEDGER_PATH.relative_to(REPO_ROOT)} is current")
        return 0
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {LEDGER_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
