"""Generate the replacement pipeline's exact Stage-A validation manifest."""

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
from scripts.dump_expected_runs import MATRIX_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_NAME = "baseline_tuning_wide"
MANIFEST_PATH = REPO_ROOT / "docs" / "recut" / "stage_a_manifest.json"
SCHEMA_VERSION = 1
STAGE = "matched_tuning"
SPLIT = "val"
LEDGER = "scientific"
SEEDS = (0, 42, 123, 7, 21)
EXPECTED_COUNT = 145
EXPECTED_ALGORITHMS = {"fedprox", "fedpaq", "dadaquant", "feddistill", "fedkd", "fedmaq"}


def _matrix() -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(MATRIX_DIR / f"{MATRIX_NAME}.yaml"), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"{MATRIX_NAME} must be a mapping")
    return value


def _cell_id(task: dict[str, Any], alpha: float, formulation: Any) -> str:
    return (
        identity_key(
            dataset=str(task["dataset"]),
            experiment_group=str(task["experiment_group"]),
            algorithm_config=str(task["algorithm_config"]),
            variant=str(task["variant"]),
            alpha=alpha,
            formulation=formulation,
            seed=int(task["seed"]),
        )
        + f"|stage={STAGE}|split={SPLIT}"
    )


def build_manifest() -> dict[str, Any]:
    """Expand and validate the complete 145-cell Stage-A contract."""
    matrix = _matrix()
    if matrix.get("stage") != STAGE or matrix.get("protocol_stage") != STAGE:
        raise ValueError("Stage A must register protocol stage matched_tuning")
    if matrix.get("split") != SPLIT or matrix.get("ledger") != LEDGER:
        raise ValueError("Stage A must use the scientific validation ledger and val split")
    if tuple(int(seed) for seed in matrix.get("seeds", [])) != SEEDS:
        raise ValueError(f"Stage A seeds must be exactly {SEEDS}")
    identity = matrix.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Stage A must register identity metadata")
    alpha = float(identity.get("alpha"))
    formulation = int(identity.get("fedmaq_formulation"))
    expected_heterogeneity = f"dirichlet_alpha_{alpha:g}"
    if matrix.get("heterogeneities") != [expected_heterogeneity]:
        raise ValueError("Stage A identity alpha does not match its heterogeneity registry")

    tuning = matrix.get("tuning")
    if not isinstance(tuning, dict) or set(tuning) != EXPECTED_ALGORITHMS:
        raise ValueError("Stage A tuning registry must cover exactly the six tunable algorithms")

    tasks = expand_matrix(matrix, MATRIX_NAME)
    if len(tasks) != EXPECTED_COUNT:
        raise ValueError(f"Stage A expands to {len(tasks)} cells, expected {EXPECTED_COUNT}")

    cells: list[dict[str, Any]] = []
    for task in tasks:
        algorithm = str(task["algorithm_config"])
        registration = tuning.get(algorithm)
        if not isinstance(registration, dict):
            raise ValueError(f"missing tuning registration for {algorithm}")
        values = registration.get("values")
        if not isinstance(values, dict) or task["variant"] not in values:
            raise ValueError(
                f"{algorithm} variant {task['variant']!r} is not in its registered domain"
            )
        cells.append(
            {
                "cell_id": _cell_id(task, alpha, formulation if algorithm == "fedmaq" else None),
                "canonical_index": int(task["canonical_index"]),
                "stage": STAGE,
                "ledger": LEDGER,
                "split": SPLIT,
                "dataset": str(task["dataset"]),
                "model": str(task["model"]),
                "heterogeneity": str(task["heterogeneity"]),
                "algorithm": algorithm,
                "variant": str(task["variant"]),
                "seed": int(task["seed"]),
                "hyperparameter": {
                    "name": str(registration["knob"]),
                    "variant": str(task["variant"]),
                    "value": values[task["variant"]],
                },
                "overrides": list(task["overrides"]),
            }
        )

    ids = [cell["cell_id"] for cell in cells]
    if len(set(ids)) != EXPECTED_COUNT:
        raise ValueError("Stage A contains duplicate cell identities")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "ledger": LEDGER,
        "split": SPLIT,
        "matrix": f"conf/matrix/{MATRIX_NAME}.yaml",
        "registered_seeds": list(SEEDS),
        "registered_algorithms": sorted(EXPECTED_ALGORITHMS),
        "registered_hyperparameters": {
            algorithm: {
                "knob": registration["knob"],
                "values": registration["values"],
            }
            for algorithm, registration in sorted(tuning.items())
        },
        "cell_count": len(cells),
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_manifest(), indent=2) + "\n"
    if args.check:
        if not MANIFEST_PATH.is_file() or MANIFEST_PATH.read_text(encoding="utf-8") != rendered:
            print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} is stale or missing", file=sys.stderr)
            return 1
        print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} is current")
        return 0
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({EXPECTED_COUNT} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
