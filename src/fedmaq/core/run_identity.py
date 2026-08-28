"""Canonical run-directory and run-identity contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NO_GROUP = "<none>"


def _algorithm_segment(algorithm: str, variant: str) -> str:
    return f"{algorithm}__{variant}" if variant else algorithm


@dataclass(frozen=True)
class RunDirectory:
    """Fields encoded by one canonical output directory."""

    phase: str
    dataset_model: str
    experiment_group: str
    algorithm_path: str
    variant: str
    heterogeneity_path: str
    seed: int

    @property
    def algorithm_segment(self) -> str:
        return _algorithm_segment(self.algorithm_path, self.variant)


def get_canonical_output_dir(
    phase: str,
    dataset: str,
    model: str,
    exp_group: str,
    algorithm: str,
    heterogeneity: str,
    seed: int,
    variant: str = "",
) -> Path:
    """Build the canonical output path for one matrix task."""
    algorithm_segment = _algorithm_segment(algorithm, variant)
    return (
        Path(f"outputs/{phase}/{dataset}_{model}/{exp_group}")
        / algorithm_segment
        / heterogeneity
        / f"seed_{seed}"
    )


def parse_run_directory(job_dir: Path, repo_root: Path) -> RunDirectory | None:
    """Parse a canonical output path, returning ``None`` for every other path."""
    try:
        parts = job_dir.resolve().relative_to(repo_root.resolve()).parts
    except ValueError:
        return None
    if len(parts) != 7 or parts[0] != "outputs" or not parts[6].startswith("seed_"):
        return None
    try:
        seed = int(parts[6].removeprefix("seed_"))
    except ValueError:
        return None
    if parts[6] != f"seed_{seed}":
        return None
    algorithm_path, separator, variant = parts[4].partition("__")
    if not algorithm_path or (separator and not variant):
        return None
    return RunDirectory(
        phase=parts[1],
        dataset_model=parts[2],
        experiment_group=parts[3],
        algorithm_path=algorithm_path,
        variant=variant if separator else "",
        heterogeneity_path=parts[5],
        seed=seed,
    )


def identity_key(
    dataset: str,
    experiment_group: str | None,
    algorithm_config: str,
    variant: str,
    alpha: float,
    formulation: int | str | None,
    seed: int,
) -> str:
    """Serialize the canonical identity of one run."""
    group = experiment_group if experiment_group else NO_GROUP
    form = "none" if formulation is None else str(formulation)
    return f"{dataset}|{group}|{algorithm_config}|{variant}|a{float(alpha)!r}|f{form}|s{int(seed)}"


def config_sha256(config: dict[str, Any]) -> str:
    """Return the stable digest of a resolved configuration."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
