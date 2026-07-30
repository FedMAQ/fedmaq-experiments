"""Shared execution utilities for process-isolated experiment runners.

Provides cross-platform Ray cleanup, canonical path construction, command line formatting,
and subprocess execution helpers used by ``scripts/run_matrix.py``.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME

logger = logging.getLogger("fedmaq.runner")

SWEEP_STATUS_FILENAME = "sweep_status.json"


def kill_ray_processes() -> None:
    """Stop Ray cluster and forcibly kill lingering Ray sub-processes.

    Ensures VRAM and process memory are released between sequential runs to
    prevent CUDA OOM and Ray actor leaks.
    """
    logger.info("Stopping Ray and cleaning up lingering processes...")
    subprocess.run(
        ["uv", "run", "ray", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "raylet.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "gcs_server.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(3)


def get_sweep_group_dir(phase: str, dataset: str, model: str, exp_group: str) -> Path:
    """Return the directory holding every run of one matrix.

    ``outputs/<phase>/<dataset>_<model>/<exp_group>/`` -- the common ancestor of
    that matrix's canonical run directories, and so where sweep-level rather than
    run-level artifacts belong (``sweep_status.json``).
    """
    return Path(f"outputs/{phase}/{dataset}_{model}/{exp_group}")


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
    """Construct canonical output directory path matching thesis taxonomy:

    ``outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<heterogeneity>/seed_<seed>/``

    ``variant`` disambiguates several runs of the *same* algorithm config within
    one matrix, appended to the algorithm segment as ``<algorithm>__<variant>``.
    The path keys on the algorithm rather than the run label, so without it a
    matrix that sweeps an override (the formulation study's five formulations of
    ``fedmaq``) silently writes every run into one directory and keeps only the
    last. That has bitten once already, in the uniform-memory control arm, which
    was worked around by splitting the heterogeneity config per alpha. Depth is
    unchanged, so ``analysis.experiment_group_of`` still reads the group.
    """
    algorithm_segment = f"{algorithm}__{variant}" if variant else algorithm
    return (
        get_sweep_group_dir(phase, dataset, model, exp_group)
        / algorithm_segment
        / heterogeneity
        / f"seed_{seed}"
    )


def is_run_complete(output_dir: Path) -> bool:
    """Report whether ``output_dir`` holds a run that reached its final round.

    ``final_global_model.pt`` is the only artifact written exclusively on the
    last round. ``run_manifest.json`` is written before round 1 and the
    telemetry CSV/JSONL are opened just as early, so a run killed at round 3 is
    indistinguishable from a finished one by their presence alone.

    One asymmetry is deliberate: ``write_final_global_model`` logs and swallows
    disk errors, so a run that trained fully but failed to save its checkpoint
    reads as incomplete here and would be redone. Redoing a finished run costs
    wall-clock; skipping an unfinished one leaves a hole that only surfaces at
    analysis time, so the bias points the safe way.
    """
    return (output_dir / FINAL_MODEL_FILENAME).is_file()


def build_run_command(
    dataset: str,
    heterogeneity: str,
    algorithm: str,
    total_rounds: int,
    seed: int,
    client_gpus: float,
    target_dir: Path,
    overrides: list[str] | None = None,
    experiment: str | None = None,
) -> list[str]:
    """Construct the command array for launching scripts/run.py via uv.

    ``experiment`` selects a non-default ``conf/experiment/`` group. It exists
    for FEMNIST, whose grid is not the default one overridden a few values at a
    time: ``conf/experiment/femnist.yaml`` carries ``num_clients=200`` (one real
    LEAF writer per client) and the SimpleCNN throughput constants. Without this
    the matrix runner could dispatch FEMNIST only at the CIFAR grid's K=100 and
    MobileNetV2GN telemetry, which contradicts Table 4.1's note (a) and §4.3.4.

    The group override is emitted *before* the ``experiment.*`` value overrides.
    Hydra applies group selection during composition and values afterward, so
    the order is not strictly required, but reading the command back is how a
    dispatch mistake gets caught, and value-after-group is how it reads.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run.py",
        f"dataset={dataset}",
        f"heterogeneity={heterogeneity}",
        f"algorithm={algorithm}",
    ]
    if experiment:
        cmd.append(f"experiment={experiment}")
    cmd += [
        f"experiment.total_rounds={total_rounds}",
        f"seed={seed}",
        f"experiment.client_gpus={client_gpus}",
        f"hydra.run.dir={target_dir.as_posix()}",
    ]
    if overrides:
        cmd.extend(overrides)
    return cmd
