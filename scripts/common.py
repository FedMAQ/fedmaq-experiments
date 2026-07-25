"""Shared execution utilities for process-isolated experiment runners.

Provides cross-platform Ray cleanup, canonical path construction, command line formatting,
and subprocess execution helpers used by ``scripts/run_matrix.py``.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("fedmaq.runner")


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
    return Path(
        f"outputs/{phase}/{dataset}_{model}/{exp_group}/{algorithm_segment}/"
        f"{heterogeneity}/seed_{seed}"
    )


def build_run_command(
    dataset: str,
    heterogeneity: str,
    algorithm: str,
    total_rounds: int,
    seed: int,
    client_gpus: float,
    target_dir: Path,
    overrides: list[str] | None = None,
) -> list[str]:
    """Construct the command array for launching scripts/run.py via uv."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run.py",
        f"dataset={dataset}",
        f"heterogeneity={heterogeneity}",
        f"algorithm={algorithm}",
        f"experiment.total_rounds={total_rounds}",
        f"seed={seed}",
        f"experiment.client_gpus={client_gpus}",
        f"hydra.run.dir={target_dir.as_posix()}",
    ]
    if overrides:
        cmd.extend(overrides)
    return cmd
