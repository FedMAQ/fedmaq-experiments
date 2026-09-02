"""Shared execution utilities for process-isolated experiment runners.

Provides cross-platform Ray cleanup, canonical path construction, command line formatting,
and subprocess execution helpers used by ``scripts/run_matrix.py``.
"""

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from fedmaq.core.run_identity import NO_GROUP, get_canonical_output_dir, identity_key
from fedmaq.core.validation import is_run_evidence_complete, validate_run_evidence

__all__ = [
    "NO_GROUP",
    "get_canonical_output_dir",
    "identity_key",
    "is_run_complete",
    "is_run_evidence_complete",
    "validate_run_evidence",
]

logger = logging.getLogger("fedmaq.runner")

SWEEP_STATUS_FILENAME = "sweep_status.json"
SHARDED_SWEEP_STATUS_TEMPLATE = "sweep_status.shard-{index}-of-{count}.json"


def parse_shard(value: str) -> tuple[int, int]:
    """Parse a 1-based shard selector in the form ``i/N``.

    Shard membership is deliberately expressed in terms of the canonical matrix
    list, so the selector must be a positive index within a positive shard count.
    Keeping parsing here makes the CLI and any future dispatcher use the same
    validation rules.
    """
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", value.strip())
    if not match:
        raise ValueError(f"shard must have the form i/N with 1 <= i <= N, got {value!r}")
    index, count = (int(part) for part in match.groups())
    if index > count:
        raise ValueError(f"shard index {index} is outside 1..{count}")
    return index, count


def partition_tasks(tasks: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    """Return one deterministic round-robin partition of a canonical task list.

    The caller owns the canonical order. This function neither sorts nor consults
    completion state, host identity, or wall-clock time. Consequently every host
    given the same matrix and ``N`` computes the same disjoint membership.
    """
    if not 1 <= shard_index <= shard_count:
        raise ValueError(
            f"shard index must satisfy 1 <= index <= count, got {shard_index}/{shard_count}"
        )
    return [
        task
        for zero_based_index, task in enumerate(tasks)
        if zero_based_index % shard_count == shard_index - 1
    ]


def sharded_sweep_status_filename(shard_index: int, shard_count: int) -> str:
    """Return the collision-free status filename for one shard invocation."""
    if not 1 <= shard_index <= shard_count:
        raise ValueError(
            f"shard index must satisfy 1 <= index <= count, got {shard_index}/{shard_count}"
        )
    return SHARDED_SWEEP_STATUS_TEMPLATE.format(index=shard_index, count=shard_count)


def validate_unique_output_dirs(tasks: list[dict]) -> None:
    """Reject a matrix whose concrete runs would overwrite one another."""
    seen: dict[str, str] = {}
    for task in tasks:
        output_dir = str(task["output_dir"])
        prior = seen.get(output_dir)
        if prior is not None:
            raise ValueError(
                f"matrix maps runs {prior!r} and {task['label']!r} onto {output_dir}; "
                "give colliding runs distinct variants"
            )
        seen[output_dir] = task["label"]


def kill_ray_processes() -> None:
    """Stop Ray cluster and forcibly kill lingering Ray sub-processes.

    Ensures VRAM and process memory are released between sequential runs to
    prevent CUDA OOM and Ray actor leaks.

    ``ray stop``'s exit status is reported rather than discarded. A failed cleanup
    used to be completely invisible here, and on a contended GPU with roughly 11 GB
    of usable headroom that is the difference between a clean sweep and an OOM
    cascade through every remaining run. It stays non-fatal: no cluster to stop is
    the normal case between runs and must not raise.
    """
    logger.info("Stopping Ray and cleaning up lingering processes...")
    res = subprocess.run(
        ["uv", "run", "ray", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if res.returncode != 0:
        logger.warning(
            f"'ray stop' exited with code {res.returncode}; leaked actors may still "
            f"hold VRAM. stderr: {(res.stderr or '').strip()[:500]}"
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
    else:
        # `ray stop` asks nicely; a raylet/GCS wedged by a timeout-killed run (the
        # driver process is dead, but Ray's grandchildren survive it) can outlive
        # that request with nothing here to force it down -- unlike the Windows
        # branch above, which always force-kills unconditionally. The surviving
        # process then holds the port/session state the next task's `ray.init()`
        # needs, so that task fails at Ray startup rather than running at all,
        # turning one stall into a chain of unrelated-looking failures.
        for pattern in ("raylet", "gcs_server", "plasma_store"):
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
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


def is_run_complete(output_dir: Path) -> bool:
    """Report whether ``output_dir`` holds a run that reached its final round with valid evidence.

    Requires:
    1. A valid final checkpoint (``final_global_model.pt``).
    2. A valid run manifest (``run_manifest.json``) with matching identity.
    3. Contiguous unique finite telemetry for rounds 1 through R with monotonically
       non-decreasing cumulative communication metrics.
    """
    return is_run_evidence_complete(output_dir)


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


def expand_matrix(matrix: dict, matrix_name: str) -> list[dict]:
    """Expand one ``conf/matrix/*.yaml`` into the concrete runs it dispatches.

    Takes a plain container, not a ``DictConfig``: callers hold their own resolved
    config and the expansion must not depend on OmegaConf's lazy interpolation.

    ``seeds`` is per-run overridable, and that is load-bearing rather than
    cosmetic. The exploration factorial's unrefined reference cell defines the
    sigma every keep-or-drop call is judged against (ADR-0008), and a sigma
    estimated from three seeds carries roughly +/-50% of itself, so that one cell
    runs five. A plain ``heterogeneities x seeds x runs`` product therefore
    overcounts every matrix that deepens a cell and undercounts none of them --
    it silently disagrees with what was dispatched.

    Expansion order is the dispatch order (``het``, then seed, then run), because
    ``scripts/run_matrix.py`` resumes on position and reordering the sweep would
    change which runs a ``--start_at`` skips.

    Scope-agnostic by contract: it expands whatever matrix it is handed, including
    ``ci_test`` and the smoke matrices. Any decision about which matrices are
    reportable belongs to the caller, so that narrowing one caller cannot narrow
    ``tests/test_config_and_dispatch.py``'s guard over *every* file in ``conf/matrix/``.
    """
    seeds = [int(s) for s in matrix.get("seeds", [0])]
    runs_spec = matrix.get("runs", []) or []

    def seeds_for(run_item: dict) -> list[int]:
        return [int(s) for s in run_item.get("seeds", seeds)]

    # Matrix-level seeds first, so a matrix declaring no per-run seeds expands in
    # exactly the order it did before per-run seeds existed.
    all_seeds = list(seeds)
    for run_item in runs_spec:
        for s in seeds_for(run_item):
            if s not in all_seeds:
                all_seeds.append(s)

    tasks: list[dict] = []
    for het in matrix.get("heterogeneities", ["dirichlet_alpha_0.1"]):
        for seed in all_seeds:
            for run_item in runs_spec:
                if seed not in seeds_for(run_item):
                    continue
                alg = run_item.get("alg")
                tasks.append(
                    {
                        "canonical_index": len(tasks) + 1,
                        "phase": matrix.get("phase", "smoke"),
                        "dataset": matrix.get("dataset", "cifar10"),
                        "model": matrix.get("model", "mobilenetv2"),
                        "experiment_group": matrix.get("experiment_group", matrix_name),
                        # A per-matrix property, not a per-phase one: `explore`
                        # covers both the 50-round factorial passes and the
                        # 100-round formulation study, so nothing downstream may
                        # infer the round budget from the phase.
                        "total_rounds": int(matrix.get("total_rounds", 50)),
                        "algorithm_config": alg,
                        "variant": run_item.get("variant", ""),
                        "heterogeneity": het,
                        "seed": seed,
                        "label": run_item.get("label", alg),
                        "overrides": list(run_item.get("overrides", []) or []),
                        "stage": matrix.get("stage", matrix.get("protocol_stage", "unregistered")),
                        "protocol_stage": matrix.get("protocol_stage", "unregistered"),
                        "split": matrix.get("split", "val"),
                        "ledger": matrix.get("ledger", "unregistered"),
                    }
                )
    return tasks
