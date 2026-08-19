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
    ``tests/test_simulation.py``'s guard over *every* file in ``conf/matrix/``.
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
                    }
                )
    return tasks


# ``experiment_group`` is None for any run outside the canonical output layout --
# a bare scripts/run.py invocation, or a legacy pre-matrix tree. Such a run belongs
# to no experiment group and is certified against none, so it needs a rendering
# that cannot collide with a real group name.
NO_GROUP = "<none>"


def identity_key(
    dataset: str,
    experiment_group: str | None,
    algorithm_config: str,
    variant: str,
    alpha: float,
    formulation: int | None,
    seed: int,
) -> str:
    """The canonical identity of one run, serialized in exactly one place.

    ADR-0009 is the authority on what identifies a run, and each field here closes
    a collision that has actually occurred or is reachable from tracked config:

    ``dataset``            the three ``benchmark_grid*`` files share one group and
                           differ in nothing else the analysis reads, so a key
                           without it folds 105 primary-grid runs onto 42.
    ``experiment_group``   ``uniform_memory_control`` runs ``fedmaq`` at the grid's
                           own dataset, skews and seeds; only the group separates
                           them.
    ``algorithm_config``   every §4.3.7 ablation arm declares ``name: fedmaq``.
    ``variant``            Stage 1b sweeps one override per baseline, so those cells
                           differ in *nothing* else a RunRecord carries. Without it
                           baseline_tuning's 15 cells fold onto 5 and
                           pass2_factorial's 8 arms onto 1.

    ``phase`` and ``post_process`` are the two remaining ADR-0009 identity fields
    and are deliberately absent: across every reportable matrix each is a function
    of ``experiment_group``, so keying on them adds no discrimination. That is a
    checked property, not an assumption -- see
    ``test_each_reportable_group_carries_one_phase_and_one_post_process_regime``.

    **The serialization is a contract, not a convenience.** One side of the closure
    certificate builds these from RunRecords and the other from
    ``conf/matrix/*.yaml`` via Hydra, and a formatting disagreement on any field
    yields *paired* missing-and-unexpected entries rather than an error -- a
    certificate that reads as a total mismatch while every unit test still passes.
    So ``alpha`` is coerced through ``float`` on both sides (``1`` and ``1.0`` must
    not be two runs), ``formulation`` renders its absence as a word rather than as
    an empty field, and both sides call this function rather than formatting their
    own.
    """
    group = experiment_group if experiment_group else NO_GROUP
    form = "none" if formulation is None else str(int(formulation))
    return f"{dataset}|{group}|{algorithm_config}|{variant}|a{float(alpha)!r}|f{form}|s{int(seed)}"
