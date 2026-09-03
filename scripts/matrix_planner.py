"""Pure matrix expansion and command planning for the sweep runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fedmaq.core.protocol import validate_matrix_against_protocol
from scripts.common import (
    build_run_command,
    expand_matrix,
    get_canonical_output_dir,
    get_sweep_group_dir,
    partition_tasks,
    sharded_sweep_status_filename,
    validate_unique_output_dirs,
)

REPO_MATRIX_DIR = (Path(__file__).resolve().parents[1] / "conf" / "matrix").resolve()

# Closed set, because `expand_matrix` reads every run-spec key through `.get`: an
# unrecognised key is silently ignored rather than rejected, so a misspelt
# `pending_selection` would disarm the guard below with no signal.
RUN_SPEC_KEYS = frozenset({"alg", "label", "overrides", "pending_selection", "seeds", "variant"})


@dataclass(frozen=True)
class MatrixTask:
    """One fully resolved, executable matrix row."""

    canonical_index: int
    label: str
    algorithm: str
    heterogeneity: str
    seed: int
    output_dir: Path
    command: tuple[str, ...]

    @property
    def command_list(self) -> list[str]:
        return list(self.command)


@dataclass(frozen=True)
class MatrixPlan:
    """The immutable plan shared by dry-run output and execution."""

    matrix_path: Path
    matrix: dict[str, Any]
    phase: str
    experiment_group: str
    dataset: str
    model: str
    total_rounds: int
    client_gpus: float
    experiment: str | None
    seeds: tuple[int, ...]
    heterogeneities: tuple[str, ...]
    canonical_tasks: tuple[MatrixTask, ...]
    tasks: tuple[MatrixTask, ...]
    shard: tuple[int, int] | None
    stage: str = "downstream"
    split: str = "val"
    ledger: str = "scientific"

    @property
    def status_path(self) -> Path:
        filename = (
            sharded_sweep_status_filename(*self.shard)
            if self.shard is not None
            else "sweep_status.json"
        )
        return (
            get_sweep_group_dir(self.phase, self.dataset, self.model, self.experiment_group)
            / filename
        )


def _resolved_mapping(matrix: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    resolved = (
        OmegaConf.to_container(matrix, resolve=True) if isinstance(matrix, DictConfig) else matrix
    )
    return dict(resolved)


def unresolved_selection(run: Mapping[str, Any]) -> list[str]:
    """Return the keys of one matrix row still awaiting a selection verdict.

    Two markers, because the ``???`` sentinel does not fail closed on every read.
    Measured, on a composed ``algorithm.p=???``: a ``DictConfig`` subscript raises
    ``MissingMandatoryValue``, but ``.get("p", 0.0)`` returns ``0.0`` and ``.get("p")``
    returns ``None``. ``quantization_planner.py`` takes the subscript only when the
    formulation requires ``p``, and ``manifest.py`` records the run with ``.get`` -- so
    the sentinel's safety is contingent, and provenance would not even show the
    placeholder. This planner is the one read that fails closed regardless.

    ``???`` is visible here because it sits inside an ``overrides`` *list*: a string
    element, not a node, so ``to_container(..., resolve=True)`` passes it through
    verbatim instead of raising.

    ``pending_selection`` covers the keys the sentinel would actively break. A boolean
    read as ``alg_cfg.get("soft_voting", False)`` sees the string ``'???'`` as truthy
    and switches the mechanism ON -- the inverse of a placeholder's purpose.
    """
    sentinel = [
        key.strip().lstrip("+~")
        for key, _, value in (
            str(override).partition("=") for override in (run.get("overrides") or [])
        )
        # Quotes survive YAML into the override string, so `p='???'` is the same marker.
        if value.strip().strip("\"'") == "???"
    ]
    pending = [str(key) for key in (run.get("pending_selection") or [])]
    return sentinel + pending


def plan_matrix(
    matrix_path: Path,
    matrix: Mapping[str, Any] | DictConfig,
    *,
    overrides: Sequence[str] = (),
    only_labels: Sequence[str] = (),
    shard: tuple[int, int] | None = None,
) -> MatrixPlan:
    """Resolve a matrix into canonical commands without touching the filesystem.

    Filtering, expansion, path construction, command ordering, and sharding all
    happen here.  The executor receives only this result and therefore cannot
    accidentally invent a second definition of the campaign's task list.
    """

    if shard is not None and only_labels:
        raise ValueError("--only cannot be combined with --shard; shard the full matrix")

    resolved = _resolved_mapping(matrix)
    ledger = str(resolved.get("ledger", "unregistered"))
    if ledger == "unregistered" and matrix_path.resolve().parent == REPO_MATRIX_DIR:
        raise ValueError(f"{matrix_path.name} is not registered to an execution ledger")
    runs_spec = list(resolved.get("runs", []) or [])
    if only_labels:
        available = [str(item.get("label", item.get("alg"))) for item in runs_spec]
        unknown = [label for label in only_labels if label not in available]
        if unknown:
            raise ValueError(
                f"--only label(s) {unknown} not in {matrix_path.name}; "
                f"available labels: {available}"
            )

    phase = str(resolved.get("phase", "smoke"))
    experiment_group = str(resolved.get("experiment_group", matrix_path.stem))
    dataset = str(resolved.get("dataset", "cifar10"))
    model = str(resolved.get("model", "mobilenetv2"))
    total_rounds = int(resolved.get("total_rounds", 50))
    client_gpus = float(resolved.get("client_gpus", 1.0))
    experiment = resolved.get("experiment")
    seeds = tuple(int(seed) for seed in resolved.get("seeds", [0]))
    heterogeneities = tuple(
        str(value) for value in resolved.get("heterogeneities", ["dirichlet_alpha_0.1"])
    )

    for spec in runs_spec:
        unknown = sorted(set(spec) - RUN_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"{matrix_path.name} run {spec.get('label', spec.get('alg'))!r} declares "
                f"unrecognised key(s) {unknown}; known keys are {sorted(RUN_SPEC_KEYS)}"
            )

    # Whole-matrix, ahead of the protocol check: a pre-selection matrix has no stable
    # fingerprint yet, so a hash mismatch here would report the placeholder as tampering.
    # Filtering is ignored deliberately -- one unresolved row means the matrix identity
    # is still moving, and rows dispatched under the old identity are not comparable.
    unresolved = {
        str(spec.get("label", spec.get("alg"))): keys
        for spec in runs_spec
        if (keys := unresolved_selection(spec))
    }
    if unresolved:
        detail = "; ".join(f"{label}: {', '.join(keys)}" for label, keys in unresolved.items())
        raise ValueError(
            f"{matrix_path.name} has unresolved selections and cannot be dispatched -- {detail}"
        )

    if matrix_path.resolve().parent == REPO_MATRIX_DIR and ledger == "scientific":
        validate_matrix_against_protocol(
            matrix_path.stem, resolved, len(expand_matrix(resolved, matrix_path.stem))
        )

    task_dicts = []
    for spec in expand_matrix(resolved, matrix_path.stem):
        output_dir = get_canonical_output_dir(
            phase=spec["phase"],
            dataset=spec["dataset"],
            model=spec["model"],
            exp_group=spec["experiment_group"],
            algorithm=spec["algorithm_config"],
            heterogeneity=spec["heterogeneity"],
            seed=spec["seed"],
            variant=spec["variant"],
        )
        command = build_run_command(
            dataset=spec["dataset"],
            heterogeneity=spec["heterogeneity"],
            algorithm=spec["algorithm_config"],
            total_rounds=total_rounds,
            seed=spec["seed"],
            client_gpus=client_gpus,
            target_dir=output_dir,
            # Host overrides precede matrix overrides so a row's declared regime
            # cannot be displaced by a command-line convenience flag.
            overrides=[
                *overrides,
                f"protocol_stage={spec['protocol_stage']}",
                f"split={spec['split']}",
                *spec["overrides"],
            ],
            experiment=experiment,
        )
        task_dicts.append(
            {
                "canonical_index": spec["canonical_index"],
                "label": f"{spec['label']}-{spec['heterogeneity']}-seed{spec['seed']}",
                "base_label": str(spec["label"]),
                "alg": spec["algorithm_config"],
                "het": spec["heterogeneity"],
                "seed": spec["seed"],
                "output_dir": output_dir,
                "cmd": command,
            }
        )

    validate_unique_output_dirs(task_dicts)

    # Keep canonical indices from the complete matrix when selecting labels. A
    # filtered recovery must still agree with a full dispatch's --start_at view.
    selected_dicts = (
        [task for task in task_dicts if task["base_label"] in only_labels]
        if only_labels
        else task_dicts
    )

    def make_task(task: dict[str, Any]) -> MatrixTask:
        return MatrixTask(
            canonical_index=int(task["canonical_index"]),
            label=str(task["label"]),
            algorithm=str(task["alg"]),
            heterogeneity=str(task["het"]),
            seed=int(task["seed"]),
            output_dir=Path(task["output_dir"]),
            command=tuple(str(arg) for arg in task["cmd"]),
        )

    canonical_tasks = tuple(make_task(task) for task in task_dicts)
    # Partition only after the complete task list is fixed; shard membership is
    # the canonical round-robin partition, independent of completion state.
    scheduled_dicts = (
        partition_tasks(selected_dicts, *shard) if shard is not None else selected_dicts
    )
    scheduled_indices = {task["canonical_index"] for task in scheduled_dicts}
    scheduled_tasks = tuple(
        task for task in canonical_tasks if task.canonical_index in scheduled_indices
    )
    return MatrixPlan(
        matrix_path=matrix_path,
        matrix=resolved,
        phase=phase,
        experiment_group=experiment_group,
        dataset=dataset,
        model=model,
        total_rounds=total_rounds,
        client_gpus=client_gpus,
        experiment=str(experiment) if experiment is not None else None,
        stage=str(resolved.get("stage", resolved.get("protocol_stage", "downstream"))),
        split=str(resolved.get("split", "val")),
        ledger=ledger,
        seeds=seeds,
        heterogeneities=heterogeneities,
        canonical_tasks=canonical_tasks,
        tasks=scheduled_tasks,
        shard=shard,
    )


__all__ = ["MatrixPlan", "MatrixTask", "plan_matrix", "unresolved_selection"]
