"""KD-repair configuration and telemetry for the server-side distillation seam (ADR-0028).

Every repair family acts inside :func:`fedmaq.core.kd_utils.distill_ensemble_into_global`,
which the FedMAQ and FedAvg-KD hooks both call after aggregation. The families are
configured under one ``kd_repair`` group of the algorithm config, each family off by
default. The group is read in code and never declared in ``conf/algorithm/*.yaml``,
so the first-study and ``v2_confirm`` resolved-config hashes stay unchanged; a repair
arm adds it with ``+algorithm.kd_repair.<family>.enabled=true`` and its settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fedmaq.core.config_defaults import resolve_algorithm_config

KD_REPAIR_FAMILIES: tuple[str, ...] = (
    "schedule",
    "temperature",
    "teacher_selection",
    "teacher_weighting",
    "guard",
    "class_balanced",
)


@dataclass(frozen=True)
class KDRepairConfig:
    """The single enabled repair family of one arm, or none."""

    family: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    @property
    def needs_class_counts(self) -> bool:
        """Only class-balanced KD reads the participants' label histograms."""
        return self.family == "class_balanced"


def resolve_kd_repair(alg_cfg: Mapping[str, Any]) -> KDRepairConfig:
    """Return the arm's repair family, refusing unknown families and combined arms.

    ADR-0028 item 2 tests one mechanism per arm, so an arm that enables two
    families is a configuration error, not a combination to run.
    """
    group = alg_cfg.get("kd_repair") or {}
    if not isinstance(group, Mapping):
        raise ValueError(f"algorithm.kd_repair must be a mapping of families, got {group!r}")
    unknown = sorted(set(group) - set(KD_REPAIR_FAMILIES))
    if unknown:
        raise ValueError(
            f"Unknown KD-repair families {unknown}; expected a subset of {list(KD_REPAIR_FAMILIES)}"
        )
    enabled = [
        family
        for family in KD_REPAIR_FAMILIES
        if isinstance(group.get(family), Mapping) and bool(group[family].get("enabled", False))
    ]
    if len(enabled) > 1:
        raise ValueError(
            f"KD-repair arm enables {enabled}; ADR-0028 allows one repair family per arm"
        )
    if not enabled:
        return KDRepairConfig()
    family = enabled[0]
    settings = {k: v for k, v in group[family].items() if k != "enabled"}
    return KDRepairConfig(family=family, settings=settings)


def validate_kd_repair_config(config: Mapping[str, Any]) -> KDRepairConfig:
    """Resolve the repair group from a run config so a bad arm fails before round 1."""
    return resolve_kd_repair(resolve_algorithm_config(config))


def unrepaired_telemetry(
    num_teachers: int,
    *,
    kd_weight: float,
    server_sim_time: float,
) -> dict[str, float]:
    """Per-round repair telemetry describing plain ensemble KD.

    ``kd_weight`` is 1 when the distillation pass ran and 0 when it was skipped.
    """
    equal_weight = 1.0 / num_teachers if num_teachers > 0 else 0.0
    return {
        "kd_applied_weight": float(kd_weight),
        "kd_effective_teachers": float(num_teachers),
        "kd_teacher_weight_min": equal_weight,
        "kd_teacher_weight_max": equal_weight,
        "kd_skipped_batches": 0.0,
        "kd_guard_accept": 1.0,
        "kd_server_sim_time": float(server_sim_time),
    }


def participant_class_counts(
    partition_ids: Sequence[int],
    client_indices: Mapping[Any, Sequence[int]],
    labels: np.ndarray,
    num_classes: int,
) -> list[list[int]]:
    """Label histogram of each participating client's partition, in ``partition_ids`` order."""
    counts: list[list[int]] = []
    for pid in partition_ids:
        indices = client_indices.get(str(pid), client_indices.get(int(pid)))
        if indices is None:
            raise KeyError(f"Partition {pid} is not in the client index map")
        histogram = np.bincount(labels[list(indices)], minlength=num_classes)
        counts.append([int(c) for c in histogram[:num_classes]])
    return counts
