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

# Families whose rule the seam applies; any other enabled family is refused.
IMPLEMENTED_KD_REPAIR_FAMILIES: frozenset[str] = frozenset({"schedule", "temperature"})

_FAMILY_SETTINGS: dict[str, frozenset[str]] = {
    "schedule": frozenset({"start", "stop", "ramp_rounds"}),
    "temperature": frozenset({"value"}),
}


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
    _check_settings(family, settings)
    return KDRepairConfig(family=family, settings=settings)


def _check_settings(family: str, settings: Mapping[str, Any]) -> None:
    allowed = _FAMILY_SETTINGS.get(family)
    if allowed is None:
        return
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"KD-repair {family} has unknown settings {unknown}")
    if family == "schedule":
        start = int(settings.get("start", 1))
        stop = settings.get("stop")
        ramp = int(settings.get("ramp_rounds", 1))
        if start < 1 or ramp < 1 or (stop is not None and int(stop) < start):
            raise ValueError(
                "KD-repair schedule needs 1 <= start <= stop and ramp_rounds >= 1, "
                f"got {dict(settings)}"
            )
    elif family == "temperature":
        if "value" not in settings or not float(settings["value"]) > 0.0:
            raise ValueError(f"KD-repair temperature needs a positive value, got {dict(settings)}")


def schedule_weight(settings: Mapping[str, Any], server_round: int) -> float:
    """KD weight of ``server_round`` (1-based) under the schedule family.

    KD is off before ``start`` and after ``stop`` (both inclusive bounds of the
    window). Over the first ``ramp_rounds`` rounds from ``start`` the weight rises
    linearly, ``(server_round - start + 1) / ramp_rounds``, and then stays at 1.
    ``start=1`` with no ``stop`` and no ramp is KD in every round.
    """
    start = int(settings.get("start", 1))
    stop = settings.get("stop")
    ramp = int(settings.get("ramp_rounds", 1))
    if server_round < start or (stop is not None and server_round > int(stop)):
        return 0.0
    return min(1.0, (server_round - start + 1) / ramp)


def kd_temperature(repair: KDRepairConfig, alg_cfg: Mapping[str, Any]) -> float:
    """Distillation temperature of the pass: the temperature family's value, else the frozen one.

    The value softens teachers and student alike, and the KD loss keeps its T^2
    rescaling (Hinton et al.), so gradient magnitudes stay comparable across T.
    """
    if repair.family == "temperature":
        return float(repair.settings["value"])
    return float(alg_cfg.get("temperature", 1.0))


def validate_kd_repair_config(config: Mapping[str, Any]) -> KDRepairConfig:
    """Resolve the repair group from a run config so a bad arm fails before round 1."""
    return resolve_kd_repair(resolve_algorithm_config(config))


def kd_repair_telemetry(
    num_teachers: int,
    *,
    kd_weight: float,
    server_sim_time: float,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Per-round KD-repair telemetry of one distillation pass.

    ``kd_weight`` is the weight of the distilled student in the returned parameters:
    1 for a full pass, 0 when the pass was skipped, and in between on a schedule ramp.
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
        "kd_temperature": float(temperature),
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
