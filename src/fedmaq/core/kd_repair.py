"""KD-repair configuration and telemetry for the server-side distillation seam (ADR-0028).

Every repair family acts inside :func:`fedmaq.core.kd_utils.distill_ensemble_into_global`,
which the FedMAQ and FedAvg-KD hooks both call after aggregation. The families are
configured under one ``kd_repair`` group of the algorithm config, each family off by
default. The group is read in code and never declared in ``conf/algorithm/*.yaml``,
so the first-study and ``v2_confirm`` resolved-config hashes stay unchanged; a repair
arm adds it with ``+algorithm.kd_repair.<family>.enabled=true`` and its settings.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

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
IMPLEMENTED_KD_REPAIR_FAMILIES: frozenset[str] = frozenset(
    {"schedule", "temperature", "teacher_selection", "teacher_weighting"}
)

_FAMILY_SETTINGS: dict[str, frozenset[str]] = {
    "schedule": frozenset({"start", "stop", "ramp_rounds"}),
    "temperature": frozenset({"value"}),
    "teacher_selection": frozenset({"threshold"}),
    "teacher_weighting": frozenset({"beta"}),
}

# Maps one batch's teacher predictions [T, B, C] to per-teacher weights [T] that sum
# to 1, or to None when the batch is skipped.
TeacherWeighter = Callable[[torch.Tensor], "torch.Tensor | None"]


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
    elif family == "teacher_selection":
        if "threshold" not in settings or not 0.0 < float(settings["threshold"]) <= 1.0:
            raise ValueError(
                f"KD-repair teacher_selection needs a threshold in (0, 1], got {dict(settings)}"
            )
    elif family == "teacher_weighting":
        if "beta" not in settings or not float(settings["beta"]) >= 0.0:
            raise ValueError(
                f"KD-repair teacher_weighting needs a non-negative beta, got {dict(settings)}"
            )


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


def normalized_teacher_entropy(preds_stack: torch.Tensor) -> torch.Tensor:
    """Batch-mean predictive entropy of each teacher over ``log C``, clamped to [0, 1]."""
    num_classes = preds_stack.shape[2]
    entropy = -(preds_stack * torch.log(preds_stack.clamp_min(1e-12))).sum(dim=2)
    return (entropy.mean(dim=1) / math.log(num_classes)).clamp(0.0, 1.0)


def select_teachers(preds_stack: torch.Tensor, threshold: float) -> torch.Tensor | None:
    """Equal weights over the teachers at or below the entropy threshold; None if none is."""
    kept = (normalized_teacher_entropy(preds_stack) <= threshold).to(preds_stack.dtype)
    if kept.sum() == 0:
        return None
    return kept / kept.sum()


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence (natural log) of two [B, C] distributions."""
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-12))

    def kl(a: torch.Tensor) -> torch.Tensor:
        return (a * (torch.log(a.clamp_min(1e-12)) - log_m)).sum(dim=1)

    return 0.5 * kl(p) + 0.5 * kl(q)


def weight_teachers(preds_stack: torch.Tensor, beta: float) -> torch.Tensor:
    """Weights proportional to exp(-beta * JS(teacher, leave-one-out ensemble)).

    JS is the batch mean. A lone teacher has no leave-one-out ensemble and gets
    weight 1. With two teachers the leave-one-out JS is symmetric, so they always
    tie; the rule separates teachers only in ensembles of three or more.
    """
    num_teachers = preds_stack.shape[0]
    if num_teachers == 1:
        return torch.ones(1, dtype=preds_stack.dtype, device=preds_stack.device)
    total = preds_stack.sum(dim=0)
    divergence = torch.stack(
        [
            _js_divergence(preds_stack[t], (total - preds_stack[t]) / (num_teachers - 1)).mean()
            for t in range(num_teachers)
        ]
    )
    weights = torch.exp(-beta * divergence)
    return weights / weights.sum()


def teacher_weighter(repair: KDRepairConfig) -> TeacherWeighter | None:
    """The arm's per-batch teacher weighter, or None when its family sets no weights."""
    if repair.family == "teacher_selection":
        threshold = float(repair.settings["threshold"])
        return lambda preds_stack: select_teachers(preds_stack, threshold)
    if repair.family == "teacher_weighting":
        beta = float(repair.settings["beta"])
        return lambda preds_stack: weight_teachers(preds_stack, beta)
    return None


@dataclass
class TeacherWeightStats:
    """Per-batch teacher weights of one pass, summarized into repair telemetry."""

    skipped_batches: int = 0
    weighted_batches: int = 0
    weight_min: float = math.inf
    weight_max: float = -math.inf
    effective_sum: float = 0.0

    def record(self, weights: torch.Tensor | None) -> None:
        if weights is None:
            self.skipped_batches += 1
            return
        self.weighted_batches += 1
        self.weight_min = min(self.weight_min, float(weights.min()))
        self.weight_max = max(self.weight_max, float(weights.max()))
        self.effective_sum += float(1.0 / (weights**2).sum())

    def telemetry(self) -> dict[str, float]:
        """Overrides for :func:`kd_repair_telemetry`; zeros when no batch was weighted."""
        if not self.weighted_batches:
            return {
                "effective_teachers": 0.0,
                "weight_min": 0.0,
                "weight_max": 0.0,
                "skipped_batches": float(self.skipped_batches),
            }
        return {
            "effective_teachers": self.effective_sum / self.weighted_batches,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "skipped_batches": float(self.skipped_batches),
        }


def validate_kd_repair_config(config: Mapping[str, Any]) -> KDRepairConfig:
    """Resolve the repair group from a run config so a bad arm fails before round 1."""
    return resolve_kd_repair(resolve_algorithm_config(config))


def kd_repair_telemetry(
    num_teachers: int,
    *,
    kd_weight: float,
    server_sim_time: float,
    temperature: float = 1.0,
    effective_teachers: float | None = None,
    weight_min: float | None = None,
    weight_max: float | None = None,
    skipped_batches: float = 0.0,
) -> dict[str, float]:
    """Per-round KD-repair telemetry of one distillation pass.

    ``kd_weight`` is the weight of the distilled student in the returned parameters:
    1 for a full pass, 0 when the pass was skipped, and in between on a schedule ramp.
    The teacher-weight fields default to an equal-weight ensemble; the selection and
    weighting families pass their per-batch summary instead, where the effective
    teacher count is the batch mean of ``1 / sum(w^2)``.
    """
    equal_weight = 1.0 / num_teachers if num_teachers > 0 else 0.0
    return {
        "kd_applied_weight": float(kd_weight),
        "kd_effective_teachers": float(
            num_teachers if effective_teachers is None else effective_teachers
        ),
        "kd_teacher_weight_min": float(equal_weight if weight_min is None else weight_min),
        "kd_teacher_weight_max": float(equal_weight if weight_max is None else weight_max),
        "kd_skipped_batches": float(skipped_batches),
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
