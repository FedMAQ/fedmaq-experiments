"""Shared knowledge distillation helpers used by multiple strategy hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from torch.utils.data import DataLoader

from fedmaq.core.kd_repair import (
    IMPLEMENTED_KD_REPAIR_FAMILIES,
    TeacherWeighter,
    TeacherWeightStats,
    kd_repair_telemetry,
    kd_temperature,
    resolve_kd_repair,
    schedule_weight,
    teacher_weighter,
)
from fedmaq.core.models import get_model_parameters, set_model_parameters
from fedmaq.core.partitioning import get_server_loaders

logger = logging.getLogger(__name__)


def apply_student_ema(
    aggregated_parameters: Parameters,
    ema_params: list[Any] | None,
    alg_cfg: dict[str, Any],
) -> tuple[Parameters, list[Any] | None]:
    """Smooth the post-distillation global student with an EMA across rounds.

    Shared by every hook that runs server-side KD. The refinement is independent
    of quantization, so manuscript §4.3.7's parity requirement applies it to the
    no-quantization arm (Configuration 6) as well as to the FedMAQ arms; keeping
    one implementation is what makes that parity checkable rather than asserted.

    Returns the (possibly smoothed) parameters and the updated EMA state, so the
    caller owns the per-run state without duplicating the update rule.
    """
    if not alg_cfg.get("ema_student", False):
        return aggregated_parameters, ema_params

    ema_decay = float(alg_cfg.get("ema_decay", 0.99))
    new_params = parameters_to_ndarrays(aggregated_parameters)
    if ema_params is None:
        ema_params = [p.copy() for p in new_params]
    else:
        ema_params = [
            ema_decay * ema + (1.0 - ema_decay) * new
            for ema, new in zip(ema_params, new_params, strict=True)
        ]
    return ndarrays_to_parameters(ema_params), ema_params


def kd_distill_step(
    student_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    kl_criterion: nn.Module,
    images: torch.Tensor,
    teacher_soft_preds: torch.Tensor,
    temperature: float,
) -> float:
    """One SGD step distilling ``teacher_soft_preds`` into ``student_model`` via KL divergence."""
    optimizer.zero_grad()
    student_logits = student_model(images)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
    loss = kl_criterion(student_log_soft, teacher_soft_preds) * (temperature**2)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_server_side_kd(
    student_model: nn.Module,
    teachers: list[nn.Module],
    public_loader: DataLoader,
    temperature: float,
    learning_rate: float,
    momentum: float,
    device: torch.device,
    epochs: int = 1,
    teacher_bit_widths: list[int] | None = None,
    entropy_weight_scale: float = 1.0,
    precision_weight_scale: float = 1.0,
    teacher_weighter: TeacherWeighter | None = None,
    weight_stats: TeacherWeightStats | None = None,
) -> float:
    """Run server-side knowledge distillation to transfer ensemble knowledge to student model.

    The student is updated in-place via SGD minimising KL divergence against the
    soft-label average of all teacher outputs.

    ``teacher_weighter`` (the selection and weighting repair families) returns
    per-teacher weights for each batch, which multiply into the ensemble weights;
    equal weights leave the target untouched, and None skips the batch. Each
    batch's weights are recorded in ``weight_stats``.
    """
    optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum)
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    student_model.train()
    loss_sum = 0.0
    batches = 0
    for _ in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)

            with torch.no_grad():
                teacher_soft_preds_list = []
                for teacher in teachers:
                    t_out = teacher(images)
                    teacher_soft_preds_list.append(F.softmax(t_out / temperature, dim=1))

                reweight = None
                if teacher_weighter is not None:
                    repair_weights = teacher_weighter(torch.stack(teacher_soft_preds_list))
                    if weight_stats is not None:
                        weight_stats.record(repair_weights)
                    if repair_weights is None:
                        continue
                    if not torch.all(repair_weights == repair_weights[0]):
                        reweight = repair_weights

                if teacher_bit_widths is not None:
                    # Per-sample entropy weighting + precision scaling
                    preds_stack = torch.stack(teacher_soft_preds_list)  # [T, B, C]
                    eps = 1e-8
                    entropy = -torch.sum(
                        preds_stack * torch.log(preds_stack + eps), dim=2
                    )  # [T, B]
                    entropy_weights = torch.exp(-entropy_weight_scale * entropy)  # [T, B]

                    q_max = max(teacher_bit_widths)
                    precision_weights = (
                        torch.tensor(
                            [q / q_max for q in teacher_bit_widths],
                            device=device,
                            dtype=torch.float32,
                        ).unsqueeze(1)
                        ** precision_weight_scale
                    )  # [T, 1]

                    combined = entropy_weights * precision_weights  # [T, B]
                    if reweight is not None:
                        combined = combined * reweight.unsqueeze(1)
                    combined = combined / (
                        combined.sum(dim=0, keepdim=True) + eps
                    )  # normalize over teachers

                    teacher_soft_preds = (preds_stack * combined.unsqueeze(2)).sum(dim=0)  # [B, C]
                elif reweight is not None:
                    teacher_soft_preds = (
                        torch.stack(teacher_soft_preds_list) * reweight.view(-1, 1, 1)
                    ).sum(dim=0)
                else:
                    teacher_soft_preds = torch.stack(teacher_soft_preds_list).mean(dim=0)

            loss_sum += kd_distill_step(
                student_model, optimizer, kl_criterion, images, teacher_soft_preds, temperature
            )
            batches += 1

    return loss_sum / batches if batches > 0 else 0.0


def kd_server_sim_time(
    num_public: int,
    kd_epochs: int,
    num_teachers: int,
    server_compute_speed: float,
) -> float:
    """Simulated server-side distillation time for ensemble KD, in seconds.

    Scales with the proxy-set size, KD epochs, and number of teacher models,
    divided by the (simulated) server compute speed. Shared by the FedMAQ and
    FedAvgKD telemetry paths.
    """
    if server_compute_speed <= 0.0:
        return 0.0
    return (num_public * kd_epochs * num_teachers) / server_compute_speed


def distill_ensemble_into_global(
    model_factory: Callable[[str, int], nn.Module],
    aggregated_parameters: Parameters,
    results: list[tuple[ClientProxy, FitRes]],
    public_indices: list[int] | None,
    dataset_name: str,
    num_classes: int,
    batch_size: int,
    alg_cfg: dict[str, Any],
    device: torch.device,
    teacher_bit_widths: list[int] | None = None,
    *,
    server_round: int | None = None,
    class_counts: list[list[int]] | None = None,
    num_public_samples: int | None = None,
    server_compute_speed: float = 0.0,
) -> tuple[Parameters, dict[str, float]]:
    """Refine an aggregated global model via ensemble server-side KD.

    Shared body of the FedMAQ and FedAvgKD ``aggregate_fit`` hooks. ``model_factory``
    (e.g. ``get_model`` or ``get_kd_student_model``) builds both the student -- seeded
    with ``aggregated_parameters`` -- and each client teacher from its returned
    parameters, then the teacher ensemble is distilled into the student over the
    server's public dataset.

    This routine is the single seam of the ADR-0028 KD-repair families. It receives
    the server round and the participants' class counts (``class_counts[i]`` belongs
    to ``results[i]``), reads the arm's ``kd_repair`` group, and reports per-round
    repair telemetry. With every family off the pass is plain ensemble KD. The
    schedule family weights the distilled student by round (a zero weight skips the
    pass); the temperature family replaces the distillation temperature; the
    teacher selection and weighting families reweight the ensemble per batch, and a
    pass whose every batch they skip counts as skipped. Families that are not
    implemented yet are refused.

    Returns the updated ``Parameters`` and a dictionary of KD metrics. Falls back to
    ``aggregated_parameters`` and skip metrics when there are no loadable teachers,
    no public set, or an error occurs.
    """
    repair = resolve_kd_repair(alg_cfg)
    if repair.family is not None and repair.family not in IMPLEMENTED_KD_REPAIR_FAMILIES:
        raise NotImplementedError(f"KD-repair family {repair.family!r} is not implemented")
    del class_counts  # consumed by the class-balanced family once it lands

    temperature = kd_temperature(repair, alg_cfg)
    kd_weight = 1.0
    if repair.family == "schedule":
        if server_round is None:
            raise ValueError("The KD-repair schedule family needs the server round")
        kd_weight = schedule_weight(repair.settings, server_round)
    if kd_weight == 0.0:
        return aggregated_parameters, {
            "dropped_teachers": 0.0,
            "kd_skipped": 1.0,
            **kd_repair_telemetry(
                len(results), kd_weight=0.0, server_sim_time=0.0, temperature=temperature
            ),
        }

    kd_epochs = int(alg_cfg.get("kd_epochs", 1))
    if num_public_samples is None:
        num_public_samples = len(public_indices) if public_indices is not None else 0
    # Same units as the hooks' KD term in server_sim_time: every result counts as a teacher.
    kd_time = kd_server_sim_time(
        num_public=num_public_samples,
        kd_epochs=kd_epochs,
        num_teachers=len(results),
        server_compute_speed=server_compute_speed,
    )

    weighter = teacher_weighter(repair)
    weight_stats = TeacherWeightStats() if weighter is not None else None

    student_model = model_factory(dataset_name, num_classes)
    student_model.to(device)
    set_model_parameters(student_model, parameters_to_ndarrays(aggregated_parameters))

    teachers: list[nn.Module] = []
    actual_bit_widths: list[int] | None = [] if teacher_bit_widths is not None else None
    dropped_teachers = 0
    for i, (_, fit_res) in enumerate(results):
        try:
            teacher = model_factory(dataset_name, num_classes)
            set_model_parameters(teacher, parameters_to_ndarrays(fit_res.parameters))
            teacher.eval()
            teacher.to(device)
            teachers.append(teacher)
            if actual_bit_widths is not None:
                assert teacher_bit_widths is not None
                actual_bit_widths.append(teacher_bit_widths[i])
        except (ValueError, RuntimeError):
            # F6: an arch/shape mismatch (what set_model_parameters' strict checks
            # raise) is a config bug — fail loud instead of silently distilling from
            # a degraded ensemble.
            raise
        except Exception as exc:
            # Genuinely transient fault: drop this teacher but record it so the
            # silent degradation is observable in run telemetry (F6).
            dropped_teachers += 1
            logger.warning(f"Failed to load client model from parameters: {exc}")

    if not (teachers and public_indices is not None):
        return aggregated_parameters, {
            "dropped_teachers": float(dropped_teachers),
            "kd_skipped": 1.0,
            **kd_repair_telemetry(
                len(teachers), kd_weight=0.0, server_sim_time=kd_time, temperature=temperature
            ),
        }

    try:
        public_loader, _, _ = get_server_loaders(
            dataset_name, public_indices, batch_size=batch_size
        )
        kd_loss = run_server_side_kd(
            student_model=student_model,
            teachers=teachers,
            public_loader=public_loader,
            temperature=temperature,
            learning_rate=float(alg_cfg.get("server_kd_lr", 0.01)),
            momentum=float(alg_cfg.get("server_kd_momentum", 0.9)),
            epochs=kd_epochs,
            device=device,
            teacher_bit_widths=actual_bit_widths,
            entropy_weight_scale=float(alg_cfg.get("entropy_weight", 1.0)),
            precision_weight_scale=float(alg_cfg.get("precision_weight", 1.0)),
            teacher_weighter=weighter,
            weight_stats=weight_stats,
        )
        weight_telemetry = weight_stats.telemetry() if weight_stats is not None else {}
        if weight_stats is not None and weight_stats.weighted_batches == 0:
            return aggregated_parameters, {
                "dropped_teachers": float(dropped_teachers),
                "kd_skipped": 1.0,
                **kd_repair_telemetry(
                    len(teachers),
                    kd_weight=0.0,
                    server_sim_time=kd_time,
                    temperature=temperature,
                    **weight_telemetry,
                ),
            }
        distilled = get_model_parameters(student_model)
        if kd_weight < 1.0:
            distilled = _blend(parameters_to_ndarrays(aggregated_parameters), distilled, kd_weight)
        updated = ndarrays_to_parameters(distilled)
        logger.info(
            f"Server-side KD: successfully distilled knowledge from {len(teachers)} teacher models."
        )
        return updated, {
            "server_kd_loss": kd_loss,
            "dropped_teachers": float(dropped_teachers),
            "kd_skipped": 0.0,
            **kd_repair_telemetry(
                len(teachers),
                kd_weight=kd_weight,
                server_sim_time=kd_time,
                temperature=temperature,
                **weight_telemetry,
            ),
        }
    except (ValueError, RuntimeError):
        # F6: shape/config bug in the KD pass — fail loud, don't return an
        # un-distilled aggregate that looks like a normal round.
        raise
    except Exception as exc:
        logger.error(f"Error during server-side KD: {exc}")
        return aggregated_parameters, {
            "dropped_teachers": float(dropped_teachers),
            "kd_skipped": 1.0,
            **kd_repair_telemetry(
                len(teachers), kd_weight=0.0, server_sim_time=kd_time, temperature=temperature
            ),
        }


def _blend(
    aggregated: list[np.ndarray], distilled: list[np.ndarray], weight: float
) -> list[np.ndarray]:
    """``aggregated + weight * (distilled - aggregated)`` per floating array.

    Integer buffers (BatchNorm's ``num_batches_tracked``) keep the distilled value.
    """
    blended = []
    for before, after in zip(aggregated, distilled, strict=True):
        if np.issubdtype(after.dtype, np.floating):
            blended.append((before + weight * (after - before)).astype(after.dtype))
        else:
            blended.append(after)
    return blended
