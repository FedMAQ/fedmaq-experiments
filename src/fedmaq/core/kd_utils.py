"""Shared knowledge distillation helpers used by multiple strategy hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

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

from fedmaq.core.models import get_model_parameters, set_model_parameters
from fedmaq.core.partitioning import get_server_loaders

logger = logging.getLogger(__name__)


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
) -> float:
    """Run server-side knowledge distillation to transfer ensemble knowledge to student model.

    The student is updated in-place via SGD minimising KL divergence against the
    soft-label average of all teacher outputs.
    """
    optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum)
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    student_model.train()
    loss_sum = 0.0
    batches = 0
    for _ in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)

            # Get soft targets from teachers
            with torch.no_grad():
                teacher_soft_preds_list = []
                for teacher in teachers:
                    t_out = teacher(images)
                    teacher_soft_preds_list.append(F.softmax(t_out / temperature, dim=1))

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
                    combined = combined / (
                        combined.sum(dim=0, keepdim=True) + eps
                    )  # normalize over teachers

                    teacher_soft_preds = (preds_stack * combined.unsqueeze(2)).sum(dim=0)  # [B, C]
                else:
                    teacher_soft_preds = torch.stack(teacher_soft_preds_list).mean(dim=0)

            optimizer.zero_grad()
            student_logits = student_model(images)
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)

            # Distillation loss
            loss = kl_criterion(student_log_soft, teacher_soft_preds) * (temperature**2)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
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
) -> tuple[Parameters, dict[str, float]]:
    """Refine an aggregated global model via ensemble server-side KD.

    Shared body of the FedMAQ and FedAvgKD ``aggregate_fit`` hooks. ``model_factory``
    (e.g. ``get_model`` or ``get_kd_student_model``) builds both the student -- seeded
    with ``aggregated_parameters`` -- and each client teacher from its returned
    parameters, then the teacher ensemble is distilled into the student over the
    server's public dataset.

    Returns the updated ``Parameters`` and a dictionary of KD metrics. Falls back to
    ``aggregated_parameters`` and empty metrics when there are no loadable teachers,
    no public set, or an error occurs.
    """
    student_model = model_factory(dataset_name, num_classes)
    student_model.to(device)
    set_model_parameters(student_model, parameters_to_ndarrays(aggregated_parameters))

    teachers: list[nn.Module] = []
    actual_bit_widths: list[int] = [] if teacher_bit_widths is not None else None
    for i, (_, fit_res) in enumerate(results):
        try:
            teacher = model_factory(dataset_name, num_classes)
            set_model_parameters(teacher, parameters_to_ndarrays(fit_res.parameters))
            teacher.eval()
            teacher.to(device)
            teachers.append(teacher)
            if actual_bit_widths is not None:
                actual_bit_widths.append(teacher_bit_widths[i])
        except Exception as exc:
            logger.warning(f"Failed to load client model from parameters: {exc}")

    if not (teachers and public_indices is not None):
        return aggregated_parameters, {}

    try:
        public_loader, _ = get_server_loaders(dataset_name, public_indices, batch_size=batch_size)
        kd_loss = run_server_side_kd(
            student_model=student_model,
            teachers=teachers,
            public_loader=public_loader,
            temperature=float(alg_cfg.get("temperature", 1.0)),
            learning_rate=float(alg_cfg.get("server_kd_lr", 0.01)),
            momentum=float(alg_cfg.get("server_kd_momentum", 0.9)),
            epochs=int(alg_cfg.get("kd_epochs", 1)),
            device=device,
            teacher_bit_widths=actual_bit_widths,
            entropy_weight_scale=float(alg_cfg.get("entropy_weight", 1.0)),
            precision_weight_scale=float(alg_cfg.get("precision_weight", 1.0)),
        )
        updated = ndarrays_to_parameters(get_model_parameters(student_model))
        logger.info(
            f"Server-side KD: successfully distilled knowledge from {len(teachers)} teacher models."
        )
        return updated, {"server_kd_loss": kd_loss}
    except Exception as exc:
        logger.error(f"Error during server-side KD: {exc}")
        return aggregated_parameters, {}
