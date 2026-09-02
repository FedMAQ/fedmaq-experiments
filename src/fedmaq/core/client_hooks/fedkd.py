"""FedKD client-side fit strategy (student-teacher mutual distillation)."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedmaq.core.client_hooks.base import ClientFitStrategy, attach_payloads_if_enabled
from fedmaq.core.client_hooks.training_skeleton import compress_and_reconstruct
from fedmaq.core.config_defaults import (
    resolve_algorithm_config,
    resolve_experiment_config,
    resolve_run_context,
)
from fedmaq.core.models import (
    get_kd_teacher_model,
    get_model_parameters,
    set_model_parameters,
)

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient


def inverse_task_loss_weight(
    student_task_loss: torch.Tensor, teacher_task_loss: torch.Tensor
) -> torch.Tensor:
    """Return the continuous inverse-loss weight for mutual distillation."""
    return 1.0 / (student_task_loss + teacher_task_loss + 1e-6)


class FedKDFit(ClientFitStrategy):
    """FedKD: joint student-teacher training with mutual KL distillation.

    The persistent per-client teacher and the global student are optimized together;
    the student delta is SVD-compressed (via the injected compressor hook) for upload.
    """

    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        exp_config = resolve_experiment_config(client.config)
        persistence_dir = client.config.get("_persistence_dir") or exp_config.get(
            "persistence_dir", ".data_partitions/fedkd_models"
        )
        model_dir = Path(persistence_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = model_dir / f"teacher_{client.cid}.pth"

        context = resolve_run_context(client.config)
        dataset_name = context.dataset_name
        num_classes = context.num_classes
        teacher_model = get_kd_teacher_model(dataset_name, num_classes)
        teacher_model.to(client.device)

        if teacher_path.exists():
            teacher_model.load_state_dict(torch.load(teacher_path, map_location=client.device))

        set_model_parameters(client.model, parameters)

        lr = client._get_decayed_lr(config)
        weight_decay = float(exp_config.get("weight_decay", 0.0))
        alg_config = resolve_algorithm_config(client.config)
        momentum = float(alg_config.get("momentum", exp_config.get("momentum", 0.9)))
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))

        optimizer = torch.optim.SGD(
            list(client.model.parameters()) + list(teacher_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        ce_criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        temperature = float(alg_config.get("temperature", 2.0))

        client.model.train()
        teacher_model.train()
        total_loss_sum = 0.0
        loss_kd_s_sum = 0.0
        loss_kd_t_sum = 0.0
        loss_s_task_sum = 0.0
        loss_t_task_sum = 0.0
        correct_s = 0
        correct_t = 0
        total_samples = 0
        batches = 0

        for _ in range(epochs):
            for images, labels in client.trainloader:
                images, labels = images.to(client.device), labels.to(client.device)
                optimizer.zero_grad()

                outputs_s = client.model(images)
                outputs_t = teacher_model(images)

                loss_s_task = ce_criterion(outputs_s, labels)
                loss_t_task = ce_criterion(outputs_t, labels)

                outputs_s_log_soft = F.log_softmax(outputs_s / temperature, dim=1)
                outputs_t_log_soft = F.log_softmax(outputs_t / temperature, dim=1)
                outputs_s_soft = F.softmax(outputs_s / temperature, dim=1)
                outputs_t_soft = F.softmax(outputs_t / temperature, dim=1)

                kl_t_to_s = kl_criterion(outputs_s_log_soft, outputs_t_soft) * (temperature**2)
                kl_s_to_t = kl_criterion(outputs_t_log_soft, outputs_s_soft) * (temperature**2)

                inverse_loss_weight = inverse_task_loss_weight(loss_s_task, loss_t_task)
                loss_kd_s = kl_t_to_s * inverse_loss_weight
                loss_kd_t = kl_s_to_t * inverse_loss_weight

                loss_s = loss_s_task + loss_kd_s
                loss_t = loss_t_task + loss_kd_t
                total_loss = loss_s + loss_t

                total_loss.backward()
                optimizer.step()

                total_loss_sum += total_loss.item()
                loss_kd_s_sum += loss_kd_s.item()
                loss_kd_t_sum += loss_kd_t.item()
                loss_s_task_sum += loss_s_task.item()
                loss_t_task_sum += loss_t_task.item()
                batches += 1

                _, pred_s = torch.max(outputs_s.data, 1)
                _, pred_t = torch.max(outputs_t.data, 1)
                total_samples += labels.size(0)
                correct_s += (pred_s == labels).sum().item()
                correct_t += (pred_t == labels).sum().item()

        torch.save(teacher_model.state_dict(), teacher_path)

        if "energy" in config:
            if hasattr(client.compressor_hook, "energy"):
                client.compressor_hook.energy = float(config["energy"])

        updated_params = get_model_parameters(client.model)
        reconstructed_params, report = compress_and_reconstruct(
            parameters, updated_params, client.compressor_hook
        )

        avg_total_loss = total_loss_sum / batches if batches > 0 else 0.0
        avg_train_acc = correct_s / total_samples if total_samples > 0 else 0.0
        avg_teacher_acc = correct_t / total_samples if total_samples > 0 else 0.0
        avg_kd_loss_student = loss_kd_s_sum / batches if batches > 0 else 0.0
        avg_kd_loss_teacher = loss_kd_t_sum / batches if batches > 0 else 0.0
        avg_task_loss_student = loss_s_task_sum / batches if batches > 0 else 0.0
        avg_task_loss_teacher = loss_t_task_sum / batches if batches > 0 else 0.0

        fit_metrics = {
            "bytes_uploaded": report.measured_bytes,
            "payload_bytes": report.payload_bytes,
            "partition_id": int(client.cid),
            "local_loss": avg_total_loss,
            "train_loss": avg_total_loss,
            "train_acc": avg_train_acc,
            "epochs_trained": epochs,
            "kd_loss_student": avg_kd_loss_student,
            "kd_loss_teacher": avg_kd_loss_teacher,
            "task_loss_student": avg_task_loss_student,
            "task_loss_teacher": avg_task_loss_teacher,
            "teacher_acc": avg_teacher_acc,
        }
        attach_payloads_if_enabled(client, fit_metrics, report.payloads)

        return (
            reconstructed_params,
            len(cast(Sized, client.trainloader.dataset)),
            fit_metrics,
        )
