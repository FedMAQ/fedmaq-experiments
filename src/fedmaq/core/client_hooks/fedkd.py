"""FedKD client-side fit strategy (student-teacher mutual distillation)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedmaq.core.client_hooks.base import ClientFitStrategy
from fedmaq.core.models import (
    get_kd_teacher_model,
    get_model_parameters,
    set_model_parameters,
)

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient


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
        persistence_dir = client.config.get("experiment", {}).get(
            "persistence_dir", ".data_partitions/fedkd_models"
        )
        model_dir = Path(persistence_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = model_dir / f"teacher_{client.cid}.pth"

        # 1. Instantiate teacher model based on dataset
        dataset_name = client.config.get("dataset", {}).get("name", "")
        num_classes = int(client.config.get("dataset", {}).get("num_classes", 10))
        teacher_model = get_kd_teacher_model(dataset_name, num_classes)
        teacher_model.to(client.device)

        # 2. Load teacher weights if file exists, otherwise keep random initialization
        if teacher_path.exists():
            teacher_model.load_state_dict(
                torch.load(teacher_path, map_location=client.device)
            )

        # 3. Load global student parameters
        set_model_parameters(client.model, parameters)

        # 4. Setup Joint Optimizer
        exp_config = client.config.get("experiment", client.config)
        lr = client._get_decayed_lr(config)
        weight_decay = float(exp_config.get("weight_decay", 0.0))
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))

        optimizer = torch.optim.SGD(
            list(client.model.parameters()) + list(teacher_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        ce_criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        temperature = float(client.config.get("algorithm", {}).get("temperature", 2.0))

        # 5. Local Training: Student-Teacher Mutual Distillation
        client.model.train()
        teacher_model.train()
        for _ in range(epochs):
            for images, labels in client.trainloader:
                images, labels = images.to(client.device), labels.to(client.device)
                optimizer.zero_grad()

                # Forward pass
                outputs_s = client.model(images)
                outputs_t = teacher_model(images)

                # Task Loss
                loss_s_task = ce_criterion(outputs_s, labels)
                loss_t_task = ce_criterion(outputs_t, labels)

                # Soft predictions for KL divergence
                outputs_s_log_soft = F.log_softmax(outputs_s / temperature, dim=1)
                outputs_t_log_soft = F.log_softmax(outputs_t / temperature, dim=1)
                outputs_s_soft = F.softmax(outputs_s / temperature, dim=1)
                outputs_t_soft = F.softmax(outputs_t / temperature, dim=1)

                # Mutual Knowledge Distillation Loss (scaled by temperature^2)
                kl_t_to_s = kl_criterion(outputs_s_log_soft, outputs_t_soft) * (
                    temperature**2
                )
                kl_s_to_t = kl_criterion(outputs_t_log_soft, outputs_s_soft) * (
                    temperature**2
                )

                # Adaptive scaling: divide by sum of task losses
                denom = loss_s_task + loss_t_task + 1e-6
                loss_kd_s = kl_t_to_s / denom
                loss_kd_t = kl_s_to_t / denom

                # Joint optimization loss
                loss_s = loss_s_task + loss_kd_s
                loss_t = loss_t_task + loss_kd_t
                total_loss = loss_s + loss_t

                total_loss.backward()
                optimizer.step()

        # 6. Save updated teacher model parameters
        torch.save(teacher_model.state_dict(), teacher_path)

        # 7. Extract updated student model parameters and compute delta
        updated_params = get_model_parameters(client.model)
        deltas = [u - o for u, o in zip(updated_params, parameters, strict=True)]

        # 8. Update compressor hook with dynamic energy if provided in configuration
        if "energy" in config:
            if hasattr(client.compressor_hook, "energy"):
                client.compressor_hook.energy = float(config["energy"])

        # 9. Compress updates
        compressed_deltas, byte_size = client.compressor_hook.compress(deltas)

        # Reconstruct parameter update: w_new_reconstructed = w_old + compressed_deltas
        reconstructed_params = [
            o + cd for o, cd in zip(parameters, compressed_deltas, strict=True)
        ]

        return (
            reconstructed_params,
            len(client.trainloader.dataset),
            {
                "bytes_uploaded": byte_size,
                "partition_id": int(client.cid),
                "local_loss": 0.0,
            },
        )
