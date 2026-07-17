"""Standard local-training fit strategy and its DAdaQuant/FedMAQ variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from fedmaq.core.client_hooks.base import ClientFitStrategy
from fedmaq.core.models import get_model_parameters, set_model_parameters

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient


class StandardFit(ClientFitStrategy):
    """Default FL local training (FedAvg, FedProx, FedPAQ, FedAvgKD, FedDistill).

    The reported ``local_loss`` metric is algorithm-specific and delegated to two
    overridable hooks so subclasses can define it coherently:

    * :meth:`_pretrain_local_loss` — optional loss measured on the *incoming*
      global model, before the training loop (DAdaQuant plateau signal).
    * :meth:`_reported_local_loss` — which value to report given the pre-train and
      final training-batch losses.
    """

    def _pretrain_local_loss(self, client: GenericClient) -> float | None:
        """Loss on the incoming global model before training. Default: not measured."""
        return None

    def _reported_local_loss(
        self, pretrain_loss: float | None, last_loss: float
    ) -> float:
        """Value reported as ``local_loss``. Default: 0.0 (unused by the strategy)."""
        return 0.0

    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        # Load incoming server weights
        set_model_parameters(client.model, parameters)

        # Update compressor hook with dynamic q if provided in configuration
        if "q" in config:
            if hasattr(client.compressor_hook, "q"):
                client.compressor_hook.q = int(config["q"])

        # Optional pre-training loss (e.g. DAdaQuant plateau signal), measured on
        # the incoming global model before any local update.
        pretrain_loss = self._pretrain_local_loss(client)

        # Retrieve round configurations (with defaults from config)
        exp_config = client.config.get("experiment", client.config)
        lr = client._get_decayed_lr(config)
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
        weight_decay = float(exp_config.get("weight_decay", 0.0))
        momentum = float(
            exp_config.get(
                "momentum", client.config.get("algorithm", {}).get("momentum", 0.9)
            )
        )

        # Setup training
        client.model.train()
        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        criterion = nn.CrossEntropyLoss()

        client.loss_hook.on_train_begin(client.model)

        # F14 instrumentation: only active for FedProx, negligible overhead otherwise.
        from fedmaq.core.client import FedProxLossHook

        instrument_fedprox = isinstance(client.loss_hook, FedProxLossHook)
        grad_norm_sum = 0.0
        grad_norm_batches = 0

        last_loss = 0.0
        loss_sum = 0.0
        correct = 0
        total = 0
        batches = 0
        for _ in range(epochs):
            for images, labels in client.trainloader:
                images, labels = images.to(client.device), labels.to(client.device)
                optimizer.zero_grad()
                outputs = client.model(images)
                loss = client.loss_hook.compute_loss(
                    client.model, outputs, labels, criterion, inputs=images
                )
                loss.backward()
                if instrument_fedprox:
                    total_norm = 0.0
                    for p in client.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            total_norm += p.grad.detach().norm(2).item() ** 2
                    grad_norm_sum += total_norm**0.5
                    grad_norm_batches += 1
                optimizer.step()
                last_loss = loss.item()
                loss_sum += last_loss
                batches += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Extract updated parameters
        updated_params = get_model_parameters(client.model)

        # Compute delta = w_new - w_old
        deltas = [u - o for u, o in zip(updated_params, parameters, strict=True)]

        # Compress updates
        compressed_deltas, byte_size = client.compressor_hook.compress(deltas)

        # Reconstruct parameter update: w_new_reconstructed = w_old + compressed_deltas
        reconstructed_params = [
            o + cd for o, cd in zip(parameters, compressed_deltas, strict=True)
        ]

        avg_train_loss = loss_sum / batches if batches > 0 else 0.0
        avg_train_acc = correct / total if total > 0 else 0.0

        fit_metrics = {
            "bytes_uploaded": byte_size,
            "partition_id": int(client.cid),
            "local_loss": self._reported_local_loss(pretrain_loss, last_loss),
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "epochs_trained": epochs,
        }
        if "q" in config:
            fit_metrics["q"] = int(config["q"])

        if instrument_fedprox:
            gn_affine_norm = 0.0
            for module in client.model.modules():
                if isinstance(module, nn.GroupNorm):
                    if module.weight is not None:
                        gn_affine_norm += module.weight.detach().norm(2).item() ** 2
                    if module.bias is not None:
                        gn_affine_norm += module.bias.detach().norm(2).item() ** 2
            fit_metrics["f14_grad_norm"] = (
                grad_norm_sum / grad_norm_batches if grad_norm_batches > 0 else 0.0
            )
            fit_metrics["f14_gn_affine_norm"] = gn_affine_norm**0.5
            fit_metrics["f14_ce_loss"] = client.loss_hook.last_ce
            fit_metrics["f14_prox_penalty"] = client.loss_hook.last_prox

        return (
            reconstructed_params,
            len(client.trainloader.dataset),
            fit_metrics,
        )


class DAdaQuantFit(StandardFit):
    """Standard training that reports the pre-training loss for plateau detection.

    DAdaQuant's server-side hook doubles the global quantization level ``q_t`` when
    the weighted client loss stops improving, so the client measures cross-entropy
    on the incoming global model *before* training and reports it as ``local_loss``.
    """

    def _pretrain_local_loss(self, client: GenericClient) -> float | None:
        client.model.eval()
        loss_sum = 0.0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in client.trainloader:
                images, labels = images.to(client.device), labels.to(client.device)
                outputs = client.model(images)
                loss_sum += criterion(outputs, labels).item() * len(labels)
                total_samples += len(labels)
        return loss_sum / total_samples if total_samples > 0 else 0.0

    def _reported_local_loss(
        self, pretrain_loss: float | None, last_loss: float
    ) -> float:
        return float(pretrain_loss) if pretrain_loss is not None else 0.0


class FedMAQFit(StandardFit):
    """Standard training that reports the final training-batch loss as ``local_loss``."""

    def _reported_local_loss(
        self, pretrain_loss: float | None, last_loss: float
    ) -> float:
        return float(last_loss)
