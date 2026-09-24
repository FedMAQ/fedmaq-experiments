"""Standard local-training fit strategy and its DAdaQuant/FedMAQ variants."""

from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn

from fedmaq.core.client_hooks.base import ClientFitStrategy, attach_payloads_if_enabled
from fedmaq.core.client_hooks.training_skeleton import (
    StepResult,
    compress_and_reconstruct,
    run_epochs,
)
from fedmaq.core.config_defaults import resolve_algorithm_config, resolve_experiment_config
from fedmaq.core.models import get_model_parameters, set_model_parameters
from fedmaq.core.randomness import derive_numpy_rng

if TYPE_CHECKING:
    from fedmaq.baselines.transport import UploadReport
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

    def _reported_local_loss(self, pretrain_loss: float | None, last_loss: float) -> float:
        """Value reported as ``local_loss``. Default: 0.0 (unused by the strategy)."""
        return 0.0

    def _extra_fit_metrics(self, report: UploadReport) -> dict[str, Any]:
        """Additional per-round ``fit_metrics`` beyond the shared fields below.

        Default: none. Overridden by :class:`DAdaQuantFit` to report the
        secondary byte axis (#26) when the compressor hook measured one.
        """
        return {}

    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        set_model_parameters(client.model, parameters)

        if "q" in config:
            if hasattr(client.compressor_hook, "q"):
                client.compressor_hook.q = int(config["q"])

        if hasattr(client.compressor_hook, "rng"):
            # Reseed per round, not just at client_fn construction time: Flower
            # rebuilds client_fn fresh every message, so a construction-time-only
            # seed replays the identical stochastic-rounding draw sequence every
            # round for a given client, breaking cross-round independence (#24).
            # `seed` and `server_round` are both read directly (no default): a
            # silent fallback for either would quietly reseed the compression
            # stream out of step with the rest of the run under a fixed seed
            # (ADR-0006), the exact bug this reseed fixes. Combined via the same
            # (seed, partition_id, round) shape as quantization_planner.py:268's
            # dataloader seed, but as a tuple rather than summed into one int, so
            # it draws from a stream independent of that (and the training) RNG.
            seed = int(client.config["seed"])
            partition_id = int(client.cid)
            server_round = int(config["server_round"])
            client.compressor_hook.rng = derive_numpy_rng(
                "compression", seed, partition_id, server_round
            )

        # Optional pre-training loss (e.g. DAdaQuant plateau signal), measured on
        # the incoming global model before any local update.
        pretrain_loss = self._pretrain_local_loss(client)

        exp_config = resolve_experiment_config(client.config)
        alg_config = resolve_algorithm_config(client.config)
        lr = client._get_decayed_lr(config)
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
        weight_decay = float(exp_config.get("weight_decay", 0.0))
        momentum = float(exp_config.get("momentum", alg_config.get("momentum", 0.9)))

        client.model.train()
        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        criterion = nn.CrossEntropyLoss()

        client.loss_hook.on_train_begin(client.model)
        from fedmaq.core.client import read_training_metrics

        initial_training_metrics = read_training_metrics(client.loss_hook)
        instrument_training_metrics = {
            "f14_ce_loss",
            "f14_prox_penalty",
        }.issubset(initial_training_metrics)

        grad_norm_sum = 0.0
        grad_norm_batches = 0

        def step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
            outputs = client.model(images)
            loss = client.loss_hook.compute_loss(
                client.model, outputs, labels, criterion, inputs=images
            )
            _, predicted = torch.max(outputs.data, 1)
            return StepResult(
                loss=loss,
                correct=int((predicted == labels).sum().item()),
                total=int(labels.size(0)),
            )

        def on_after_backward() -> None:
            nonlocal grad_norm_sum, grad_norm_batches
            total_norm = 0.0
            for p in client.model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm += p.grad.detach().norm(2).item() ** 2
            grad_norm_sum += total_norm**0.5
            grad_norm_batches += 1

        result = run_epochs(
            model=client.model,
            loader=client.trainloader,
            optimizer=optimizer,
            epochs=epochs,
            step_fn=step_fn,
            device=client.device,
            on_after_backward=on_after_backward if instrument_training_metrics else None,
        )

        updated_params = get_model_parameters(client.model)
        reconstructed_params, report = compress_and_reconstruct(
            parameters, updated_params, client.compressor_hook
        )

        fit_metrics = {
            "bytes_uploaded": report.measured_bytes,
            "payload_bytes": report.payload_bytes,
            "partition_id": int(client.cid),
            "local_loss": self._reported_local_loss(pretrain_loss, result.last_loss),
            "train_loss": result.avg_loss,
            "train_acc": result.accuracy if result.accuracy is not None else 0.0,
            "epochs_trained": epochs,
        }
        if "q" in config:
            fit_metrics["q"] = int(config["q"])
        if (
            client.config.get("v2_diagnostic", {}).get("enabled", False)
            and alg_config.get("name") == "fedmaq"
        ):
            raw_sq = 0.0
            error_sq = 0.0
            for original, updated, reconstructed in zip(
                parameters, updated_params, reconstructed_params, strict=True
            ):
                raw_delta = updated.astype(np.float64) - original.astype(np.float64)
                reconstructed_delta = reconstructed.astype(np.float64) - original.astype(np.float64)
                raw_sq += float(np.sum(raw_delta * raw_delta))
                difference = raw_delta - reconstructed_delta
                error_sq += float(np.sum(difference * difference))
            fit_metrics["v2_raw_update_norm"] = raw_sq**0.5
            fit_metrics["v2_raw_to_reconstructed_norm"] = error_sq**0.5
            residual_norm = getattr(client.compressor_hook, "diagnostic_residual_norm", None)
            if callable(residual_norm):
                fit_metrics["v2_corrected_quantization_residual_norm"] = residual_norm()
        fit_metrics.update(self._extra_fit_metrics(report))
        attach_payloads_if_enabled(client, fit_metrics, report.payloads)

        training_metrics = read_training_metrics(client.loss_hook)
        fit_metrics.update(training_metrics)
        if instrument_training_metrics:
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

        return (
            reconstructed_params,
            len(cast(Sized, client.trainloader.dataset)),
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

    def _reported_local_loss(self, pretrain_loss: float | None, last_loss: float) -> float:
        return float(pretrain_loss) if pretrain_loss is not None else 0.0

    def _extra_fit_metrics(self, report: UploadReport) -> dict[str, Any]:
        if report.secondary_bytes is None:
            return {}
        return {"secondary_bytes_uploaded": report.secondary_bytes}


class FedMAQFit(StandardFit):
    """Standard training that reports the final training-batch loss as ``local_loss``."""

    def _reported_local_loss(self, pretrain_loss: float | None, last_loss: float) -> float:
        return float(last_loss)
