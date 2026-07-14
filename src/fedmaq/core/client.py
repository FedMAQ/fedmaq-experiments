"""Generic Flower Client implementation with customizable hooks for loss and compression."""

from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Config

from fedmaq.core.client_hooks import ClientFitStrategy, get_fit_strategy
from fedmaq.core.models import DEVICE


class LossHook:
    """Base class for customizing local training loss functions."""

    def on_train_begin(self, model: nn.Module) -> None:
        """Hook called before the first training batch."""
        pass

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the local loss."""
        return criterion(outputs, targets)


class FedProxLossHook(LossHook):
    """Loss hook adding proximal L2 regularization for FedProx."""

    def __init__(self, mu: float = 0.01) -> None:
        self.mu = mu
        self.global_params: list[torch.Tensor] = []

    def on_train_begin(self, model: nn.Module) -> None:
        # Save a frozen copy of the initial global weights
        self.global_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = criterion(outputs, targets)
        proximal_term = 0.0
        params = [p for p in model.parameters() if p.requires_grad]
        for p, gp in zip(params, self.global_params, strict=True):
            proximal_term += torch.sum((p - gp) ** 2)
        return loss + (self.mu / 2.0) * proximal_term


class CompressionHook:
    """Base class for compressing client model updates (deltas)."""

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas and return (compressed_deltas, byte_size)."""
        # Default: Identity (uncompressed Float32 weights -> 4 bytes per element)
        byte_size = sum(d.nbytes for d in deltas)
        return deltas, byte_size


def get_loss_hook(alg_name: str, alg_cfg: dict[str, Any]) -> LossHook:
    """Factory: return the appropriate LossHook for the given algorithm name."""
    if alg_name == "fedprox":
        return FedProxLossHook(mu=float(alg_cfg.get("mu", 0.01)))
    if alg_name in {"fedmaq", "fedmaq_lite"}:
        if alg_cfg.get("client_kd_reg", False):
            from fedmaq.core.kd_loss_hook import ClientKDLossHook

            return ClientKDLossHook(
                alpha=float(alg_cfg.get("kd_reg_alpha", 0.5)),
                temperature=float(alg_cfg.get("kd_reg_temp", 2.0)),
            )
    return LossHook()


class GenericClient(fl.client.NumPyClient):
    """Extensible client wrapping a PyTorch model and executing local epochs."""

    def __init__(
        self,
        cid: str,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_hook: LossHook,
        compressor_hook: CompressionHook,
        config: dict[str, Any],
        public_loader: torch.utils.data.DataLoader | None = None,
        state: Any | None = None,
    ) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.loss_hook = loss_hook
        self.compressor_hook = compressor_hook
        self.config = config
        self.public_loader = public_loader
        self.state = state
        self.device = torch.device(config.get("device") or DEVICE)
        self.model.to(self.device)

        alg_name = config.get("algorithm", {}).get("name", "")
        self.fit_strategy: ClientFitStrategy = get_fit_strategy(alg_name)

    def get_properties(self, config: Config) -> dict[str, Any]:
        return {"cid": self.cid}

    def _get_decayed_lr(self, config: dict[str, Any]) -> float:
        exp_config = self.config.get("experiment", self.config)
        base_lr = float(config.get("lr", exp_config.get("learning_rate", 0.01)))
        lr_decay = float(exp_config.get("learning_rate_decay", 1.0))
        server_round = int(config.get("server_round", 1))
        return base_lr * (lr_decay ** (server_round - 1))

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        return self.fit_strategy.fit(self, parameters, config)

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        return self.fit_strategy.evaluate(self, parameters, config)
