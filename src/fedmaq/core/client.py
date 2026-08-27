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
        # F14 instrumentation: last-batch CE/proximal magnitudes (cheap, always-on).
        self.last_ce: float = 0.0
        self.last_prox: float = 0.0

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
        ce_loss = criterion(outputs, targets)
        proximal_term: torch.Tensor | float = 0.0
        params = [p for p in model.parameters() if p.requires_grad]
        for p, gp in zip(params, self.global_params, strict=True):
            proximal_term += torch.sum((p - gp) ** 2)
        prox_penalty = (self.mu / 2.0) * proximal_term
        self.last_ce = float(ce_loss.item())
        self.last_prox = (
            float(prox_penalty.item())
            if isinstance(prox_penalty, torch.Tensor)
            else float(prox_penalty)
        )
        return ce_loss + prox_penalty


class CompressionHook:
    """Base class for compressing client model updates (deltas).

    ``last_payload_bytes`` is the pre-encoding payload size from the most
    recent ``compress()`` call — logged alongside the post-encoding
    ``bytes_uploaded`` so a future encoder change can be re-scored offline
    without re-running training (see ``TelemetryManager.record_fit_round``).
    Every concrete hook sets it; it defaults to 0 here for a hook that hasn't
    compressed anything yet.

    ``last_payloads`` is the same round's payloads as an actual list of byte
    strings, one per ``measure_bytes`` call the hook made (one per tensor for
    the quantized/SVD hooks, one per non-empty tensor here). A summed length
    alone cannot be re-scored against a different encoder — compression is
    content-sensitive, not just size-sensitive — so full AC 2 reproducibility
    needs the payloads themselves, with call boundaries preserved (summing a
    new encoder over the list differs from running it once over a
    concatenation). Every concrete hook sets it; kept unconditionally since
    building the list costs nothing beyond what ``last_payload_bytes`` already
    computes — callers decide whether to carry it further (see
    ``fedmaq.baselines.transport.pack_payloads``).
    """

    last_payload_bytes: int = 0
    last_payloads: list[bytes] = []

    #: Secondary, as-published byte total (#26) for a hook whose source paper
    #: mandates its own transport coder -- currently only
    #: ``DAdaQuantCompressionHook``. ``None`` means "not applicable", distinct
    #: from a measured 0; every other hook leaves this at the class default.
    last_secondary_bytes: int | None = None

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Pass ``deltas`` through unchanged; measure their transmitted size.

        Default: identity (uncompressed float32 weights), routed through the
        same held-constant transport every other arm uses (see
        ``fedmaq.baselines.transport.measure_bytes``) rather than raw
        ``nbytes`` — so this "uncompressed control" is charged for the same
        encoder every compressed arm pays for.
        """
        # Deferred: fedmaq.baselines imports this module (CompressionHook), so
        # a top-level import here would cycle.
        from fedmaq.baselines.transport import measure_bytes

        payloads = [d.astype(np.float32).tobytes() for d in deltas if d.size]
        self.last_payload_bytes = sum(len(p) for p in payloads)
        self.last_payloads = payloads
        byte_size = sum(measure_bytes(p) for p in payloads)
        return deltas, byte_size


def get_loss_hook(alg_name: str, alg_cfg: dict[str, Any]) -> LossHook:
    """Factory: return the appropriate LossHook for the given algorithm name."""
    if alg_name == "fedprox":
        return FedProxLossHook(mu=float(alg_cfg.get("mu", 0.01)))
    if alg_name == "fedmaq":
        if alg_cfg.get("client_kd_reg", False):
            from fedmaq.core.kd_loss_hook import ClientKDLossHook

            return ClientKDLossHook(  # type: ignore[return-value]
                alpha=float(alg_cfg.get("kd_reg_alpha", 0.5)),
                temperature=float(alg_cfg.get("kd_reg_temp", 2.0)),
                mu=float(alg_cfg.get("kd_prox_mu", 0.0)),
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
