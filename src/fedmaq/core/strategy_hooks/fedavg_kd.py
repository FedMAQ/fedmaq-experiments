"""Strategy hook implementing FedAvgKD (FedAvg + server-side knowledge distillation)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
)
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.kd_utils import distill_ensemble_into_global
from fedmaq.core.models import DEVICE, get_model
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


class FedAvgKDHook(StrategyHook):
    """Server-side Knowledge Distillation hook for FedAvgKD.

    FedAvgKD uses standard FedAvg weight aggregation on the client side, then
    refines the global model via ensemble KD on the server's public dataset.
    Both teacher and student use the standard model architecture (unlike FedMAQ
    which uses TinyCNN/SimpleCNN as its student).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        return client_instructions

    def aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        aggregated_parameters: Parameters | None,
        metrics: dict[str, Scalar],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if aggregated_parameters is None:
            return aggregated_parameters, metrics

        dataset_name = self._config.get("dataset", {}).get("name", "mnist")
        num_classes = int(self._config.get("dataset", {}).get("num_classes", 10))
        batch_size = int(self._config.get("experiment", {}).get("batch_size", 64))
        device = torch.device(self._config.get("device") or DEVICE)
        alg_cfg = self._config.get("algorithm", {})

        # FedAvgKD uses the standard model for both teacher and student.
        aggregated_parameters = distill_ensemble_into_global(
            model_factory=get_model,
            aggregated_parameters=aggregated_parameters,
            results=results,
            public_indices=strategy.public_indices,
            dataset_name=dataset_name,
            num_classes=num_classes,
            batch_size=batch_size,
            alg_cfg=alg_cfg,
            device=device,
        )
        return aggregated_parameters, metrics

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
