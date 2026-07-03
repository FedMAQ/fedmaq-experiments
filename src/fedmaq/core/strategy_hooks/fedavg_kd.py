"""Strategy hook implementing FedAvgKD (FedAvg + server-side knowledge distillation)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.kd_utils import run_server_side_kd
from fedmaq.core.models import (
    DEVICE,
    get_model,
    get_model_parameters,
    set_model_parameters,
)
from fedmaq.core.partitioning import get_server_loaders
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

        # FedAvgKD uses the standard model for both teacher and student
        student_model = get_model(dataset_name, num_classes)
        student_model.to(device)
        set_model_parameters(
            student_model, parameters_to_ndarrays(aggregated_parameters)
        )

        teachers: list[nn.Module] = []
        for _, fit_res in results:
            try:
                teacher = get_model(dataset_name, num_classes)
                set_model_parameters(
                    teacher, parameters_to_ndarrays(fit_res.parameters)
                )
                teacher.eval()
                teacher.to(device)
                teachers.append(teacher)
            except Exception as exc:
                logger.warning(f"Failed to load client model from parameters: {exc}")

        if teachers and strategy.public_indices is not None:
            try:
                public_loader, _ = get_server_loaders(
                    dataset_name, strategy.public_indices, batch_size=batch_size
                )
                run_server_side_kd(
                    student_model=student_model,
                    teachers=teachers,
                    public_loader=public_loader,
                    temperature=float(alg_cfg.get("temperature", 1.0)),
                    learning_rate=float(alg_cfg.get("server_kd_lr", 0.01)),
                    momentum=float(alg_cfg.get("server_kd_momentum", 0.9)),
                    epochs=int(alg_cfg.get("kd_epochs", 1)),
                    device=device,
                )
                updated_ndarrays = get_model_parameters(student_model)
                aggregated_parameters = ndarrays_to_parameters(updated_ndarrays)
                logger.info(
                    f"Server-side KD: successfully distilled knowledge "
                    f"from {len(teachers)} teacher models."
                )
            except Exception as exc:
                logger.error(f"Error during server-side KD: {exc}")

        return aggregated_parameters, metrics

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
