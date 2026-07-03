"""Default no-op hook for algorithms that need no configure_fit / aggregate_fit logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flwr.common import Parameters, Scalar
from flwr.common.typing import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg


class PassthroughHook(StrategyHook):
    """Identity hook — passes all arguments through unchanged.

    Used for FedAvg, FedProx, FedPAQ, and any algorithm that only customises
    the client-side (loss or compressor hook) with no server-side logic.
    """

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
        return aggregated_parameters, metrics

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
