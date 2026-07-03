"""Strategy hook stub for CFD (Task 11 — Oct 2026).

CFD (Communication-efficient Federated Distillation) is a variant of
knowledge-distillation-based FL that further reduces uplink communication cost
by selectively transmitting distilled representations rather than full gradients.

Implementation notes (to be filled in during Task 11)
------------------------------------------------------
- ``pre_aggregate_fit``: aggregate distilled representations (bypass FedAvg).
- ``configure_fit``: broadcast distillation targets or compressed global signal.
- ``aggregate_fit``: optional server-side refinement.

Reference
---------
To be confirmed against the CFD paper once ported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flwr.common import FitIns, Parameters, Scalar
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg


class CFDHook(StrategyHook):
    """Stub hook for CFD — not yet implemented (Task 11, Oct 2026)."""

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
        raise NotImplementedError("CFD is not yet implemented (Task 11).")

    def aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        aggregated_parameters: Parameters | None,
        metrics: dict[str, Scalar],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        raise NotImplementedError("CFD is not yet implemented (Task 11).")

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
