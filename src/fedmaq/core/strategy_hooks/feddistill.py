"""Strategy hook stub for FedDistill (Task 10 — Sep 2026).

FedDistill (Communication-Efficient Federated Distillation) transfers knowledge
between clients without sharing model parameters.  Clients send logits or
feature vectors; the server aggregates them and broadcasts a consensus signal
back to clients in the next round.

Implementation notes (to be filled in during Task 10)
------------------------------------------------------
- ``pre_aggregate_fit``: aggregate client logits/features (bypass FedAvg).
- ``configure_fit``: broadcast consensus logits / feature targets to clients.
- ``aggregate_fit``: optional server-side refinement step.

Reference
---------
To be confirmed against the FedDistill paper once ported.
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


class FedDistillHook(StrategyHook):
    """Stub hook for FedDistill — not yet implemented (Task 10, Sep 2026)."""

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
        raise NotImplementedError("FedDistill is not yet implemented (Task 10).")

    def aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        aggregated_parameters: Parameters | None,
        metrics: dict[str, Scalar],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        raise NotImplementedError("FedDistill is not yet implemented (Task 10).")

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
