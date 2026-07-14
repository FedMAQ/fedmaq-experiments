"""Strategy hook implementing FedMD's logit-based aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import FitIns, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg


class FedMDHook(StrategyHook):
    """Logit-aggregation hook for FedMD.

    FedMD does not aggregate model weights; instead clients exchange predictions
    (logits) on a shared public dataset.

    - ``pre_aggregate_fit``: averages client logit predictions; bypasses FedAvg.
    - ``configure_fit``/``aggregate_fit``: pass-through (no additional logic needed).
    - ``local_train_sample_count``: folds in round-1 public/private pre-training.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def local_train_sample_count(
        self,
        num_samples: int,
        epochs: int,
        num_public: int,
        public_epochs: int,
        server_round: int,
    ) -> float:
        # Digest (public) + revisit (private) every round.
        base = num_public * public_epochs + num_samples * epochs
        if server_round != 1:
            return base
        # Round 1 additionally runs mandatory public/private pre-training.
        alg_cfg = self._config.get("algorithm", {})
        pub_pretrain = int(alg_cfg.get("public_pretrain_epochs", 10))
        priv_pretrain = int(alg_cfg.get("private_pretrain_epochs", 10))
        return num_public * pub_pretrain + num_samples * priv_pretrain + base

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        return client_instructions

    def pre_aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]] | None:
        if not results:
            return None, {}
        # Extract predictions from client results and perform simple average
        predictions_list = [parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results]
        avg_predictions = np.mean(predictions_list, axis=0)
        return ndarrays_to_parameters([avg_predictions]), {}

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

    def get_eval_metrics(self, strategy: TelemetryFedAvg, server_round: int) -> dict[str, Any]:
        return {}
