"""Strategy hook implementing FedDistill+ (FedAvg weights + label-wise logit KD).

Reference: Zhu et al. 2021, "Data-Free Knowledge Distillation for Heterogeneous
Federated Learning" (FedGen codebase, ``serverFedDistill.py``). FEDDISTILL+ keeps
standard FedAvg weight aggregation and additionally averages each client's per-class
mean-logit matrix, broadcasting the consensus matrix back as a distillation target.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import FitIns, Parameters, Scalar
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.client_hooks.feddistill import bytes_to_logits, logits_to_bytes
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


class FedDistillHook(StrategyHook):
    """FedDistill+: FedAvg weight aggregation + per-class logit averaging/broadcast.

    - ``configure_fit``: broadcast the current global logit matrix (once available).
    - ``pre_aggregate_fit``: returns None so FedAvg still averages model weights.
    - ``aggregate_fit``: average the clients' per-class logit matrices for next round.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self.num_classes = int(config.get("dataset", {}).get("num_classes", 10))
        # Consensus per-class logit matrix; None until the first aggregation.
        self.global_logits: np.ndarray | None = None

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        if self.global_logits is not None:
            gl_bytes = logits_to_bytes(self.global_logits)
            for _, fit_ins in client_instructions:
                fit_ins.config["global_logits"] = gl_bytes
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
        matrices: list[np.ndarray] = []
        for _, fit_res in results:
            buf = fit_res.metrics.get("client_logits")
            if isinstance(buf, bytes):
                matrices.append(bytes_to_logits(buf, self.num_classes))
        if matrices:
            self.global_logits = np.mean(matrices, axis=0).astype(np.float32)
            logger.info(
                f"FedDistill+: averaged per-class logits from {len(matrices)} clients."
            )
        return aggregated_parameters, metrics

    def download_size_bytes(
        self,
        strategy: TelemetryFedAvg,
        ndarrays: list[Any],
    ) -> int:
        # FedAvg model weights plus the broadcast global logit matrix (once present).
        base = sum(int(arr.nbytes) for arr in ndarrays)
        if self.global_logits is not None:
            base += int(self.global_logits.astype(np.float32).nbytes)
        return base

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
