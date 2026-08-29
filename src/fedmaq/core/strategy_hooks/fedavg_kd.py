"""Strategy hook implementing FedAvgKD (FedAvg + server-side knowledge distillation)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
)
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.config_defaults import (
    require_num_public_samples,
    resolve_algorithm_config,
    resolve_run_context,
)
from fedmaq.core.kd_utils import (
    apply_student_ema,
    distill_ensemble_into_global,
    kd_server_sim_time,
)
from fedmaq.core.models import get_model
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
        self._run_context = resolve_run_context(config)
        self.dataset_name = self._run_context.dataset_name
        self.num_classes = self._run_context.num_classes
        self.batch_size = self._run_context.batch_size
        self.device = self._run_context.device
        self.alg_cfg = resolve_algorithm_config(config)
        self._ema_params: list[Any] | None = None
        self._last_round_kd_metrics: dict[str, float] = {}

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

        # FedAvgKD uses the standard model for both teacher and student.
        aggregated_parameters, self._last_round_kd_metrics = distill_ensemble_into_global(
            model_factory=get_model,
            aggregated_parameters=aggregated_parameters,
            results=results,
            public_indices=strategy.public_indices,
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            alg_cfg=self.alg_cfg,
            device=self.device,
        )

        # Student EMA is quantization-independent, so §4.3.7's refinement parity
        # requires this arm to carry it too (soft_voting and grad_norm_ema are
        # recorded as inapplicable here: both act on a quantization signal).
        if aggregated_parameters is not None:
            aggregated_parameters, self._ema_params = apply_student_ema(
                aggregated_parameters, self._ema_params, self.alg_cfg
            )
        return aggregated_parameters, metrics

    def server_sim_time(
        self,
        strategy: TelemetryFedAvg,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters | None,
    ) -> float:
        if aggregated_parameters is None:
            return 0.0
        num_public = self._run_context.num_public_samples
        if num_public is None:
            num_public = require_num_public_samples(self._config)
        return kd_server_sim_time(
            num_public=num_public,
            kd_epochs=int(self.alg_cfg.get("kd_epochs", 1)),
            num_teachers=len(results),
            server_compute_speed=self._run_context.server_compute_speed,
        )

    def get_eval_metrics(self, strategy: TelemetryFedAvg, server_round: int) -> dict[str, Any]:
        metrics = {}
        if self._last_round_kd_metrics:
            for k, v in self._last_round_kd_metrics.items():
                metrics[f"algorithm/fedavg_kd/{k}"] = v
        return metrics

    def metric_keys(self) -> list[str]:
        return ["algorithm/fedavg_kd/server_kd_loss"]
