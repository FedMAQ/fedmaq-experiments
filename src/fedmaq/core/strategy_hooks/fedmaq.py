"""Strategy hook implementing FedMAQ's multi-adaptive quantization and server-side KD."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
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

from fedmaq.core.config_defaults import (
    BATCH_SIZE,
    require_num_public_samples,
    resolve_run_context,
    resolve_server_compute_speed,
)
from fedmaq.core.kd_utils import distill_ensemble_into_global, kd_server_sim_time
from fedmaq.core.models import get_server_model_factory
from fedmaq.core.quantization_planner import QuantizationPlanner, QuantPlan, inject_client_q
from fedmaq.core.strategy_hooks._partition import resolve_partition_id
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


class FedMAQHook(StrategyHook):
    """Multi-Adaptive Quantization (configure_fit) + server-side KD (aggregate_fit).

    Quantization state (grad-norm probe model, EMA, per-round plan) lives in
    :class:`~fedmaq.core.quantization_planner.QuantizationPlanner`; this hook only
    orchestrates the call and reports the plan's values as telemetry.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        alg_name = config.get("algorithm", {}).get("name", "fedmaq")
        self._planner = QuantizationPlanner(alg_name, get_server_model_factory(alg_name))
        self._current_plan: QuantPlan = QuantPlan(client_q={}, grad_norms=[])
        self._ema_params: list[np.ndarray] | None = None
        self._last_round_kd_metrics: dict[str, float] = {}

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        ctx = resolve_run_context(self._config)
        client_pids = [resolve_partition_id(c, strategy) for c, _ in client_instructions]
        client_cids = [c.cid for c, _ in client_instructions]

        self._current_plan = self._planner.plan_round(
            parameters,
            client_pids,
            client_cids,
            strategy.client_indices_dict,
            strategy.client_memory,
            ctx,
            ctx.alg_cfg,
        )
        return inject_client_q(client_instructions, self._current_plan.client_q)

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

        ctx = resolve_run_context(self._config)
        alg_cfg = ctx.alg_cfg

        # Select student/teacher architecture factory (single source of truth).
        alg_name = self._config.get("algorithm", {}).get("name", "fedmaq")
        model_fn = get_server_model_factory(alg_name)

        teacher_bit_widths = None
        if alg_cfg.get("soft_voting", False):
            teacher_bit_widths = []
            for client_proxy, _ in results:
                q_val = self._current_plan.client_q.get(
                    client_proxy.cid, int(alg_cfg.get("q_max", 8))
                )
                teacher_bit_widths.append(q_val)

        aggregated_parameters, self._last_round_kd_metrics = distill_ensemble_into_global(
            model_factory=model_fn,
            aggregated_parameters=aggregated_parameters,
            results=results,
            public_indices=strategy.public_indices,
            dataset_name=ctx.dataset_name,
            num_classes=ctx.num_classes,
            batch_size=ctx.batch_size,
            alg_cfg=alg_cfg,
            device=ctx.device,
            teacher_bit_widths=teacher_bit_widths,
        )

        # Apply student EMA if enabled
        if aggregated_parameters is not None and alg_cfg.get("ema_student", False):
            ema_decay = float(alg_cfg.get("ema_decay", 0.99))
            new_params = parameters_to_ndarrays(aggregated_parameters)
            if self._ema_params is None:
                self._ema_params = [p.copy() for p in new_params]
            else:
                self._ema_params = [
                    ema_decay * ema + (1.0 - ema_decay) * new
                    for ema, new in zip(self._ema_params, new_params, strict=True)
                ]
            aggregated_parameters = ndarrays_to_parameters(self._ema_params)

        return aggregated_parameters, metrics

    def get_eval_metrics(self, strategy: TelemetryFedAvg, server_round: int) -> dict[str, Any]:
        metrics = {}
        if self._last_round_kd_metrics:
            for k, v in self._last_round_kd_metrics.items():
                metrics[f"algorithm/fedmaq/{k}"] = v

        # Add grad norm statistics
        grad_norms = self._current_plan.grad_norms
        if grad_norms:
            metrics["algorithm/fedmaq/avg_grad_norm"] = float(np.mean(grad_norms))
            metrics["algorithm/fedmaq/min_grad_norm"] = float(np.min(grad_norms))
            metrics["algorithm/fedmaq/max_grad_norm"] = float(np.max(grad_norms))
            metrics["algorithm/fedmaq/std_grad_norm"] = float(np.std(grad_norms))

        # Add assigned Q statistics
        client_q = self._current_plan.client_q
        if client_q:
            q_vals = list(client_q.values())
            metrics["algorithm/fedmaq/avg_q"] = float(np.mean(q_vals))
            metrics["algorithm/fedmaq/min_q"] = float(np.min(q_vals))
            metrics["algorithm/fedmaq/max_q"] = float(np.max(q_vals))
            metrics["algorithm/fedmaq/std_q"] = float(np.std(q_vals))

        return metrics

    def metric_keys(self) -> list[str]:
        return [
            "algorithm/fedmaq/server_kd_loss",
            "algorithm/fedmaq/avg_grad_norm",
            "algorithm/fedmaq/min_grad_norm",
            "algorithm/fedmaq/max_grad_norm",
            "algorithm/fedmaq/std_grad_norm",
            "algorithm/fedmaq/avg_q",
            "algorithm/fedmaq/min_q",
            "algorithm/fedmaq/max_q",
            "algorithm/fedmaq/std_q",
        ]

    def server_sim_time(
        self,
        strategy: TelemetryFedAvg,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters | None,
    ) -> float:
        if aggregated_parameters is None:
            return 0.0
        alg_cfg = self._config.get("algorithm", {})
        num_public = require_num_public_samples(self._config)
        server_compute_speed = resolve_server_compute_speed(self._config)
        kd_time = kd_server_sim_time(
            num_public=num_public,
            kd_epochs=int(alg_cfg.get("kd_epochs", 1)),
            num_teachers=len(results),
            server_compute_speed=server_compute_speed,
        )
        # F2: account for the per-client grad-norm probe run in configure_fit — one
        # forward+backward on one batch (batch_size samples) per sampled client — in
        # the same sample-pass units as the KD term. Previously unmodeled, so the
        # server-side cost of the adaptive-quantization signal was under-reported.
        if server_compute_speed > 0.0:
            batch_size = int(self._config.get("experiment", {}).get("batch_size", BATCH_SIZE))
            probe_time = (len(results) * batch_size) / server_compute_speed
        else:
            probe_time = 0.0
        return kd_time + probe_time
