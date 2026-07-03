"""Strategy hook implementing DAdaQuant's doubly-adaptive quantization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import FitIns, Parameters, Scalar
from flwr.common.typing import FitRes, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


def compute_dadaquant_client_q(
    sizes: list[int],
    q_t: int,
) -> list[int]:
    """Compute optimal client-adaptive quantization levels for each client in DAdaQuant."""
    if not sizes:
        return []
    total_size = sum(sizes)
    if total_size == 0:
        return [q_t] * len(sizes)

    w = [size / total_size for size in sizes]
    w_pow = [wi ** (2.0 / 3.0) for wi in w]
    w_sq = [wi**2 for wi in w]

    a = sum(w_pow)
    b = sum(ws / (q_t**2) for ws in w_sq)

    q_i_list = []
    for wi, wi_pow in zip(w, w_pow):
        if b > 0:
            q_val = np.sqrt(a / b) * wi_pow
            q_i = int(max(1, np.round(q_val)))
        else:
            q_i = q_t
        q_i_list.append(q_i)
    return q_i_list


def _resolve_partition_id(
    client: ClientProxy,
    strategy: TelemetryFedAvg,
) -> int:
    """Resolve a client's partition ID, caching the result on the strategy."""
    cid_str = str(client.cid)
    if cid_str in strategy.proxy_cid_to_partition_id:
        return strategy.proxy_cid_to_partition_id[cid_str]

    # Flower simulation shortcut: client.cid is typically the stringified partition ID
    if cid_str.isdigit():
        pid = int(cid_str)
        if pid < strategy.num_clients:
            strategy.proxy_cid_to_partition_id[cid_str] = pid
            return pid

    try:
        try:
            res = client.get_properties(
                GetPropertiesIns(config={}), timeout=5.0, group_id=0
            )
        except TypeError:
            res = client.get_properties(GetPropertiesIns(config={}), timeout=5.0)
        pid = int(res.properties["cid"])
        strategy.proxy_cid_to_partition_id[cid_str] = pid
        logger.info(f"Queried partition ID {pid} for Client Proxy {cid_str}")
        return pid
    except Exception as exc:
        pid = hash(client.cid) % strategy.num_clients
        strategy.proxy_cid_to_partition_id[cid_str] = pid
        logger.warning(
            f"Could not resolve partition ID for client {cid_str} ({exc}). "
            f"Falling back to hash-based mapping → partition {pid}. "
            "Verify that GenericClient.get_properties exposes 'cid'."
        )
        return pid


class DAdaQuantHook(StrategyHook):
    """Doubly-adaptive quantization (time-adaptive q_t + client-adaptive q_i).

    State is fully self-contained in this hook; the strategy holds no DAdaQuant
    specific attributes.

    Attributes
    ----------
    q_t : int
        Current time-adaptive global quantization level (doubles when plateau detected).
    moving_average_history : list[float]
        Exponential moving-average loss history for plateau detection.
    running_average_loss : float | None
        Current exponential moving average of global loss.
    last_quantization_increase_round : int
        Round number of the most recent q_t increase.
    last_raw_estimated_loss : float
        Most recently observed weighted client loss (before EMA).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        alg_cfg = config.get("algorithm", {})
        self._config = config
        self.q_t: int = int(alg_cfg.get("q_min", 1))
        self.psi: float = float(alg_cfg.get("psi", 0.9))
        self.phi: int = int(alg_cfg.get("phi", 5))
        self.moving_average_history: list[float] = []
        self.running_average_loss: float | None = None
        self.last_quantization_increase_round: int = 0
        self.last_raw_estimated_loss: float = 0.0

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        q_min = int(self._config.get("algorithm", {}).get("q_min", 1))
        q_max = int(self._config.get("algorithm", {}).get("q_max", 8))

        # 1. Update time-adaptive quantization level q_t
        if server_round == 1:
            self.q_t = q_min
            self.moving_average_history = []
            self.running_average_loss = None
            self.last_quantization_increase_round = 0
        else:
            history_len = len(self.moving_average_history)
            rounds_since_increase = (
                server_round - 1
            ) - self.last_quantization_increase_round

            if history_len >= self.phi + 1 and rounds_since_increase >= self.phi:
                latest_loss = self.moving_average_history[-1]
                past_loss = self.moving_average_history[-1 - self.phi]
                if latest_loss >= past_loss:
                    old_q = self.q_t
                    self.q_t = min(2 * self.q_t, q_max)
                    if self.q_t > old_q:
                        self.last_quantization_increase_round = server_round - 1
                        logger.info(
                            f"Plateau detected (loss: {latest_loss:.4f} >= {past_loss:.4f}). "
                            f"Doubling quantization level from {old_q} to {self.q_t} "
                            f"for round {server_round}."
                        )

        # 2. Compute client-adaptive quantization levels q_i
        clients = [c for c, _ in client_instructions]
        client_indices_dict = strategy.client_indices_dict
        if client_indices_dict is not None:
            sizes = []
            for c in clients:
                pid = _resolve_partition_id(c, strategy)
                if str(pid) in client_indices_dict:
                    sizes.append(len(client_indices_dict[str(pid)]))
                elif int(pid) in client_indices_dict:
                    sizes.append(len(client_indices_dict[int(pid)]))
                else:
                    logger.warning(
                        f"Partition ID {pid} not found in client_indices_dict. "
                        "Defaulting size to 1."
                    )
                    sizes.append(1)
        else:
            sizes = [1] * len(clients)

        q_i_list = compute_dadaquant_client_q(sizes, self.q_t)
        updated_instructions: list[tuple[ClientProxy, FitIns]] = []
        for (client, fit_ins), q_i in zip(client_instructions, q_i_list):
            new_fit_ins = FitIns(fit_ins.parameters, dict(fit_ins.config))
            new_fit_ins.config["q"] = q_i
            updated_instructions.append((client, new_fit_ins))
            logger.info(
                f"Client {client.cid} assigned quantization level: {q_i} "
                f"(base q_t: {self.q_t})"
            )

        return updated_instructions

    def aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        aggregated_parameters: Parameters | None,
        metrics: dict[str, Scalar],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return aggregated_parameters, metrics

        # Cache partition IDs from metrics for future configure_fit calls
        for client_proxy, fit_res in results:
            cid = int(fit_res.metrics.get("partition_id", -1))
            if cid >= 0:
                strategy.proxy_cid_to_partition_id[str(client_proxy.cid)] = cid

        # Update running exponential moving average loss
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples > 0:
            weighted_loss_sum = sum(
                float(fit_res.metrics.get("local_loss", 0.0))
                * (fit_res.num_examples / total_examples)
                for _, fit_res in results
            )
            self.last_raw_estimated_loss = weighted_loss_sum

            if self.running_average_loss is None:
                self.running_average_loss = weighted_loss_sum
            else:
                self.running_average_loss = (
                    self.psi * self.running_average_loss
                    + (1.0 - self.psi) * weighted_loss_sum
                )
            self.moving_average_history.append(self.running_average_loss)
            logger.info(
                f"Round {server_round} - DAdaQuant estimated global loss: "
                f"{weighted_loss_sum:.4f}, moving average: "
                f"{self.running_average_loss:.4f}, current q_t: {self.q_t}"
            )

        return aggregated_parameters, metrics

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {"algorithm/dadaquant/q_t": self.q_t}
        if self.running_average_loss is not None:
            metrics["algorithm/dadaquant/moving_average_loss"] = (
                self.running_average_loss
            )
        if self.last_raw_estimated_loss:
            metrics["algorithm/dadaquant/estimated_global_loss"] = (
                self.last_raw_estimated_loss
            )
        return metrics
