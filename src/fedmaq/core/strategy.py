"""Custom Flower Strategy extending FedAvg with simulated physical time tracking and telemetry."""

import logging
from typing import Any

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedmaq.core.quantization_planner import (
    compute_fedmaq_q_k_t,  # noqa: F401 — re-exported for backward compatibility
)
from fedmaq.core.strategy_hooks import StrategyHook, get_strategy_hook
from fedmaq.core.strategy_hooks.dadaquant import (
    compute_dadaquant_client_q,  # noqa: F401 — re-exported for backward compatibility
)
from fedmaq.core.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


class NetworkSimulator:
    """Simulates communication delays and training delays for FL rounds."""

    def __init__(
        self,
        client_upload_bw: np.ndarray,
        client_download_bw: np.ndarray,
        client_comp_speed: np.ndarray,
        num_clients: int,
    ) -> None:
        self.client_upload_bw = client_upload_bw
        self.client_download_bw = client_download_bw
        self.client_comp_speed = client_comp_speed
        self.num_clients = num_clients

    def simulate_client_delay(
        self,
        cid: int,
        model_size_bytes: int,
        bytes_uploaded: int,
        train_sample_count: float,
        compute_scale: float = 1.0,
    ) -> tuple[float, float, float]:
        """Return (t_download, t_train, t_upload) for a single client.

        ``train_sample_count`` is the effective number of sample-epochs processed
        during local training and ``compute_scale`` a multiplicative factor on the
        client's compute speed. Both are supplied by the algorithm's strategy hook
        so this method carries no per-algorithm branching.
        """
        # Speed in bytes per second (from Mbps)
        upload_speed = (self.client_upload_bw[cid] * 10**6) / 8.0
        download_speed = (self.client_download_bw[cid] * 10**6) / 8.0
        comp_speed = self.client_comp_speed[cid] * compute_scale

        t_download = model_size_bytes / download_speed
        t_upload = bytes_uploaded / upload_speed
        t_train = train_sample_count / comp_speed

        return t_download, t_train, t_upload


class TelemetryFedAvg(FedAvg):
    """Custom FedAvg strategy tracking bandwidth delays and simulated physical time.

    Logs all results and telemetry to console, local logs, and Weight & Biases.

    Algorithm-specific logic (quantization assignment, server-side KD, logit
    aggregation, etc.) is fully delegated to a :class:`~fedmaq.core.strategy_hooks.StrategyHook`
    instance.  To add a new baseline, implement a hook and register it in
    ``core/strategy_hooks/__init__.py``.
    """

    def __init__(
        self,
        telemetry_manager: TelemetryManager,
        config: dict[str, Any],
        client_indices_dict: dict[str, list[int]] | None = None,
        public_indices: list[int] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.telemetry_manager = telemetry_manager
        self.config = config
        self.public_indices = public_indices
        self.client_indices_dict = client_indices_dict

        # Shared partition-ID cache used by multiple hooks
        self.proxy_cid_to_partition_id: dict[str, int] = {}

        # Simulation parameters
        exp_config = config.get("experiment", config)
        self.num_clients = exp_config.get("num_clients", 10)
        self.alg_name: str = config.get("algorithm", {}).get("name", "")

        # Seeded generator for reproducibility
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

        # Bandwidth and Compute are uniform
        if "bandwidth_mbps" in exp_config:
            bandwidth_mbps = float(exp_config["bandwidth_mbps"])
        else:
            bw_cfg = exp_config.get("heterogeneity", {}).get("bandwidth", {})
            bandwidth_mbps = float(bw_cfg.get("min_mbps", 10.0))

        if "compute_samples_per_sec" in exp_config:
            compute_samples_per_sec = float(exp_config["compute_samples_per_sec"])
        else:
            comp_cfg = exp_config.get("heterogeneity", {}).get("compute", {})
            compute_samples_per_sec = float(comp_cfg.get("min_samples_per_sec", 200.0))

        self.client_upload_bw = np.full(self.num_clients, bandwidth_mbps)
        self.client_download_bw = np.full(self.num_clients, bandwidth_mbps)
        self.client_comp_speed = np.full(self.num_clients, compute_samples_per_sec)

        # Memory: uniform fixed value (control group) or heterogeneous U(2048, 16384) MB (§4.1)
        uniform_mb = None
        for cfg_source in [exp_config, config]:
            if isinstance(cfg_source, dict):
                h_cfg = cfg_source.get("heterogeneity", {})
                if isinstance(h_cfg, dict):
                    uniform_mb = h_cfg.get("uniform_memory_mb", None)
                    if uniform_mb is not None:
                        break

        if uniform_mb is not None:
            self.client_memory = np.full(self.num_clients, float(uniform_mb))
            memory_desc = f"fixed {uniform_mb} MB (control group)"
        else:
            self.client_memory = rng.uniform(2048.0, 16384.0, size=self.num_clients)
            memory_desc = "U(2048, 16384) MB"

        self.network_simulator = NetworkSimulator(
            client_upload_bw=self.client_upload_bw,
            client_download_bw=self.client_download_bw,
            client_comp_speed=self.client_comp_speed,
            num_clients=self.num_clients,
        )

        # Instantiate the per-algorithm strategy hook
        self.hook: StrategyHook = get_strategy_hook(self.alg_name, config)

        # Declare this hook's metric keys so the CSV header stays stable even
        # when a key only appears starting round 1 (e.g. FedMAQ grad-norm stats).
        self.telemetry_manager.register_hook_metric_keys(self.hook.metric_keys())

        logger.info(
            f"Initialized TelemetryFedAvg for {self.num_clients} clients. "
            f"Algorithm: {self.alg_name}. "
            f"Bandwidth (uniform): {bandwidth_mbps} Mbps. "
            f"Compute (uniform): {compute_samples_per_sec} samples/s. "
            f"Memory: {memory_desc}. "
            f"Hook: {type(self.hook).__name__}"
        )

    @property
    def simulated_time(self) -> float:
        """Cumulative simulated round time (client + server) across all rounds.

        Delegates to ``TelemetryManager``, which owns the accumulation (see
        ``telemetry.py``); kept as a passthrough since external callers read
        ``strategy.simulated_time`` directly.
        """
        return self.telemetry_manager.cumulative_time

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        # 1. Let the hook optionally compress parameters for the download path
        parameters = self.hook.pre_configure_fit(self, server_round, parameters)

        # 1b. Seed the round's deterministic client draw (SeededPartitionClientManager).
        # Done here, before super().configure_fit() calls client_manager.sample(),
        # so the selection is reproducible and robust to sample() call count.
        if hasattr(client_manager, "set_round_seed"):
            client_manager.set_round_seed(server_round)

        # 2. Call FedAvg client sampling
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        if not client_instructions:
            return client_instructions

        # 3. Inject server_round into every client's fit config
        for _, fit_ins in client_instructions:
            fit_ins.config["server_round"] = server_round

        # 4. Delegate algorithm-specific instruction modification to the hook
        return self.hook.configure_fit(
            self, server_round, parameters, client_manager, client_instructions
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        # 1. Check if the hook wants to bypass FedAvg aggregation entirely
        pre_result = self.hook.pre_aggregate_fit(self, server_round, results, failures)
        if pre_result is not None:
            aggregated_parameters, metrics = pre_result
        else:
            # Standard FedAvg weighted aggregation
            aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # 2. Hook post-processing (server KD, loss tracking, etc.)
        aggregated_parameters, metrics = self.hook.aggregate_fit(
            self, server_round, results, failures, aggregated_parameters, metrics
        )

        # 3. Telemetry: client-metric aggregation + simulated delay/byte accounting,
        # relocated behind TelemetryManager (see telemetry.py:record_fit_round).
        round_time, round_total_bytes = self.telemetry_manager.record_fit_round(
            self, server_round, results, aggregated_parameters
        )

        if not results:
            return aggregated_parameters, metrics

        metrics["round_time"] = round_time
        metrics["round_bytes"] = round_total_bytes

        return aggregated_parameters, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        # Let the hook decompress parameters if needed (e.g. FedKD SVD)
        parameters = self.hook.pre_evaluate(self, server_round, parameters)

        eval_res = super().evaluate(server_round, parameters)

        if eval_res is not None:
            loss, metrics = eval_res
            acc = float(metrics.get("accuracy", 0.0))

            tm = self.telemetry_manager
            round_bytes = tm.last_round_bytes if server_round > 0 else 0
            round_time = tm.last_round_time if server_round > 0 else 0.0
            client_time = tm.last_client_time if server_round > 0 else 0.0
            server_time = tm.last_server_time if server_round > 0 else 0.0
            wall_time = tm.last_wall_time if server_round > 0 else 0.0

            log_metrics: dict[str, Any] = {
                "round": server_round,
                "test/loss": loss,
                "test/accuracy": acc,
                "communication/round_bytes": round_bytes,
                "system/round_time_sec": round_time,
                "system/client_sim_time_sec": client_time,
                "system/server_sim_time_sec": server_time,
                "system/wall_time_sec": wall_time,
            }

            # Per-client communication breakdown (min/mean/max/std) — shows the
            # adaptive-quantization mechanism (DAdaQuant/FedMAQ) at work, not just
            # the aggregate total.
            if server_round > 0:
                log_metrics.update(tm.last_client_bytes_stats)

            # Merge other metrics returned by evaluate_fn (e.g. precision, recall, f1)
            for k, v in metrics.items():
                if k != "accuracy":
                    log_metrics[f"test/{k}"] = float(v)

            # Merge client-side aggregated metrics
            if tm.last_round_client_metrics:
                log_metrics.update(tm.last_round_client_metrics)

            # Merge algorithm-specific hook metrics (e.g. DAdaQuant q_t)
            log_metrics.update(self.hook.get_eval_metrics(self, server_round))

            self.telemetry_manager.log(
                round_num=server_round,
                metrics=log_metrics,
            )

        return eval_res
