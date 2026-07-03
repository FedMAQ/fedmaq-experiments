"""Custom Flower Strategy extending FedAvg with simulated physical time tracking and telemetry."""

import logging
from typing import Any

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedmaq.baselines.compression import compress_tensor
from fedmaq.core.strategy_hooks import StrategyHook, get_strategy_hook
from fedmaq.core.strategy_hooks.dadaquant import (
    DAdaQuantHook,
    compute_dadaquant_client_q,  # noqa: F401 — re-exported for backward compatibility
)
from fedmaq.core.strategy_hooks.fedmaq import (
    compute_fedmaq_q_k_t,  # noqa: F401 — re-exported for backward compatibility
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
        num_samples: int,
        epochs: int,
        alg_name: str,
        public_epochs: int = 5,
        num_public: int = 200,
        server_round: int = 1,
    ) -> tuple[float, float, float]:
        """Return (t_download, t_train, t_upload) for a single client."""
        # Speed in bytes per second (from Mbps)
        upload_speed = (self.client_upload_bw[cid] * 10**6) / 8.0
        download_speed = (self.client_download_bw[cid] * 10**6) / 8.0
        comp_speed = self.client_comp_speed[cid]

        # Apply computational penalty scale factor for FedKD due to dual model training
        if alg_name == "fedkd":
            comp_speed = comp_speed / 2.5

        # 1. Download Delay
        t_download = model_size_bytes / download_speed

        # 2. Upload Delay
        t_upload = bytes_uploaded / upload_speed

        # 3. Local Training Delay
        if alg_name == "fedmd":
            # In round 1, include mandatory public and private pre-training (10 epochs each)
            if server_round == 1:
                t_train = (
                    num_public * 10
                    + num_samples * 10
                    + num_public * public_epochs
                    + num_samples * epochs
                ) / comp_speed
            else:
                t_train = (
                    num_public * public_epochs + num_samples * epochs
                ) / comp_speed
        else:
            t_train = (num_samples * epochs) / comp_speed

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
        self.simulated_time = 0.0
        self.cumulative_bytes = 0
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

        logger.info(
            f"Initialized TelemetryFedAvg for {self.num_clients} clients. "
            f"Algorithm: {self.alg_name}. "
            f"Bandwidth (uniform): {bandwidth_mbps} Mbps. "
            f"Compute (uniform): {compute_samples_per_sec} samples/s. "
            f"Memory: {memory_desc}. "
            f"Hook: {type(self.hook).__name__}"
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        # 1. Let the hook optionally compress parameters for the download path
        parameters = self.hook.pre_configure_fit(self, server_round, parameters)

        # 2. Call FedAvg client sampling
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
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
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round, results, failures
            )

        # 2. Hook post-processing (server KD, loss tracking, etc.)
        aggregated_parameters, metrics = self.hook.aggregate_fit(
            self, server_round, results, failures, aggregated_parameters, metrics
        )

        if not results:
            return aggregated_parameters, metrics

        # 3. Telemetry: compute model download size in bytes
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            if self.alg_name == "fedkd":
                # Compute SVD-compressed download size using the same energy as configure_fit
                from fedmaq.core.strategy_hooks.fedkd import FedKDHook

                energy = (
                    self.hook._current_energy
                    if isinstance(self.hook, FedKDHook)
                    else 1.0
                )
                model_size_bytes = 0
                for arr in ndarrays:
                    if arr.size == 0:
                        continue
                    compressed = compress_tensor(arr, energy)
                    if len(compressed) == 3:
                        u, sigma, v = compressed
                        model_size_bytes += (u.size + sigma.size + v.size) * 4
                    else:
                        model_size_bytes += arr.nbytes
            else:
                model_size_bytes = sum(arr.nbytes for arr in ndarrays)
        else:
            model_size_bytes = 0

        round_delays = []
        round_bytes_uploaded = 0
        round_bytes_downloaded = 0

        exp_config = self.config.get("experiment", self.config)
        epochs = exp_config.get("local_epochs", 5)

        for client_proxy, fit_res in results:
            cid = int(fit_res.metrics.get("partition_id", -1))
            if cid < 0 or cid >= self.num_clients:
                cid = hash(client_proxy.cid) % self.num_clients

            bytes_uploaded = int(
                fit_res.metrics.get("bytes_uploaded", model_size_bytes)
            )
            num_samples = fit_res.num_examples
            public_epochs = int(
                self.config.get("algorithm", {}).get("public_epochs", 5)
            )
            num_public = int(
                self.config.get("experiment", {}).get("num_public_samples", 200)
            )

            t_download, t_train, t_upload = (
                self.network_simulator.simulate_client_delay(
                    cid=cid,
                    model_size_bytes=model_size_bytes,
                    bytes_uploaded=bytes_uploaded,
                    num_samples=num_samples,
                    epochs=epochs,
                    alg_name=self.alg_name,
                    public_epochs=public_epochs,
                    num_public=num_public,
                    server_round=server_round,
                )
            )

            client_total_time = t_download + t_train + t_upload
            round_delays.append(client_total_time)
            round_bytes_downloaded += model_size_bytes
            round_bytes_uploaded += bytes_uploaded

        # Decouple client and server simulated delays
        client_sim_time = max(round_delays) if round_delays else 0.0

        # Server compute time: non-zero for FedMAQ / FedAvgKD server-side distillation
        server_sim_time = 0.0
        if (
            self.alg_name in {"fedmaq", "fedavg_kd"}
            and aggregated_parameters is not None
        ):
            num_teachers = len(results)
            num_public = int(
                self.config.get("experiment", {}).get("num_public_samples", 200)
            )
            kd_epochs = int(self.config.get("algorithm", {}).get("kd_epochs", 1))
            server_comp_speed = (
                2000.0  # Simulated high-performance server speed (samples/sec)
            )
            server_sim_time = (
                num_public * kd_epochs * num_teachers
            ) / server_comp_speed

        round_time = client_sim_time + server_sim_time
        self.simulated_time += round_time

        round_total_bytes = round_bytes_downloaded + round_bytes_uploaded
        self.cumulative_bytes += round_total_bytes
        self.last_round_bytes = round_total_bytes
        self.last_round_time = round_time
        self.last_client_time = client_sim_time
        self.last_server_time = server_sim_time

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

            round_bytes = (
                getattr(self, "last_round_bytes", 0) if server_round > 0 else 0
            )
            round_time = (
                getattr(self, "last_round_time", 0.0) if server_round > 0 else 0.0
            )
            client_time = (
                getattr(self, "last_client_time", 0.0) if server_round > 0 else 0.0
            )
            server_time = (
                getattr(self, "last_server_time", 0.0) if server_round > 0 else 0.0
            )

            log_metrics: dict[str, Any] = {
                "round": server_round,
                "test/loss": loss,
                "test/accuracy": acc,
                "communication/round_bytes": round_bytes,
                "system/round_time_sec": round_time,
                "system/client_sim_time_sec": client_time,
                "system/server_sim_time_sec": server_time,
            }

            # Merge other metrics returned by evaluate_fn (e.g. precision, recall, f1)
            for k, v in metrics.items():
                if k != "accuracy":
                    log_metrics[f"test/{k}"] = float(v)

            # Merge algorithm-specific hook metrics (e.g. DAdaQuant q_t)
            log_metrics.update(self.hook.get_eval_metrics(self, server_round))

            self.telemetry_manager.log(
                round_num=server_round,
                metrics=log_metrics,
            )

        return eval_res

    # ------------------------------------------------------------------ #
    # Backward-compatible property proxies for DAdaQuant hook state.      #
    # Tests (and any external code) that access strategy.q_t,             #
    # strategy.moving_average_history, etc. continue to work unchanged.   #
    # ------------------------------------------------------------------ #

    @property
    def q_t(self) -> int:
        """Current DAdaQuant time-adaptive quantization level (proxy to hook)."""
        if isinstance(self.hook, DAdaQuantHook):
            return self.hook.q_t
        return 0

    @q_t.setter
    def q_t(self, value: int) -> None:
        if isinstance(self.hook, DAdaQuantHook):
            self.hook.q_t = value

    @property
    def moving_average_history(self) -> list[float]:
        """DAdaQuant EMA loss history (proxy to hook)."""
        if isinstance(self.hook, DAdaQuantHook):
            return self.hook.moving_average_history
        return []

    @moving_average_history.setter
    def moving_average_history(self, value: list[float]) -> None:
        if isinstance(self.hook, DAdaQuantHook):
            self.hook.moving_average_history = value

    @property
    def last_quantization_increase_round(self) -> int:
        """Round number of the most recent DAdaQuant q_t increase (proxy to hook)."""
        if isinstance(self.hook, DAdaQuantHook):
            return self.hook.last_quantization_increase_round
        return 0

    @last_quantization_increase_round.setter
    def last_quantization_increase_round(self, value: int) -> None:
        if isinstance(self.hook, DAdaQuantHook):
            self.hook.last_quantization_increase_round = value
