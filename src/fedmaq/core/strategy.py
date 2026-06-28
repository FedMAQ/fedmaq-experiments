"""Custom Flower Strategy extending FedAvg with simulated physical time tracking and telemetry."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
import numpy as np
from fedmaq.core.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


class TelemetryFedAvg(FedAvg):
    """Custom FedAvg strategy tracking bandwidth delays, simulated time, and logging to telemetry."""

    def __init__(
        self,
        telemetry_manager: TelemetryManager,
        config: Dict[str, Any],
        client_indices_dict: Optional[Dict[str, List[int]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.telemetry_manager = telemetry_manager
        self.config = config

        # Simulation parameters
        exp_config = config.get("experiment", config)
        self.num_clients = exp_config.get("num_clients", 10)
        self.simulated_time = 0.0
        self.cumulative_bytes = 0

        # Draw bandwidths from uniform U(bw_min, bw_max) in Mbps
        bw_cfg = exp_config.get("heterogeneity", {}).get("bandwidth", {})
        bw_min = float(bw_cfg.get("min_mbps", 5.0))
        bw_max = float(bw_cfg.get("max_mbps", 20.0))

        # Draw compute speeds from uniform U(comp_min, comp_max) in samples/second
        comp_cfg = exp_config.get("heterogeneity", {}).get("compute", {})
        comp_min = float(comp_cfg.get("min_samples_per_sec", 100.0))
        comp_max = float(comp_cfg.get("max_samples_per_sec", 500.0))

        # Seeded generator for reproducibility
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

        self.client_upload_bw = rng.uniform(bw_min, bw_max, size=self.num_clients)
        self.client_download_bw = rng.uniform(bw_min, bw_max, size=self.num_clients)
        self.client_comp_speed = rng.uniform(comp_min, comp_max, size=self.num_clients)

        # DAdaQuant state variables
        self.client_indices_dict = client_indices_dict
        alg_name = config.get("algorithm", {}).get("name", "")
        self.dadaquant_enabled = alg_name == "dadaquant"
        self.q_t = int(config.get("algorithm", {}).get("q_min", 1))
        self.moving_average_history = []
        self.running_average_loss = None
        self.last_quantization_increase_round = 0
        self.psi = float(config.get("algorithm", {}).get("psi", 0.9))
        self.phi = int(config.get("algorithm", {}).get("phi", 5))

        logger.info(
            f"Initialized TelemetryFedAvg for {self.num_clients} clients. "
            f"Bandwidth: U({bw_min}, {bw_max}) Mbps. Compute: U({comp_min}, {comp_max}) samples/s. "
            f"DAdaQuant enabled: {self.dadaquant_enabled}"
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        alg_name = self.config.get("algorithm", {}).get("name", "")
        if alg_name == "fedkd":
            tmin = float(self.config.get("algorithm", {}).get("tmin", 0.1))
            tmax = float(self.config.get("algorithm", {}).get("tmax", 0.9))
            total_rounds = int(
                self.config.get("experiment", {}).get("total_rounds", 10)
            )
            energy = tmin + (server_round / total_rounds) * (tmax - tmin)
            energy = min(max(0.0, energy), 1.0)

            # SVD parameter reconstruction for download path
            ndarrays = parameters_to_ndarrays(parameters)
            from fedmaq.baselines.compression import compress_tensor, decompress_tensor

            reconstructed_ndarrays = []
            for arr in ndarrays:
                if arr.size == 0:
                    reconstructed_ndarrays.append(arr)
                    continue
                orig_shape = arr.shape
                compressed = compress_tensor(arr, energy)
                if len(compressed) == 3:
                    decompressed = decompress_tensor(compressed, orig_shape)
                    reconstructed_ndarrays.append(decompressed.astype(np.float32))
                else:
                    reconstructed_ndarrays.append(arr)
            parameters = ndarrays_to_parameters(reconstructed_ndarrays)

        # Call super().configure_fit to sample clients and get baseline config
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        if not client_instructions:
            return client_instructions

        # Inject server_round into fit configuration for client tracking
        for _, fit_ins in client_instructions:
            fit_ins.config["server_round"] = server_round

        if alg_name == "fedkd":
            for _, fit_ins in client_instructions:
                fit_ins.config["energy"] = energy
            return client_instructions

        if not self.dadaquant_enabled:
            return client_instructions

        # 1. Update time-adaptive quantization level q_t if server_round > 1
        q_min = int(self.config.get("algorithm", {}).get("q_min", 1))
        q_max = int(self.config.get("algorithm", {}).get("q_max", 8))

        if server_round == 1:
            self.q_t = q_min
            self.moving_average_history = []
            self.running_average_loss = None
            self.last_quantization_increase_round = 0
        else:
            # Check convergence condition at round server_round - 1
            history_len = len(self.moving_average_history)
            rounds_since_increase = (
                server_round - 1
            ) - self.last_quantization_increase_round

            if history_len >= self.phi + 1 and rounds_since_increase >= self.phi:
                # Compare latest moving average loss (index -1) with the one phi rounds ago (index -1-phi)
                latest_loss = self.moving_average_history[-1]
                past_loss = self.moving_average_history[-1 - self.phi]
                if latest_loss >= past_loss:
                    old_q = self.q_t
                    self.q_t = min(2 * self.q_t, q_max)
                    if self.q_t > old_q:
                        self.last_quantization_increase_round = server_round - 1
                        logger.info(
                            f"Plateau detected (loss: {latest_loss:.4f} >= {past_loss:.4f}). "
                            f"Doubling quantization level from {old_q} to {self.q_t} for round {server_round}."
                        )

        # Ensure proxy_cid_to_partition_id exists
        if not hasattr(self, "proxy_cid_to_partition_id"):
            self.proxy_cid_to_partition_id = {}

        # 2. Compute client-adaptive quantization levels q_i for each sampled client
        clients = [c for c, _ in client_instructions]
        if self.client_indices_dict is not None:
            sizes = []
            for c in clients:
                cid_str = str(c.cid)
                if cid_str in self.proxy_cid_to_partition_id:
                    pid = self.proxy_cid_to_partition_id[cid_str]
                else:
                    # Try to query properties
                    try:
                        from flwr.common import GetPropertiesIns

                        try:
                            res = c.get_properties(
                                GetPropertiesIns(config={}), timeout=5.0, group_id=0
                            )
                        except TypeError:
                            res = c.get_properties(
                                GetPropertiesIns(config={}), timeout=5.0
                            )
                        pid = int(res.properties["cid"])
                        self.proxy_cid_to_partition_id[cid_str] = pid
                        logger.info(
                            f"Queried partition ID {pid} for Client Proxy {cid_str}"
                        )
                    except Exception as e:
                        # Fallback to hash-based mapping
                        pid = hash(c.cid) % self.num_clients
                        self.proxy_cid_to_partition_id[cid_str] = pid
                        logger.warning(
                            f"Could not query properties for Client Proxy {cid_str} ({e}). "
                            f"Fallback to partition ID {pid}."
                        )

                # Retrieve partition size
                if str(pid) in self.client_indices_dict:
                    sizes.append(len(self.client_indices_dict[str(pid)]))
                elif int(pid) in self.client_indices_dict:
                    sizes.append(len(self.client_indices_dict[int(pid)]))
                else:
                    logger.warning(
                        f"Partition ID {pid} not found in client_indices_dict. Defaulting size to 1."
                    )
                    sizes.append(1)
        else:
            sizes = [1] * len(clients)

        total_size = sum(sizes)
        w = [size / total_size for size in sizes]
        w_pow = [wi ** (2.0 / 3.0) for wi in w]
        w_sq = [wi**2 for wi in w]

        a = sum(w_pow)
        b = sum(ws / (self.q_t**2) for ws in w_sq)

        # Calculate optimal q_i for each client
        updated_instructions = []
        for (client, fit_ins), wi, wi_pow in zip(client_instructions, w, w_pow):
            if b > 0:
                q_val = np.sqrt(a / b) * wi_pow
                q_i = int(max(1, np.round(q_val)))
            else:
                q_i = self.q_t

            # Update the configuration dictionary sent to this client
            fit_ins.config["q"] = q_i
            updated_instructions.append((client, fit_ins))
            logger.info(
                f"Client {client.cid} (weight: {wi:.4f}) assigned quantization level: {q_i} "
                f"(base q_t: {self.q_t})"
            )

        return updated_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        alg_name = self.config.get("algorithm", {}).get("name", "")
        if alg_name == "fedmd":
            if not results:
                return None, {}
            # Extract predictions from client results and perform simple average
            predictions_list = []
            for _, fit_res in results:
                client_preds = parameters_to_ndarrays(fit_res.parameters)[0]
                predictions_list.append(client_preds)
            avg_predictions = np.mean(predictions_list, axis=0)
            aggregated_parameters = ndarrays_to_parameters([avg_predictions])
            metrics = {}
        else:
            # Call FedAvg's aggregation
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round, results, failures
            )

        # Update DAdaQuant running loss average if enabled
        if self.dadaquant_enabled and results:
            if not hasattr(self, "proxy_cid_to_partition_id"):
                self.proxy_cid_to_partition_id = {}
            for client_proxy, fit_res in results:
                cid = int(fit_res.metrics.get("partition_id", -1))
                if cid >= 0:
                    self.proxy_cid_to_partition_id[str(client_proxy.cid)] = cid

            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            if total_examples > 0:
                weighted_loss_sum = 0.0
                for _, fit_res in results:
                    local_loss = float(fit_res.metrics.get("local_loss", 0.0))
                    weight = fit_res.num_examples / total_examples
                    weighted_loss_sum += weight * local_loss

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
                    f"Round {server_round} - DAdaQuant estimated global loss: {weighted_loss_sum:.4f}, "
                    f"moving average: {self.running_average_loss:.4f}, current q_t: {self.q_t}"
                )

        if not results:
            return aggregated_parameters, metrics

        # Compute model size in bytes (based on the aggregated parameters)
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            if alg_name == "fedkd":
                # Compute SVD-compressed download size
                tmin = float(self.config.get("algorithm", {}).get("tmin", 0.1))
                tmax = float(self.config.get("algorithm", {}).get("tmax", 0.9))
                total_rounds = int(
                    self.config.get("experiment", {}).get("total_rounds", 10)
                )
                energy = tmin + (server_round / total_rounds) * (tmax - tmin)
                energy = min(max(0.0, energy), 1.0)

                from fedmaq.baselines.compression import compress_tensor

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

        # Read epochs from fit config (defaults to 5)
        exp_config = self.config.get("experiment", self.config)
        epochs = exp_config.get("local_epochs", 5)

        for client_proxy, fit_res in results:
            # Map client to 0-indexed partition ID using client metrics, with fallback
            cid = int(fit_res.metrics.get("partition_id", -1))
            if cid < 0 or cid >= self.num_clients:
                cid = hash(client_proxy.cid) % self.num_clients

            # Bandwidth (Mbps -> bytes per second)
            upload_speed = (self.client_upload_bw[cid] * 10**6) / 8.0
            download_speed = (self.client_download_bw[cid] * 10**6) / 8.0
            comp_speed = self.client_comp_speed[cid]

            # 1. Download Delay
            t_download = model_size_bytes / download_speed
            round_bytes_downloaded += model_size_bytes

            # 2. Upload Delay (based on compressed/uncompressed sizes returned by client)
            bytes_uploaded = int(
                fit_res.metrics.get("bytes_uploaded", model_size_bytes)
            )
            t_upload = bytes_uploaded / upload_speed
            round_bytes_uploaded += bytes_uploaded

            # 3. Local Training Delay
            num_samples = fit_res.num_examples
            if alg_name == "fedmd":
                public_epochs = int(
                    self.config.get("algorithm", {}).get("public_epochs", 5)
                )
                num_public = int(
                    self.config.get("experiment", {}).get("num_public_samples", 500)
                )
                # For FedMD, the training time includes public dataset distillation
                t_train = (
                    num_public * public_epochs + num_samples * epochs
                ) / comp_speed
            else:
                t_train = (num_samples * epochs) / comp_speed

            # Total round time for client
            client_total_time = t_download + t_train + t_upload
            round_delays.append(client_total_time)

        # For a synchronous server round, the round time is determined by the slowest client
        round_time = max(round_delays) if round_delays else 0.0
        self.simulated_time += round_time

        # Track total communication bytes for this round
        round_total_bytes = round_bytes_downloaded + round_bytes_uploaded
        self.cumulative_bytes += round_total_bytes
        self.last_round_bytes = round_total_bytes

        # Pass round stats in metrics dict
        metrics["round_time"] = round_time
        metrics["round_bytes"] = round_total_bytes

        return aggregated_parameters, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        alg_name = self.config.get("algorithm", {}).get("name", "")
        if alg_name == "fedkd" and server_round > 0:
            tmin = float(self.config.get("algorithm", {}).get("tmin", 0.1))
            tmax = float(self.config.get("algorithm", {}).get("tmax", 0.9))
            total_rounds = int(
                self.config.get("experiment", {}).get("total_rounds", 10)
            )
            energy = tmin + (server_round / total_rounds) * (tmax - tmin)
            energy = min(max(0.0, energy), 1.0)

            # SVD parameter reconstruction for evaluation
            ndarrays = parameters_to_ndarrays(parameters)
            from fedmaq.baselines.compression import compress_tensor, decompress_tensor

            reconstructed_ndarrays = []
            for arr in ndarrays:
                if arr.size == 0:
                    reconstructed_ndarrays.append(arr)
                    continue
                orig_shape = arr.shape
                compressed = compress_tensor(arr, energy)
                if len(compressed) == 3:
                    decompressed = decompress_tensor(compressed, orig_shape)
                    reconstructed_ndarrays.append(decompressed.astype(np.float32))
                else:
                    reconstructed_ndarrays.append(arr)
            parameters = ndarrays_to_parameters(reconstructed_ndarrays)

        # Perform global evaluation via FedAvg
        eval_res = super().evaluate(server_round, parameters)

        if eval_res is not None:
            loss, metrics = eval_res
            acc = float(metrics.get("accuracy", 0.0))

            # Retrieve round metrics tracked in self.aggregate_fit
            round_bytes = (
                getattr(self, "last_round_bytes", 0) if server_round > 0 else 0
            )

            additional_metrics = {}
            if self.dadaquant_enabled:
                additional_metrics["dadaquant/q_t"] = self.q_t
                if self.running_average_loss is not None:
                    additional_metrics["dadaquant/moving_average_loss"] = (
                        self.running_average_loss
                    )
                if hasattr(self, "last_raw_estimated_loss"):
                    additional_metrics["dadaquant/estimated_global_loss"] = (
                        self.last_raw_estimated_loss
                    )

            # Log to console and WandB
            self.telemetry_manager.log(
                round_num=server_round,
                test_loss=loss,
                test_acc=acc,
                round_bytes=round_bytes,
                simulated_time=self.simulated_time,
                additional_metrics=additional_metrics,
            )

        return eval_res
