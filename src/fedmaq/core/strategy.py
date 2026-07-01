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
        public_indices: Optional[List[int]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.telemetry_manager = telemetry_manager
        self.config = config
        self.public_indices = public_indices

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
        self.client_comp_speed = np.full(self.num_clients, comp_max)
        self.client_memory = rng.uniform(2048.0, 16384.0, size=self.num_clients)

        # DAdaQuant state variables
        self.client_indices_dict = client_indices_dict
        alg_name = config.get("algorithm", {}).get("name", "")
        self.dadaquant_enabled = alg_name == "dadaquant"
        self.fedmaq_enabled = alg_name == "fedmaq"
        self.q_t = int(config.get("algorithm", {}).get("q_min", 1))
        self.moving_average_history = []
        self.running_average_loss = None
        self.last_quantization_increase_round = 0
        self.psi = float(config.get("algorithm", {}).get("psi", 0.9))
        self.phi = int(config.get("algorithm", {}).get("phi", 5))

        logger.info(
            f"Initialized TelemetryFedAvg for {self.num_clients} clients. "
            f"Bandwidth: U({bw_min}, {bw_max}) Mbps. Compute: U({comp_min}, {comp_max}) samples/s. "
            f"Memory: U(2048, 16384) MB. "
            f"DAdaQuant enabled: {self.dadaquant_enabled}, FedMAQ enabled: {self.fedmaq_enabled}"
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

        if self.fedmaq_enabled:
            # Multi-Adaptive Quantization Logic for FedMAQ
            import torch
            from fedmaq.core.models import get_model, set_model_parameters
            from fedmaq.core.partitioning import get_client_loader

            # Get configuration settings
            alg_cfg = self.config.get("algorithm", {})
            q_min = int(alg_cfg.get("q_min", 2))
            q_max = int(alg_cfg.get("q_max", 8))
            c_unit = float(alg_cfg.get("c_unit", 2048.0))
            formulation = int(alg_cfg.get("formulation", 3))

            gamma1 = float(alg_cfg.get("gamma1", 0.5))
            gamma2 = float(alg_cfg.get("gamma2", 0.5))
            lambda_val = float(alg_cfg.get("lambda_val", 1.0))
            tau_g = float(alg_cfg.get("tau_g", 0.5))
            tau_n = float(alg_cfg.get("tau_n", 0.5))

            dataset_name = self.config.get("dataset", {}).get("name", "mnist")
            num_classes = int(self.config.get("dataset", {}).get("num_classes", 10))
            batch_size = int(self.config.get("experiment", {}).get("batch_size", 64))

            # Instantiate temporary model to compute initial gradient norm
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            temp_model = get_model(dataset_name, num_classes)
            temp_model.to(device)
            ndarrays = parameters_to_ndarrays(parameters)
            set_model_parameters(temp_model, ndarrays)
            temp_model.eval()
            criterion = torch.nn.CrossEntropyLoss()

            # Ensure proxy_cid_to_partition_id exists
            if not hasattr(self, "proxy_cid_to_partition_id"):
                self.proxy_cid_to_partition_id = {}

            # Map client proxies to partition IDs
            client_pids = []
            for client, _ in client_instructions:
                cid_str = str(client.cid)
                if cid_str in self.proxy_cid_to_partition_id:
                    pid = self.proxy_cid_to_partition_id[cid_str]
                else:
                    try:
                        from flwr.common import GetPropertiesIns

                        try:
                            res = client.get_properties(
                                GetPropertiesIns(config={}), timeout=5.0, group_id=0
                            )
                        except TypeError:
                            res = client.get_properties(
                                GetPropertiesIns(config={}), timeout=5.0
                            )
                        pid = int(res.properties["cid"])
                        self.proxy_cid_to_partition_id[cid_str] = pid
                    except Exception:
                        pid = hash(client.cid) % self.num_clients
                        self.proxy_cid_to_partition_id[cid_str] = pid
                client_pids.append(pid)

            # 1. Compute raw gradient norms for sampled clients
            grad_norms = []
            dataset_sizes = []
            for pid in client_pids:
                # Retrieve dataset size
                if self.client_indices_dict is not None:
                    if str(pid) in self.client_indices_dict:
                        n_k = len(self.client_indices_dict[str(pid)])
                    elif int(pid) in self.client_indices_dict:
                        n_k = len(self.client_indices_dict[int(pid)])
                    else:
                        n_k = 1
                else:
                    n_k = 1
                dataset_sizes.append(n_k)

                # Get client loader
                loader = get_client_loader(
                    dataset_name=dataset_name,
                    client_id=pid,
                    client_indices_dict=self.client_indices_dict,
                    batch_size=batch_size,
                    train=True,
                )

                # Fetch first batch of data to compute stochastic gradient norm
                try:
                    images, labels = next(iter(loader))
                    images, labels = images.to(device), labels.to(device)
                    temp_model.zero_grad()
                    outputs = temp_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    norm = torch.sqrt(
                        sum(
                            p.grad.detach().pow(2).sum()
                            for p in temp_model.parameters()
                            if p.grad is not None
                        )
                    ).item()
                except Exception as e:
                    logger.warning(
                        f"Error computing gradient norm for client partition {pid}: {e}. Defaulting to 1e-8."
                    )
                    norm = 1e-8

                grad_norms.append(max(1e-8, norm))

            # 2. Normalize signals
            g_max = max(grad_norms) if grad_norms else 1e-8
            n_max = max(dataset_sizes) if dataset_sizes else 1

            # 3. Compute client-specific quantization bit-widths
            updated_instructions = []
            for (client, fit_ins), pid, g_k, n_k in zip(
                client_instructions, client_pids, grad_norms, dataset_sizes
            ):
                # Normalized signals
                tilde_g = g_k / g_max
                tilde_n = n_k / n_max

                # Tier 1 hard cap: Q_max = floor(c_k / c_unit)
                c_k = float(self.client_memory[pid])
                q_max_capped = int(max(1, np.floor(c_k / c_unit)))

                # Tier 2 soft quality target based on the formulation
                if formulation == 1:
                    # Alternative 1: Linear Sum
                    term = gamma1 * tilde_g + gamma2 * tilde_n
                    q_hat = q_min + np.round((q_max - q_min) * term)
                elif formulation == 2:
                    # Alternative 2: Multiplicative
                    term = (tilde_g**gamma1) * (tilde_n**gamma2)
                    q_hat = q_min + np.round((q_max - q_min) * term)
                elif formulation == 3:
                    # Alternative 3: Gradient-Primary, Data-Modulated
                    modulator = (1.0 + lambda_val * tilde_n) / (1.0 + lambda_val)
                    q_hat = q_min + np.round((q_max - q_min) * tilde_g * modulator)
                elif formulation == 4:
                    # Alternative 4: Threshold-Based Staged Rule
                    q_mid = int(np.round((q_max + q_min) / 2.0))
                    if tilde_g >= tau_g and tilde_n >= tau_n:
                        q_hat = q_max
                    elif tilde_g >= tau_g or tilde_n >= tau_n:
                        q_hat = q_mid
                    else:
                        q_hat = q_min
                else:
                    q_hat = q_min

                # Apply constraints
                q_hat = int(max(q_min, min(q_max, q_hat)))
                q_k_t = int(min(q_max_capped, q_hat))

                # Inject assigned q (instantiate new FitIns to prevent shared reference overwrites)
                from flwr.common import FitIns

                new_fit_ins = FitIns(fit_ins.parameters, dict(fit_ins.config))
                new_fit_ins.config["q"] = q_k_t
                updated_instructions.append((client, new_fit_ins))
                logger.info(
                    f"FedMAQ - Client {client.cid} (partition {pid}): "
                    f"c_k={c_k:.1f}MB, g_k={g_k:.4f} (tilde_g={tilde_g:.4f}), "
                    f"n_k={n_k} (tilde_n={tilde_n:.4f}) -> "
                    f"Q_max={q_max_capped}, q_hat={q_hat} -> Final assigned q: {q_k_t}"
                )

            return updated_instructions

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

        if self.fedmaq_enabled and aggregated_parameters is not None:
            # Server-side knowledge distillation logic for FedMAQ
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from fedmaq.core.models import (
                get_model,
                set_model_parameters,
                get_model_parameters,
            )
            from fedmaq.core.partitioning import get_server_loaders
            from pathlib import Path

            dataset_name = self.config.get("dataset", {}).get("name", "mnist")
            num_classes = int(self.config.get("dataset", {}).get("num_classes", 10))
            batch_size = int(self.config.get("experiment", {}).get("batch_size", 64))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if alg_name in ["fedkd", "fedmaq"]:
                if "cifar" in dataset_name.lower():
                    from fedmaq.core.models import SimpleCNN

                    student_model = SimpleCNN(in_channels=3, num_classes=num_classes)
                else:
                    from fedmaq.core.models import TinyCNN

                    student_model = TinyCNN(in_channels=1, num_classes=num_classes)
            else:
                student_model = get_model(dataset_name, num_classes)
            student_model.to(device)
            set_model_parameters(
                student_model, parameters_to_ndarrays(aggregated_parameters)
            )

            # Load teacher models
            teachers = []
            if alg_name == "fedmaq":
                for client_proxy, fit_res in results:
                    try:
                        teacher = get_model(dataset_name, num_classes)
                        client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                        set_model_parameters(teacher, client_ndarrays)
                        teacher.eval()
                        teacher.to(device)
                        teachers.append(teacher)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load client model directly from parameters: {e}"
                        )
            else:
                persistence_dir = self.config.get("experiment", {}).get(
                    "persistence_dir", ".data_partitions/fedmaq_models"
                )
                for client_proxy, fit_res in results:
                    cid = int(fit_res.metrics.get("partition_id", -1))
                    if cid < 0:
                        cid = hash(client_proxy.cid) % self.num_clients

                    teacher_path = Path(persistence_dir) / f"client_{cid}.pth"
                    if teacher_path.exists():
                        try:
                            teacher = get_model(dataset_name, num_classes)
                            teacher.load_state_dict(
                                torch.load(teacher_path, map_location=device)
                            )
                            teacher.eval()
                            teacher.to(device)
                            teachers.append(teacher)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load teacher model from {teacher_path}: {e}"
                            )

            if teachers and self.public_indices is not None:
                try:
                    public_loader, _ = get_server_loaders(
                        dataset_name, self.public_indices, batch_size=batch_size
                    )
                    temperature = float(
                        self.config.get("algorithm", {}).get("temperature", 1.0)
                    )

                    optimizer = torch.optim.SGD(
                        student_model.parameters(), lr=0.01, momentum=0.9
                    )
                    kl_criterion = nn.KLDivLoss(reduction="batchmean")

                    student_model.train()
                    # Run 1 epoch of distillation over public dataset
                    for images, _ in public_loader:
                        images = images.to(device)

                        # Get soft targets from teachers
                        with torch.no_grad():
                            teacher_soft_preds_list = []
                            for teacher in teachers:
                                t_out = teacher(images)
                                teacher_soft_preds_list.append(
                                    F.softmax(t_out / temperature, dim=1)
                                )
                            teacher_soft_preds = torch.stack(
                                teacher_soft_preds_list
                            ).mean(dim=0)

                        optimizer.zero_grad()
                        student_logits = student_model(images)
                        student_log_soft = F.log_softmax(
                            student_logits / temperature, dim=1
                        )

                        # Distillation loss
                        loss = kl_criterion(student_log_soft, teacher_soft_preds) * (
                            temperature**2
                        )
                        loss.backward()
                        optimizer.step()

                    # Extract updated weights
                    updated_ndarrays = get_model_parameters(student_model)
                    aggregated_parameters = ndarrays_to_parameters(updated_ndarrays)
                    logger.info(
                        f"Server-side KD: successfully distilled knowledge from {len(teachers)} teacher models."
                    )
                except Exception as e:
                    logger.error(f"Error during server-side KD: {e}")

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
        self.last_round_time = round_time

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
            round_time = (
                getattr(self, "last_round_time", 0.0) if server_round > 0 else 0.0
            )

            # Build unified metrics dict
            log_metrics = {
                "round": server_round,
                "test/loss": loss,
                "test/accuracy": acc,
                "communication/round_bytes": round_bytes,
                "system/round_time_sec": round_time,
            }

            # Merge other metrics returned by evaluate_fn (e.g. precision, recall, f1)
            for k, v in metrics.items():
                if k != "accuracy":
                    log_metrics[f"test/{k}"] = float(v)

            # Inject algorithm-specific states
            if self.dadaquant_enabled:
                log_metrics["algorithm/dadaquant/q_t"] = self.q_t
                if self.running_average_loss is not None:
                    log_metrics["algorithm/dadaquant/moving_average_loss"] = (
                        self.running_average_loss
                    )
                if hasattr(self, "last_raw_estimated_loss"):
                    log_metrics["algorithm/dadaquant/estimated_global_loss"] = (
                        self.last_raw_estimated_loss
                    )

            # Log to console, local log files, and WandB
            self.telemetry_manager.log(
                round_num=server_round,
                metrics=log_metrics,
            )

        return eval_res
