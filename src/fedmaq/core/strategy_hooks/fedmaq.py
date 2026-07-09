"""Strategy hook implementing FedMAQ's multi-adaptive quantization and server-side KD."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import FitRes, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.kd_utils import run_server_side_kd
from fedmaq.core.models import (
    DEVICE,
    get_kd_student_model,
    get_model_parameters,
    set_model_parameters,
)
from fedmaq.core.partitioning import get_client_loader, get_server_loaders
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)

# Permissible bit-width set per manuscript §4.2: Q = {1,...,8, 16, 32}.
# 16/32-bit tiers are effectively "escape" precision levels for well-resourced
# clients; reachability depends on c_unit and configured memory range (see
# conf/algorithm/fedmaq.yaml).
DEFAULT_BIT_WIDTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32)


def _snap_nearest(value: float, bit_widths: tuple[int, ...]) -> int:
    """Snap ``value`` to the nearest permissible bit-width in ``bit_widths``."""
    return min(bit_widths, key=lambda b: abs(b - value))


def _snap_floor(value: float, bit_widths: tuple[int, ...]) -> int:
    """Snap ``value`` down to the largest permissible bit-width <= ``value``."""
    eligible = [b for b in bit_widths if b <= value]
    return max(eligible) if eligible else min(bit_widths)


def compute_fedmaq_q_k_t(
    c_k: float,
    c_unit: float,
    g_k: float,
    g_max: float,
    n_k: int,
    n_max: int,
    formulation: int,
    q_min: int,
    q_max: int,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    lambda_val: float = 1.0,
    tau_g: float = 0.5,
    tau_n: float = 0.5,
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS,
) -> int:
    """Compute client-specific quantization bit-width for FedMAQ.

    The final result is always a member of ``bit_widths`` (manuscript §4.2's
    permissible set Q), not an arbitrary continuous integer.
    """
    # Normalized signals
    tilde_g = g_k / g_max if g_max > 0.0 else 0.0
    tilde_n = n_k / n_max if n_max > 0.0 else 0.0

    # Tier 1 hard cap: Q_max = floor(c_k / c_unit), snapped down to the nearest
    # permissible bit-width so memory-limited clients never exceed a physically
    # unquantizable precision level.
    q_max_capped = _snap_floor(max(1.0, np.floor(c_k / c_unit)), bit_widths)

    # Tier 2 soft quality target based on the formulation
    q_hat: int | float
    if formulation == 0:
        # Alternative 0: Resource-Only hard cap — no soft quality signal.
        # The soft target is always q_max; only Tier-1 constrains the final value.
        q_hat = q_max
    elif formulation == 1:
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

    # Clamp intermediate result to the configured [q_min, q_max] soft-target range,
    # then snap to the nearest permissible bit-width in the manuscript's set Q.
    q_hat = max(float(q_min), min(float(q_max), float(q_hat)))
    q_hat_snapped = _snap_nearest(q_hat, bit_widths)
    # Tier-1 hard cap: memory-limited clients may receive fewer bits than q_min.
    # This is intentional — the physical bound wins over the soft quality target.
    # Both operands are already members of `bit_widths`, so the min is too.
    return min(q_max_capped, q_hat_snapped)


def _resolve_partition_id(
    client: ClientProxy,
    strategy: TelemetryFedAvg,
) -> int:
    """Resolve a client's partition ID, caching the result on the strategy.

    First checks the cache; if missing, queries ``get_properties`` with a
    ``group_id`` fallback for older Flower versions; falls back to a hash-based
    mapping and logs a WARNING so the failure is visible.
    """
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


class FedMAQHook(StrategyHook):
    """Multi-Adaptive Quantization (configure_fit) + server-side KD (aggregate_fit).

    State
    -----
    ``_grad_norm_model`` is instantiated lazily on the first round and reused
    thereafter, avoiding repeated ResNet18 allocation on every configure_fit call.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._grad_norm_model: nn.Module | None = None

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        alg_cfg = self._config.get("algorithm", {})
        q_min = int(alg_cfg.get("q_min", 2))
        q_max = int(alg_cfg.get("q_max", 8))
        c_unit = float(alg_cfg.get("c_unit", 2048.0))
        formulation = int(alg_cfg.get("formulation", 3))
        gamma1 = float(alg_cfg.get("gamma1", 0.5))
        gamma2 = float(alg_cfg.get("gamma2", 0.5))
        lambda_val = float(alg_cfg.get("lambda_val", 1.0))
        tau_g = float(alg_cfg.get("tau_g", 0.5))
        tau_n = float(alg_cfg.get("tau_n", 0.5))
        bit_widths = tuple(int(b) for b in alg_cfg.get("bit_widths", DEFAULT_BIT_WIDTHS))

        dataset_name = self._config.get("dataset", {}).get("name", "mnist")
        num_classes = int(self._config.get("dataset", {}).get("num_classes", 10))
        batch_size = int(self._config.get("experiment", {}).get("batch_size", 64))
        device = torch.device(self._config.get("device") or DEVICE)

        # Lazily instantiate and cache the gradient norm model.
        # FedMAQ's global/client model IS the KD student (TinyCNN for 1-channel,
        # SimpleCNN for CIFAR), so the grad-norm probe must use that same
        # architecture; using the full get_model() here loads mismatched
        # parameters (e.g. ResNet18GN on CIFAR) and silently zeroes every norm.
        if self._grad_norm_model is None:
            self._grad_norm_model = get_kd_student_model(dataset_name, num_classes)
            self._grad_norm_model.to(device)
        temp_model = self._grad_norm_model
        ndarrays = parameters_to_ndarrays(parameters)
        set_model_parameters(temp_model, ndarrays)
        temp_model.eval()
        criterion = nn.CrossEntropyLoss()

        # Map client proxies to partition IDs
        client_pids = [
            _resolve_partition_id(c, strategy) for c, _ in client_instructions
        ]

        # 1. Compute raw gradient norms for sampled clients
        grad_norms: list[float] = []
        dataset_sizes: list[int] = []
        client_indices_dict = strategy.client_indices_dict
        for pid in client_pids:
            # Retrieve dataset size
            if client_indices_dict is not None:
                key_str, key_int = str(pid), int(pid)
                if key_str in client_indices_dict:
                    n_k = len(client_indices_dict[key_str])
                elif key_int in client_indices_dict:
                    n_k = len(client_indices_dict[key_int])
                else:
                    n_k = 1
            else:
                n_k = 1
            dataset_sizes.append(n_k)

            # Get client loader and compute stochastic gradient norm
            loader = get_client_loader(
                dataset_name=dataset_name,
                client_id=pid,
                client_indices_dict=client_indices_dict,
                batch_size=batch_size,
                train=True,
            )
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
            except Exception as exc:
                logger.warning(
                    f"Error computing gradient norm for client partition {pid}: {exc}. "
                    "Defaulting to 1e-8."
                )
                norm = 1e-8

            grad_norms.append(max(1e-8, norm))

        # 2. Normalize signals
        g_max = max(grad_norms) if grad_norms else 1e-8
        n_max = max(dataset_sizes) if dataset_sizes else 1

        # 3. Compute and inject client-specific quantization bit-widths
        updated_instructions: list[tuple[ClientProxy, FitIns]] = []
        for (client, fit_ins), pid, g_k, n_k in zip(
            client_instructions, client_pids, grad_norms, dataset_sizes, strict=True
        ):
            c_k = float(strategy.client_memory[pid])
            q_k_t = compute_fedmaq_q_k_t(
                c_k=c_k,
                c_unit=c_unit,
                g_k=g_k,
                g_max=g_max,
                n_k=n_k,
                n_max=n_max,
                formulation=formulation,
                q_min=q_min,
                q_max=q_max,
                gamma1=gamma1,
                gamma2=gamma2,
                lambda_val=lambda_val,
                tau_g=tau_g,
                tau_n=tau_n,
                bit_widths=bit_widths,
            )
            # Instantiate new FitIns to prevent shared reference overwrites
            new_fit_ins = FitIns(fit_ins.parameters, dict(fit_ins.config))
            new_fit_ins.config["q"] = q_k_t
            updated_instructions.append((client, new_fit_ins))
            logger.info(
                f"FedMAQ - Client {client.cid} (partition {pid}): "
                f"c_k={c_k:.1f}MB, g_k={g_k:.4f} (tilde_g={g_k/g_max:.4f}), "
                f"n_k={n_k} (tilde_n={n_k/n_max:.4f}) -> "
                f"Final assigned q: {q_k_t}"
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
        if aggregated_parameters is None:
            return aggregated_parameters, metrics

        dataset_name = self._config.get("dataset", {}).get("name", "mnist")
        num_classes = int(self._config.get("dataset", {}).get("num_classes", 10))
        batch_size = int(self._config.get("experiment", {}).get("batch_size", 64))
        device = torch.device(self._config.get("device") or DEVICE)
        alg_cfg = self._config.get("algorithm", {})

        # Student model architecture for FedMAQ uses the KD student (TinyCNN/SimpleCNN)
        student_model = get_kd_student_model(dataset_name, num_classes)
        student_model.to(device)
        set_model_parameters(
            student_model, parameters_to_ndarrays(aggregated_parameters)
        )

        # Load teacher models from client parameter snapshots
        teachers: list[nn.Module] = []
        for _, fit_res in results:
            try:
                teacher = get_kd_student_model(dataset_name, num_classes)
                set_model_parameters(
                    teacher, parameters_to_ndarrays(fit_res.parameters)
                )
                teacher.eval()
                teacher.to(device)
                teachers.append(teacher)
            except Exception as exc:
                logger.warning(f"Failed to load client model from parameters: {exc}")

        if teachers and strategy.public_indices is not None:
            try:
                public_loader, _ = get_server_loaders(
                    dataset_name, strategy.public_indices, batch_size=batch_size
                )
                run_server_side_kd(
                    student_model=student_model,
                    teachers=teachers,
                    public_loader=public_loader,
                    temperature=float(alg_cfg.get("temperature", 1.0)),
                    learning_rate=float(alg_cfg.get("server_kd_lr", 0.01)),
                    momentum=float(alg_cfg.get("server_kd_momentum", 0.9)),
                    epochs=int(alg_cfg.get("kd_epochs", 1)),
                    device=device,
                )
                updated_ndarrays = get_model_parameters(student_model)
                aggregated_parameters = ndarrays_to_parameters(updated_ndarrays)
                logger.info(
                    f"Server-side KD: successfully distilled knowledge "
                    f"from {len(teachers)} teacher models."
                )
            except Exception as exc:
                logger.error(f"Error during server-side KD: {exc}")

        return aggregated_parameters, metrics
