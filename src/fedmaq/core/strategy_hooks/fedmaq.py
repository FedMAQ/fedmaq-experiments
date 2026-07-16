"""Strategy hook implementing FedMAQ's multi-adaptive quantization and server-side KD."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.config_defaults import (
    BATCH_SIZE,
    DATASET_NAME,
    NUM_CLASSES,
    SERVER_COMPUTE_SPEED,
)
from fedmaq.core.kd_utils import distill_ensemble_into_global, kd_server_sim_time
from fedmaq.core.models import (
    DEVICE,
    get_server_model_factory,
    set_model_parameters,
)
from fedmaq.core.partitioning import get_client_loader
from fedmaq.core.strategy_hooks._partition import (
    partition_dataset_size,
    resolve_partition_id,
)
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)

# Permissible bit-width set per manuscript §4.2: Q = {1,...,8, 16, 32}.
# 16/32-bit tiers are effectively "escape" precision levels for well-resourced
# clients; reachability depends on c_unit and configured memory range (see
# conf/algorithm/fedmaq.yaml).
DEFAULT_BIT_WIDTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32)


@dataclass(frozen=True)
class _QuantParams:
    """Parsed FedMAQ quantization hyperparameters for one configure_fit call.

    Grouping these keeps the q-assignment helper's signature small and, via
    ``from_cfg``, preserves the F8 fail-loud contract: the algorithm-defining
    knobs (``q_min``/``q_max``/``c_unit``/``formulation``) are read with no
    default so a missing/renamed key raises up front, before any probe work.
    """

    q_min: int
    q_max: int
    c_unit: float
    formulation: int
    gamma1: float
    gamma2: float
    lambda_val: float
    tau_g: float
    tau_n: float
    bit_widths: tuple[int, ...]

    @classmethod
    def from_cfg(cls, alg_cfg: dict[str, Any]) -> _QuantParams:
        return cls(
            q_min=int(alg_cfg["q_min"]),
            q_max=int(alg_cfg["q_max"]),
            c_unit=float(alg_cfg["c_unit"]),
            formulation=int(alg_cfg["formulation"]),
            gamma1=float(alg_cfg.get("gamma1", 0.5)),
            gamma2=float(alg_cfg.get("gamma2", 0.5)),
            lambda_val=float(alg_cfg.get("lambda_val", 1.0)),
            tau_g=float(alg_cfg.get("tau_g", 0.5)),
            tau_n=float(alg_cfg.get("tau_n", 0.5)),
            bit_widths=tuple(int(b) for b in alg_cfg.get("bit_widths", DEFAULT_BIT_WIDTHS)),
        )


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

    # Tier 1 hard cap: Q_max = floor(c_k / c_unit), kept raw (unsnapped) so it can
    # be combined with the raw Tier-2 target below before a single floor-into-Q.
    q_k_max_raw = max(1.0, np.floor(c_k / c_unit))

    # Tier 2 soft quality target based on the formulation
    q_hat: float
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

    # Clamp intermediate result to the configured [q_min, q_max] soft-target range.
    q_hat = max(float(q_min), min(float(q_max), float(q_hat)))
    # Combine raw Tier-1 cap and raw Tier-2 target via min(), then floor into the
    # permissible set Q exactly once: q_k^(t) = max{q in Q | q <= min(Q_k^max, q_hat_k^(t))}.
    # Memory-limited clients may receive fewer bits than q_min — intentional, the
    # physical bound wins over the soft quality target.
    return _snap_floor(min(q_k_max_raw, q_hat), bit_widths)


class FedMAQHook(StrategyHook):
    """Multi-Adaptive Quantization (configure_fit) + server-side KD (aggregate_fit).

    State
    -----
    ``_grad_norm_model`` is instantiated lazily on the first round and reused
    thereafter, avoiding repeated model allocation on every configure_fit call.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._grad_norm_model: nn.Module | None = None
        self._round_client_q: dict[str, int] = {}
        self._ema_params: list[np.ndarray] | None = None
        self._grad_norm_ema: dict[int, float] = {}
        # Reported-state fields, refreshed each round (F5: init here, not lazily).
        self._last_grad_norms: list[float] = []
        self._last_assigned_q: dict[str, int] = {}
        self._last_round_kd_metrics: dict[str, float] = {}

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        # F7: clear per-round q map so it never retains stale entries for clients
        # not sampled this round, nor grows unbounded across the run.
        self._round_client_q.clear()
        alg_cfg = self._config.get("algorithm", {})
        # F8: read the algorithm-defining knobs fail-loud up front (from_cfg has no
        # default for them) so a missing/renamed key raises before any probe work.
        qp = _QuantParams.from_cfg(alg_cfg)

        dataset_name = self._config.get("dataset", {}).get("name", DATASET_NAME)
        num_classes = int(self._config.get("dataset", {}).get("num_classes", NUM_CLASSES))
        batch_size = int(self._config.get("experiment", {}).get("batch_size", BATCH_SIZE))
        device = torch.device(self._config.get("device") or DEVICE)

        temp_model = self._ensure_grad_norm_model(parameters, dataset_name, num_classes, device)

        # Map client proxies to partition IDs
        client_pids = [
            resolve_partition_id(c, strategy) for c, _ in client_instructions
        ]

        grad_norms, dataset_sizes = self._probe_grad_norms(
            temp_model, client_pids, dataset_name, strategy.client_indices_dict,
            batch_size, device,
        )
        grad_norms = self._smooth_grad_norms(client_pids, grad_norms, alg_cfg)
        self._last_grad_norms = grad_norms

        return self._assign_quantization(
            strategy, client_instructions, client_pids, grad_norms, dataset_sizes, qp
        )

    def _ensure_grad_norm_model(
        self,
        parameters: Parameters,
        dataset_name: str,
        num_classes: int,
        device: torch.device,
    ) -> nn.Module:
        """Lazily build + cache the grad-norm probe model, then load ``parameters``.

        The factory (standard model for FedMAQ, KD student for FedMAQ-Lite) is
        resolved by get_server_model_factory — the single source of truth.
        """
        alg_name = self._config.get("algorithm", {}).get("name", "fedmaq")
        model_fn = get_server_model_factory(alg_name)
        if self._grad_norm_model is None:
            self._grad_norm_model = model_fn(dataset_name, num_classes)
            self._grad_norm_model.to(device)
        temp_model = self._grad_norm_model
        set_model_parameters(temp_model, parameters_to_ndarrays(parameters))
        temp_model.eval()
        return temp_model

    def _probe_grad_norms(
        self,
        temp_model: nn.Module,
        client_pids: list[int],
        dataset_name: str,
        client_indices_dict: dict[int, list[int]],
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[float], list[int]]:
        """One stochastic single-batch gradient-norm probe per sampled client.

        Returns the per-client raw grad norms (floored at 1e-8) and dataset sizes,
        aligned with ``client_pids``.
        """
        criterion = nn.CrossEntropyLoss()
        grad_norms: list[float] = []
        dataset_sizes: list[int] = []
        for pid in client_pids:
            n_k = partition_dataset_size(client_indices_dict, pid)
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
            except ValueError:
                # F6: a shape/count mismatch (raised by set_model_parameters) is a
                # config/architecture bug, not a transient batch fault — fail loud.
                raise
            except Exception as exc:
                logger.warning(
                    f"Error computing gradient norm for client partition {pid}: {exc}. "
                    "Defaulting to 1e-8."
                )
                norm = 1e-8

            grad_norms.append(max(1e-8, norm))

        return grad_norms, dataset_sizes

    def _smooth_grad_norms(
        self,
        client_pids: list[int],
        grad_norms: list[float],
        alg_cfg: dict[str, Any],
    ) -> list[float]:
        """EMA-smooth per-client grad norms across rounds (Priority 3), if enabled.

        Mutates ``self._grad_norm_ema``. When ``grad_norm_ema`` is off, returns the
        raw norms unchanged.
        """
        if not alg_cfg.get("grad_norm_ema", False):
            return grad_norms
        beta = float(alg_cfg.get("grad_norm_beta", 0.7))
        smoothed_norms = []
        for pid, raw_norm in zip(client_pids, grad_norms, strict=True):
            if pid in self._grad_norm_ema:
                smoothed = beta * self._grad_norm_ema[pid] + (1.0 - beta) * raw_norm
            else:
                smoothed = raw_norm
            self._grad_norm_ema[pid] = smoothed
            smoothed_norms.append(smoothed)
        return smoothed_norms

    def _assign_quantization(
        self,
        strategy: TelemetryFedAvg,
        client_instructions: list[tuple[ClientProxy, FitIns]],
        client_pids: list[int],
        grad_norms: list[float],
        dataset_sizes: list[int],
        qp: _QuantParams,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Normalize the signals and inject each client's bit-width ``q`` into FitIns."""
        g_max = max(grad_norms) if grad_norms else 1e-8
        n_max = max(dataset_sizes) if dataset_sizes else 1

        self._last_assigned_q = {}
        updated_instructions: list[tuple[ClientProxy, FitIns]] = []
        for (client, fit_ins), pid, g_k, n_k in zip(
            client_instructions, client_pids, grad_norms, dataset_sizes, strict=True
        ):
            c_k = float(strategy.client_memory[pid])
            q_k_t = compute_fedmaq_q_k_t(
                c_k=c_k,
                c_unit=qp.c_unit,
                g_k=g_k,
                g_max=g_max,
                n_k=n_k,
                n_max=n_max,
                formulation=qp.formulation,
                q_min=qp.q_min,
                q_max=qp.q_max,
                gamma1=qp.gamma1,
                gamma2=qp.gamma2,
                lambda_val=qp.lambda_val,
                tau_g=qp.tau_g,
                tau_n=qp.tau_n,
                bit_widths=qp.bit_widths,
            )
            # Instantiate new FitIns to prevent shared reference overwrites
            new_fit_ins = FitIns(fit_ins.parameters, dict(fit_ins.config))
            new_fit_ins.config["q"] = q_k_t
            self._round_client_q[client.cid] = q_k_t
            self._last_assigned_q[client.cid] = q_k_t
            updated_instructions.append((client, new_fit_ins))
            logger.info(
                f"FedMAQ - Client {client.cid} (partition {pid}): "
                f"c_k={c_k:.1f}MB, g_k={g_k:.4f} (tilde_g={g_k / g_max:.4f}), "
                f"n_k={n_k} (tilde_n={n_k / n_max:.4f}) -> "
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

        dataset_name = self._config.get("dataset", {}).get("name", DATASET_NAME)
        num_classes = int(self._config.get("dataset", {}).get("num_classes", NUM_CLASSES))
        batch_size = int(self._config.get("experiment", {}).get("batch_size", BATCH_SIZE))
        device = torch.device(self._config.get("device") or DEVICE)
        alg_cfg = self._config.get("algorithm", {})

        # Select student/teacher architecture factory (single source of truth).
        alg_name = self._config.get("algorithm", {}).get("name", "fedmaq")
        model_fn = get_server_model_factory(alg_name)

        teacher_bit_widths = None
        if alg_cfg.get("soft_voting", False):
            teacher_bit_widths = []
            for client_proxy, _ in results:
                q_val = self._round_client_q.get(
                    client_proxy.cid, int(alg_cfg.get("q_max", 8))
                )
                teacher_bit_widths.append(q_val)

        aggregated_parameters, self._last_round_kd_metrics = (
            distill_ensemble_into_global(
                model_factory=model_fn,
                aggregated_parameters=aggregated_parameters,
                results=results,
                public_indices=strategy.public_indices,
                dataset_name=dataset_name,
                num_classes=num_classes,
                batch_size=batch_size,
                alg_cfg=alg_cfg,
                device=device,
                teacher_bit_widths=teacher_bit_widths,
            )
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

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        metrics = {}
        if self._last_round_kd_metrics:
            for k, v in self._last_round_kd_metrics.items():
                metrics[f"algorithm/fedmaq/{k}"] = v

        # Add grad norm statistics
        if self._last_grad_norms:
            metrics["algorithm/fedmaq/avg_grad_norm"] = float(
                np.mean(self._last_grad_norms)
            )
            metrics["algorithm/fedmaq/min_grad_norm"] = float(
                np.min(self._last_grad_norms)
            )
            metrics["algorithm/fedmaq/max_grad_norm"] = float(
                np.max(self._last_grad_norms)
            )
            metrics["algorithm/fedmaq/std_grad_norm"] = float(
                np.std(self._last_grad_norms)
            )

        # Add assigned Q statistics
        if self._last_assigned_q:
            q_vals = list(self._last_assigned_q.values())
            metrics["algorithm/fedmaq/avg_q"] = float(np.mean(q_vals))
            metrics["algorithm/fedmaq/min_q"] = float(np.min(q_vals))
            metrics["algorithm/fedmaq/max_q"] = float(np.max(q_vals))
            metrics["algorithm/fedmaq/std_q"] = float(np.std(q_vals))

        return metrics

    def server_sim_time(
        self,
        strategy: TelemetryFedAvg,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters | None,
    ) -> float:
        if aggregated_parameters is None:
            return 0.0
        alg_cfg = self._config.get("algorithm", {})
        num_public = int(
            self._config.get("experiment", {}).get("num_public_samples", 200)
        )
        server_compute_speed = float(alg_cfg.get("server_compute_speed", SERVER_COMPUTE_SPEED))
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
