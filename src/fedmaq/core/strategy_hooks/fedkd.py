"""Strategy hook implementing FedKD's SVD-based dynamic compression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.baselines.compression import (
    compress_tensor,
    decompress_tensor,
    svd_compressed_nbytes,
)
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg


class FedKDHook(StrategyHook):
    """SVD download-path compression and energy injection for FedKD.

    - ``pre_configure_fit``: compresses server parameters via SVD before sending.
    - ``configure_fit``: injects the round energy scalar into each client's config.
    - ``pre_evaluate``: decompresses SVD parameters before global evaluation.
    - ``aggregate_fit``: passes through (FedAvg weight aggregation is used as-is).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        alg_cfg = config.get("algorithm", {})
        self._tmin = float(alg_cfg.get("tmin", 0.1))
        self._tmax = float(alg_cfg.get("tmax", 0.9))
        self._total_rounds = int(config.get("experiment", {}).get("total_rounds", 10))
        # Dual-model (student+teacher) training slows effective client compute.
        self._compute_penalty = float(alg_cfg.get("compute_penalty", 2.5))
        # Cached energy for current round (set in pre_configure_fit, read in configure_fit)
        self._current_energy: float = self._tmin

    def download_size_bytes(
        self,
        strategy: TelemetryFedAvg,
        ndarrays: list[Any],
    ) -> int:
        """SVD-compressed download size at the current round's energy level."""
        model_size_bytes = 0
        for arr in ndarrays:
            if arr.size == 0:
                continue
            compressed = compress_tensor(arr, self._current_energy)
            model_size_bytes += svd_compressed_nbytes(compressed, arr.nbytes)
        return model_size_bytes

    def compute_speed_scale(self) -> float:
        return 1.0 / self._compute_penalty

    def _compute_energy(self, server_round: int) -> float:
        energy = self._tmin + (server_round / self._total_rounds) * (
            self._tmax - self._tmin
        )
        return float(min(max(0.0, energy), 1.0))

    def _svd_compress_parameters(
        self, parameters: Parameters, energy: float
    ) -> Parameters:
        """Apply SVD compression to model parameters for the download path."""
        ndarrays = parameters_to_ndarrays(parameters)
        reconstructed: list[np.ndarray] = []
        for arr in ndarrays:
            if arr.size == 0:
                reconstructed.append(arr)
                continue
            orig_shape = arr.shape
            compressed = compress_tensor(arr, energy)
            if len(compressed) == 3:
                decompressed = decompress_tensor(compressed, orig_shape)
                reconstructed.append(decompressed.astype(np.float32))
            else:
                reconstructed.append(arr)
        return ndarrays_to_parameters(reconstructed)

    def pre_configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
    ) -> Parameters:
        self._current_energy = self._compute_energy(server_round)
        return self._svd_compress_parameters(parameters, self._current_energy)

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        for _, fit_ins in client_instructions:
            fit_ins.config["energy"] = self._current_energy
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
        # FedKD uses standard FedAvg weight aggregation; no server-side post-processing.
        return aggregated_parameters, metrics

    def pre_evaluate(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
    ) -> Parameters:
        if server_round <= 0:
            return parameters
        energy = self._compute_energy(server_round)
        return self._svd_compress_parameters(parameters, energy)
