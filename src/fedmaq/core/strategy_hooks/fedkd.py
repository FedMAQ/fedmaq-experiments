"""Strategy hook implementing FedKD's SVD-based dynamic compression."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
        # Client-side reference state: the last parameters the clients actually
        # hold, reconstructed from compressed deltas. SVD is applied to the
        # *delta* against this reference (gradient-like, genuinely low-rank),
        # never to the full weight matrices (which are not low-rank).
        self._reference: list[np.ndarray] | None = None
        # Reconstructed parameters cached from pre_configure_fit so pre_evaluate
        # evaluates the exact same compressed state clients received, instead of
        # running a second, independent compression pass.
        self._last_reconstructed: Parameters | None = None

    def download_size_bytes(
        self,
        strategy: TelemetryFedAvg,
        ndarrays: list[Any],
    ) -> int:
        """SVD-compressed download size at the current round's energy level."""
        reference = self._reference or [np.zeros_like(arr) for arr in ndarrays]
        model_size_bytes = 0
        for arr, ref in zip(ndarrays, reference):
            if arr.size == 0:
                continue
            delta = arr - ref
            compressed = compress_tensor(delta, self._current_energy)
            model_size_bytes += svd_compressed_nbytes(compressed, arr.nbytes)
        return model_size_bytes

    def compute_speed_scale(self) -> float:
        return 1.0 / self._compute_penalty

    def _compute_energy(self, server_round: int) -> float:
        energy = self._tmin + (server_round / self._total_rounds) * (
            self._tmax - self._tmin
        )
        return float(min(max(0.0, energy), 1.0))

    def _svd_compress_delta(
        self, parameters: Parameters, energy: float, tag: str = ""
    ) -> Parameters:
        """SVD-compress the delta against ``self._reference`` and advance it.

        Compresses ``parameters - reference`` (a genuinely low-rank,
        gradient-like update), not the raw weight matrices, then folds the
        reconstructed delta back into the reference. This mirrors the upload
        path (:class:`FedKDCompressionHook.compress`), which already
        compresses deltas rather than full weights.
        """
        ndarrays = parameters_to_ndarrays(parameters)
        if self._reference is None:
            self._reference = [np.zeros_like(arr) for arr in ndarrays]

        new_reference: list[np.ndarray] = []
        rank_ratios: list[float] = []
        for arr, ref in zip(ndarrays, self._reference):
            if arr.size == 0:
                new_reference.append(arr)
                continue
            delta = arr - ref
            orig_shape = delta.shape
            compressed = compress_tensor(delta, energy)
            if len(compressed) == 3:
                u, sigma, _v = compressed
                full_rank = min(delta.reshape(orig_shape[0], -1).shape)
                rank_ratios.append(sigma.size / full_rank)
                delta_hat = decompress_tensor(compressed, orig_shape).astype(np.float32)
            else:
                delta_hat = delta
            new_reference.append(ref + delta_hat)
        if rank_ratios:
            mean_ratio = sum(rank_ratios) / len(rank_ratios)
            logger.info(
                "FedKD SVD [%s]: energy=%.3f mean_rank_retained=%.3f (n_layers=%d)",
                tag,
                energy,
                mean_ratio,
                len(rank_ratios),
            )
        self._reference = new_reference
        return ndarrays_to_parameters(new_reference)

    def pre_configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
    ) -> Parameters:
        self._current_energy = self._compute_energy(server_round)
        reconstructed = self._svd_compress_delta(
            parameters, self._current_energy, tag="download"
        )
        self._last_reconstructed = reconstructed
        return reconstructed

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
        # Reuse the exact reconstruction already sent to clients this round
        # (avoids a second, independent compression pass diverging from what
        # clients actually train on).
        if self._last_reconstructed is not None:
            return self._last_reconstructed
        energy = self._compute_energy(server_round)
        return self._svd_compress_delta(parameters, energy, tag="eval")
