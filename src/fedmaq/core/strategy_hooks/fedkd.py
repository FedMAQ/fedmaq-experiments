"""Strategy hook implementing FedKD's SVD-based dynamic compression."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.baselines.compression import (
    compress_tensor,
    decompress_tensor,
    svd_payload,
)
from fedmaq.baselines.transport import UploadReport
from fedmaq.core.config_defaults import (
    resolve_algorithm_config,
    resolve_experiment_config,
    resolve_run_context,
)
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


class FedKDHook(StrategyHook):
    """SVD download-path compression and energy injection for FedKD.

    - ``pre_configure_fit``: compresses server parameters via SVD before sending.
    - ``configure_fit``: injects the round energy scalar into each client's config.
    - ``pre_evaluate``: preserves the post-aggregation parameters for evaluation.
    - ``aggregate_fit``: passes through (FedAvg weight aggregation is used as-is).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._run_context = resolve_run_context(config)
        alg_cfg = resolve_algorithm_config(config)
        self._tmin = float(alg_cfg.get("tmin", 0.1))
        self._tmax = float(alg_cfg.get("tmax", 0.9))
        self._min_rank_frac = float(alg_cfg.get("min_rank_frac", 0.0))
        self._total_rounds = int(resolve_experiment_config(config).get("total_rounds", 10))
        # Dual-model (student+teacher) training slows effective client compute.
        self._compute_penalty = float(alg_cfg.get("compute_penalty", 1.3))
        # Cached energy for current round (set in pre_configure_fit, read in configure_fit)
        self._current_energy: float = self._tmin
        # Client-side reference state: the last parameters the clients actually
        # hold, reconstructed from compressed deltas. SVD is applied to the
        # *delta* against this reference (gradient-like, genuinely low-rank),
        # never to the full weight matrices (which are not low-rank).
        self._reference: list[np.ndarray] | None = None
        self._last_download_report: UploadReport | None = None
        self._last_mean_rank_retained: float | None = None

    def download_size_bytes(
        self,
        strategy: TelemetryFedAvg,
        ndarrays: list[Any],
    ) -> UploadReport:
        """Return the SVD-compressed download report at the current energy."""
        # ``pre_configure_fit`` already serialized the payload sent to clients.
        # Reusing that report is essential: by the time telemetry records the
        # round, ``_reference`` has advanced to the broadcast state and a fresh
        # calculation would describe a different delta.
        if self._last_download_report is not None:
            return self._last_download_report
        reference = self._reference or [np.zeros_like(arr) for arr in ndarrays]
        payloads: list[bytes] = []
        for arr, ref in zip(ndarrays, reference, strict=True):
            if arr.size == 0:
                continue
            delta = arr - ref
            compressed = compress_tensor(delta, self._current_energy, self._min_rank_frac)
            payload = svd_payload(compressed)
            payloads.append(payload)
        self._last_download_report = UploadReport.from_payloads(payloads)
        return self._last_download_report

    def compute_speed_scale(self) -> float:
        return 1.0 / self._compute_penalty

    def _compute_energy(self, server_round: int) -> float:
        energy = self._tmin + (server_round / self._total_rounds) * (self._tmax - self._tmin)
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
        payloads: list[bytes] = []
        rank_ratios: list[float] = []
        for arr, ref in zip(ndarrays, self._reference, strict=True):
            if arr.size == 0:
                new_reference.append(arr)
                continue
            delta = arr - ref
            orig_shape = delta.shape
            compressed = compress_tensor(delta, energy, self._min_rank_frac)
            if len(compressed) == 3:
                u, sigma, _v = compressed
                full_rank = min(delta.reshape(orig_shape[0], -1).shape)
                rank_ratios.append(sigma.size / full_rank)
                delta_hat = decompress_tensor(compressed, orig_shape).astype(np.float32)
            else:
                delta_hat = delta
            new_reference.append(ref + delta_hat)
            if tag == "download":
                payloads.append(svd_payload(compressed))
        if rank_ratios:
            mean_ratio = sum(rank_ratios) / len(rank_ratios)
            if tag == "download":
                self._last_mean_rank_retained = mean_ratio
            logger.info(
                "FedKD SVD [%s]: energy=%.3f mean_rank_retained=%.3f (n_layers=%d)",
                tag,
                energy,
                mean_ratio,
                len(rank_ratios),
            )
        self._reference = new_reference
        if tag == "download":
            self._last_download_report = UploadReport.from_payloads(payloads)
        return ndarrays_to_parameters(new_reference)

    def pre_configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
    ) -> Parameters:
        self._current_energy = self._compute_energy(server_round)
        return self._svd_compress_delta(parameters, self._current_energy, tag="download")

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
        # The aggregate is already in the ordinary parameter domain. FedKD's
        # SVD transform applies only to the server-to-client broadcast; the
        # model evaluated and checkpointed after a round is the post-fit
        # aggregate, not the cached pre-fit broadcast.
        return parameters

    def get_eval_metrics(self, strategy: TelemetryFedAvg, server_round: int) -> dict[str, Any]:
        metrics = {}
        if self._last_mean_rank_retained is not None:
            metrics["algorithm/fedkd/mean_rank_retained"] = self._last_mean_rank_retained
        metrics["algorithm/fedkd/energy"] = self._current_energy
        if self._last_download_report is not None:
            metrics["algorithm/fedkd/download_payload_bytes"] = (
                self._last_download_report.payload_bytes
            )
        return metrics

    def metric_keys(self) -> list[str]:
        return [
            "algorithm/fedkd/mean_rank_retained",
            "algorithm/fedkd/energy",
            "algorithm/fedkd/download_payload_bytes",
        ]
