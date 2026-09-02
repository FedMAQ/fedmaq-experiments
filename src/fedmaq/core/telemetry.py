"""Telemetry manager for tracking metrics and logging to Weights & Biases."""

import csv
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.config_defaults import require_num_public_samples
from fedmaq.core.payload_archive import PayloadArchive, payload_capture_enabled

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from hydra.core.hydra_config import HydraConfig

    _HYDRA_AVAILABLE = True
except Exception:
    _HYDRA_AVAILABLE = False

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

COMMON_CSV_FIELDNAMES: list[str] = [
    "round",
    "test/loss",
    "test/accuracy",
    "test/precision",
    "test/recall",
    "test/f1",
    "communication/round_bytes",
    "communication/round_payload_bytes",
    "communication/round_upload_bytes",
    "communication/round_secondary_bytes",
    "communication/cumulative_bytes",
    "communication/cumulative_mb",
    "communication/cumulative_upload_bytes",
    "communication/cumulative_upload_mb",
    "system/round_time_sec",
    "system/cumulative_time_sec",
    "system/client_sim_time_sec",
    "system/cumulative_client_time_sec",
    "system/server_sim_time_sec",
    "system/cumulative_server_time_sec",
    "system/wall_time_sec",
    "system/cumulative_wall_time_sec",
    "communication/client_bytes_uploaded_mean",
    "communication/client_bytes_uploaded_min",
    "communication/client_bytes_uploaded_max",
    "communication/client_bytes_uploaded_std",
    "client/avg_train_loss",
    "client/avg_train_acc",
    "client/avg_local_loss",
    "client/avg_epochs_trained",
    "client/avg_q",
    "client/modeled_capacity_mb_mean",
    "client/modeled_capacity_mb_min",
    "client/modeled_capacity_mb_max",
]


@dataclass(frozen=True)
class RoundSnapshot:
    """Simulated delay/byte-accounting for one fit round, read back by ``evaluate()``.

    Round-0 zeroing is decided once by :meth:`TelemetryManager.snapshot_for_round`
    rather than repeated per-field at each call site.
    """

    round_bytes: int = 0
    round_payload_bytes: int = 0
    round_upload_bytes: int = 0
    round_download_bytes: int = 0
    #: DAdaQuant's as-published secondary total (#26), summed across clients
    #: that reported one. ``None`` when no client in the round reported it
    #: (every arm except DAdaQuant) -- distinct from a measured 0.
    round_secondary_bytes: int | None = None
    round_time: float = 0.0
    client_time: float = 0.0
    server_time: float = 0.0
    wall_time: float = 0.0
    client_bytes_stats: dict[str, float] = field(default_factory=dict)
    round_client_metrics: dict[str, float] = field(default_factory=dict)


class TelemetryManager:
    """Manages telemetry logging to WandB, local files, and console."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        exp_config = config.get("experiment", config)
        self.enabled = exp_config.get("telemetry", {}).get("wandb_enabled", True)
        self.project = exp_config.get("telemetry", {}).get("project", "fedmaq-experiments")
        self.run_name = exp_config.get("telemetry", {}).get("run_name", None)
        self.run: Any | None = None

        self.cumulative_bytes: int = 0
        self.cumulative_upload_bytes: int = 0
        self.cumulative_time: float = 0.0
        self.cumulative_client_time: float = 0.0
        self.cumulative_server_time: float = 0.0
        self.cumulative_wall_time: float = 0.0

        self._last_snapshot = RoundSnapshot()

        self._last_wall_ts = time.perf_counter()

        # Algorithm-specific CSV keys declared by the active hook (see
        # ``register_hook_metric_keys``), composed into the stable schema below.
        self._hook_metric_keys: list[str] = []

        if _HYDRA_AVAILABLE:
            try:
                self.log_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                self.log_dir = Path(os.getcwd())
        else:
            self.log_dir = Path(os.getcwd())

        self.jsonl_path = self.log_dir / "experiment_log.jsonl"
        self.csv_path = self.log_dir / "experiment_log.csv"
        self.payload_archive = PayloadArchive(self.log_dir)

        # Stable CSV field schema — captured on first write, held constant thereafter.
        # Rows with missing keys are written as empty strings; extra keys are silently
        # ignored. This prevents header duplication when algorithm-specific metrics
        # (e.g. DAdaQuant q_t) appear only in certain rounds.
        self._csv_fieldnames: list[str] | None = None

    def init_wandb(self) -> None:
        """Initialize WandB connection if enabled."""
        if not self.enabled:
            logger.info("WandB telemetry is disabled.")
            return

        if not _WANDB_AVAILABLE:
            logger.warning("WandB is not installed. Telemetry will be local-only.")
            self.enabled = False
            return

        flat_config = self._flatten_dict(self.config)

        exp_config = self.config.get("experiment", self.config)
        try:
            self.run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=flat_config,
                mode=exp_config.get("telemetry", {}).get("mode", "online"),
            )
            logger.info(f"WandB run initialized: {self.run.name if self.run else 'offline'}")
        except Exception as exc:
            logger.warning(f"Could not initialize WandB: {exc}. Telemetry will be console-only.")
            self.enabled = False

    def register_hook_metric_keys(self, keys: list[str]) -> None:
        """Declare the algorithm-specific metric keys the active hook may emit.

        Must be called once, before the first :meth:`log`, so the CSV header
        stays stable even when a key (e.g. FedMAQ's grad-norm stats) only
        appears starting round 1 rather than round 0.
        """
        self._hook_metric_keys = list(keys)

    def record_fit_round(
        self,
        strategy: "TelemetryFedAvg",
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters | None,
    ) -> tuple[float, int]:
        """Snapshot this round's delays, communication totals, and client metrics.

        ``round_secondary_bytes`` remains ``None`` when no client reports the
        DAdaQuant-only as-published axis. Payload replay persistence is enabled
        only by the shared payload-capture predicate.
        """
        round_client_metrics: dict[str, float] = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples > 0:
            numeric_keys = set()
            for _, fit_res in results:
                for k, v in fit_res.metrics.items():
                    if isinstance(v, (int, float)) and k not in (
                        "partition_id",
                        "bytes_uploaded",
                        "payload_bytes",
                        "secondary_bytes_uploaded",
                    ):
                        numeric_keys.add(k)

            for k in numeric_keys:
                if k == "epochs_trained":
                    simple_sum = sum(float(fit_res.metrics.get(k, 0.0)) for _, fit_res in results)
                    round_client_metrics[f"client/avg_{k}"] = simple_sum / len(results)
                else:
                    weighted_sum = sum(
                        float(fit_res.metrics.get(k, 0.0)) * fit_res.num_examples
                        for _, fit_res in results
                    )
                    round_client_metrics[f"client/avg_{k}"] = weighted_sum / total_examples

        # Modeled client memory capacity stats for sampled clients in this round (§4.1)
        cost_model = getattr(strategy, "cost_model", None)
        if results and cost_model is not None and hasattr(cost_model, "client_memory"):
            sampled_capacities = [
                float(cost_model.client_memory[strategy._partition_sort_key(cp, fr)])
                for cp, fr in results
                if strategy._partition_sort_key(cp, fr) < len(cost_model.client_memory)
            ]
            if sampled_capacities:
                round_client_metrics["client/modeled_capacity_mb_mean"] = float(
                    np.mean(sampled_capacities)
                )
                round_client_metrics["client/modeled_capacity_mb_min"] = float(
                    np.min(sampled_capacities)
                )
                round_client_metrics["client/modeled_capacity_mb_max"] = float(
                    np.max(sampled_capacities)
                )

        if not results:
            self._last_snapshot = replace(
                self._last_snapshot, round_client_metrics=round_client_metrics
            )
            return 0.0, 0

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            download_report = strategy.hook.download_size_bytes(strategy, ndarrays)
            model_size_bytes = download_report.measured_bytes
        else:
            download_report = None
            model_size_bytes = 0

        round_delays = []
        round_bytes_uploaded = 0
        round_bytes_downloaded = 0
        round_payload_bytes = 0
        round_secondary_bytes = 0
        has_secondary_bytes = False
        client_bytes_uploaded: list[int] = []
        round_framed_uploads: dict[int, bytes] = {}

        exp_config = strategy.config.get("experiment", strategy.config)
        capture_enabled = payload_capture_enabled(strategy.config)
        download_payloads = (
            list(download_report.payloads)
            if download_report is not None and capture_enabled
            else []
        )
        round_download_payloads: dict[int, list[bytes]] = {}
        epochs = exp_config.get("local_epochs", 5)
        public_epochs = int(strategy.config.get("algorithm", {}).get("public_epochs", 5))
        num_public = require_num_public_samples(strategy.config)
        compute_scale = strategy.hook.compute_speed_scale()

        for client_proxy, fit_res in results:
            cid = strategy._partition_sort_key(client_proxy, fit_res)

            bytes_uploaded = int(fit_res.metrics.get("bytes_uploaded", model_size_bytes))
            client_bytes_uploaded.append(bytes_uploaded)
            round_payload_bytes += int(fit_res.metrics["payload_bytes"])

            secondary_bytes = fit_res.metrics.get("secondary_bytes_uploaded")
            if secondary_bytes is not None:
                round_secondary_bytes += int(secondary_bytes)
                has_secondary_bytes = True

            payloads_framed = fit_res.metrics.get("payloads_framed")
            if isinstance(payloads_framed, bytes) and payloads_framed:
                round_framed_uploads[cid] = payloads_framed

            if download_payloads:
                # Every sampled client receives the same server broadcast.
                # Reusing the list lets pickle memoize the bytes while the
                # per-recipient mapping preserves aggregate replay semantics.
                round_download_payloads[cid] = download_payloads

            num_samples = fit_res.num_examples
            train_sample_count = strategy.hook.local_train_sample_count(
                num_samples=num_samples,
                epochs=epochs,
                num_public=num_public,
                public_epochs=public_epochs,
                server_round=server_round,
            )

            t_download, t_train, t_upload = strategy.cost_model.client_round_delay(
                cid=cid,
                model_size_bytes=model_size_bytes,
                bytes_uploaded=bytes_uploaded,
                train_sample_count=train_sample_count,
                compute_scale=compute_scale,
            )

            client_total_time = t_download + t_train + t_upload
            round_delays.append(client_total_time)
            round_bytes_downloaded += model_size_bytes
            round_bytes_uploaded += bytes_uploaded

        self.payload_archive.record_round(
            server_round, round_framed_uploads, round_download_payloads
        )

        client_sim_time = max(round_delays) if round_delays else 0.0

        server_sim_time = strategy.hook.server_sim_time(strategy, results, aggregated_parameters)

        round_time = client_sim_time + server_sim_time
        round_total_bytes = round_bytes_downloaded + round_bytes_uploaded

        if client_bytes_uploaded:
            arr = np.array(client_bytes_uploaded, dtype=np.float64)
            client_bytes_stats = {
                "communication/client_bytes_uploaded_mean": float(arr.mean()),
                "communication/client_bytes_uploaded_min": float(arr.min()),
                "communication/client_bytes_uploaded_max": float(arr.max()),
                "communication/client_bytes_uploaded_std": float(arr.std()),
            }
        else:
            client_bytes_stats = {}

        now = time.perf_counter()
        wall_time = now - self._last_wall_ts
        self._last_wall_ts = now

        self._last_snapshot = RoundSnapshot(
            round_bytes=round_total_bytes,
            round_payload_bytes=round_payload_bytes,
            round_upload_bytes=round_bytes_uploaded,
            round_download_bytes=round_bytes_downloaded,
            round_secondary_bytes=round_secondary_bytes if has_secondary_bytes else None,
            round_time=round_time,
            client_time=client_sim_time,
            server_time=server_sim_time,
            wall_time=wall_time,
            client_bytes_stats=client_bytes_stats,
            round_client_metrics=round_client_metrics,
        )

        return round_time, round_total_bytes

    def snapshot_for_round(self, server_round: int) -> RoundSnapshot:
        """Round-0 zeroing, decided once here rather than per-field at each call site.

        ``round_client_metrics`` is carried through unguarded for round 0 (matching
        pre-refactor behavior, where that field was never gated by ``server_round``).
        """
        if server_round > 0:
            return self._last_snapshot
        return RoundSnapshot(round_client_metrics=self._last_snapshot.round_client_metrics)

    def log(
        self,
        round_num: int,
        metrics: dict[str, Any],
    ) -> None:
        """Log key metrics for a communication round using hierarchical namespaces."""
        round_bytes = metrics.get("communication/round_bytes", 0)
        self.cumulative_bytes += round_bytes
        cumulative_kb = self.cumulative_bytes / 1024.0
        cumulative_mb = cumulative_kb / 1024.0

        if "communication/cumulative_bytes" not in metrics:
            metrics["communication/cumulative_bytes"] = self.cumulative_bytes
        if "communication/cumulative_mb" not in metrics:
            metrics["communication/cumulative_mb"] = cumulative_mb

        round_upload_bytes = metrics.get("communication/round_upload_bytes", 0)
        self.cumulative_upload_bytes += round_upload_bytes
        cumulative_upload_mb = (self.cumulative_upload_bytes / 1024.0) / 1024.0

        if "communication/cumulative_upload_bytes" not in metrics:
            metrics["communication/cumulative_upload_bytes"] = self.cumulative_upload_bytes
        if "communication/cumulative_upload_mb" not in metrics:
            metrics["communication/cumulative_upload_mb"] = cumulative_upload_mb

        round_time = metrics.get("system/round_time_sec", 0.0)
        self.cumulative_time += round_time
        if "system/cumulative_time_sec" not in metrics:
            metrics["system/cumulative_time_sec"] = self.cumulative_time

        client_time = metrics.get("system/client_sim_time_sec", 0.0)
        server_time = metrics.get("system/server_sim_time_sec", 0.0)
        self.cumulative_client_time += client_time
        self.cumulative_server_time += server_time
        if "system/cumulative_client_time_sec" not in metrics:
            metrics["system/cumulative_client_time_sec"] = self.cumulative_client_time
        if "system/cumulative_server_time_sec" not in metrics:
            metrics["system/cumulative_server_time_sec"] = self.cumulative_server_time

        wall_time = metrics.get("system/wall_time_sec", 0.0)
        self.cumulative_wall_time += wall_time
        if "system/cumulative_wall_time_sec" not in metrics:
            metrics["system/cumulative_wall_time_sec"] = self.cumulative_wall_time

        test_acc = metrics.get("test/accuracy", 0.0)
        test_loss = metrics.get("test/loss", 0.0)
        logger.info(
            f"Round {round_num:3d} | Test Acc: {test_acc * 100:6.2f}% | "
            f"Test Loss: {test_loss:6.4f} | Comm: {cumulative_mb:7.3f} MB | "
            f"Sim Time: {self.cumulative_time:8.2f}s"
        )

        # Local logs first, and WandB after: the CSV/JSONL are the artifacts every
        # reported result is computed from, while WandB is for watching a sweep in
        # progress. In ``mode: "offline"`` the ordering was academic, but online it
        # is not -- a transient network failure at round 57 of a 100-round run would
        # otherwise both abort the run and lose that round's local row.
        self._write_local_logs(metrics)

        if self.enabled and self.run is not None:
            try:
                self.run.log(metrics, step=round_num)
            except Exception as exc:  # noqa: BLE001 - see comment above
                logger.warning(
                    f"WandB log failed at round {round_num}: {exc}. "
                    f"Local logs are unaffected; continuing."
                )

    def _write_local_logs(self, metrics: dict[str, Any]) -> None:
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write to local JSONL log: {exc}")

        try:
            if self._csv_fieldnames is None:
                canonical = COMMON_CSV_FIELDNAMES + self._hook_metric_keys
                seen = set(canonical)
                fieldnames = list(canonical)
                for key in sorted(metrics.keys()):
                    if key not in seen:
                        fieldnames.append(key)
                self._csv_fieldnames = fieldnames

            file_exists = self.csv_path.exists()
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self._csv_fieldnames,
                    extrasaction="ignore",
                    restval="",
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)
        except Exception as exc:
            logger.warning(f"Failed to write to local CSV log: {exc}")

    def finish(self) -> None:
        """Close the WandB run.

        Guarded for the same reason as :meth:`log`: online mode flushes over the
        network here, and a run that has already trained for 100 rounds and
        written its checkpoint must not be reported as failed because its
        telemetry upload could not complete.
        """
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
                logger.info("WandB run finished.")
            except Exception as exc:  # noqa: BLE001 - see docstring
                logger.warning(f"WandB finish failed: {exc}. Local logs are complete.")

    def _flatten_dict(
        self, d: Mapping[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Helper to flatten nested dictionaries (such as Hydra Omegaconf)."""
        items: dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, Mapping):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
