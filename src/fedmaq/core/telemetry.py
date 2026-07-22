"""Telemetry manager for tracking metrics and logging to Weights & Biases."""

import csv
import json
import logging
import os
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.config_defaults import require_num_public_samples

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

# Metric keys every algorithm may emit, independent of which hook is active.
# Algorithm-specific keys (e.g. ``algorithm/fedmaq/avg_q``) are supplied by the
# active hook's ``metric_keys()`` and composed in below — see
# ``register_hook_metric_keys`` and ``_write_local_logs``.
_COMMON_CSV_FIELDNAMES: list[str] = [
    "round",
    "test/loss",
    "test/accuracy",
    "test/precision",
    "test/recall",
    "test/f1",
    "communication/round_bytes",
    "communication/cumulative_bytes",
    "communication/cumulative_mb",
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
]


class TelemetryManager:
    """Manages telemetry logging to WandB, local files, and console."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        exp_config = config.get("experiment", config)
        self.enabled = exp_config.get("telemetry", {}).get("wandb_enabled", True)
        self.project = exp_config.get("telemetry", {}).get("project", "fedmaq-experiments")
        self.run_name = exp_config.get("telemetry", {}).get("run_name", None)
        self.run = None

        # Communication and time accumulators
        self.cumulative_bytes: int = 0
        self.cumulative_time: float = 0.0
        self.cumulative_client_time: float = 0.0
        self.cumulative_server_time: float = 0.0
        self.cumulative_wall_time: float = 0.0

        # Per-round fit snapshots, read back by the strategy's evaluate(). See
        # ``record_fit_round`` (called from ``aggregate_fit``).
        self.last_round_bytes = 0
        self.last_round_time = 0.0
        self.last_client_time = 0.0
        self.last_server_time = 0.0
        self.last_wall_time = 0.0
        self.last_client_bytes_stats: dict[str, float] = {}
        self.last_round_client_metrics: dict[str, float] = {}

        # Real (not simulated) wall-clock timer, measured across aggregate_fit calls.
        self._last_wall_ts = time.perf_counter()

        # Algorithm-specific CSV keys declared by the active hook (see
        # ``register_hook_metric_keys``), composed into the stable schema below.
        self._hook_metric_keys: list[str] = []

        # Local tracking setup in Hydra's output directory, falling back to current dir
        if _HYDRA_AVAILABLE:
            try:
                self.log_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                self.log_dir = Path(os.getcwd())
        else:
            self.log_dir = Path(os.getcwd())

        self.jsonl_path = self.log_dir / "experiment_log.jsonl"
        self.csv_path = self.log_dir / "experiment_log.csv"

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

        # Flatten config dict for wandb configuration logging
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
        """Compute this round's simulated delays, byte counts, and client-metric
        aggregates from the raw fit results, and snapshot them for the strategy's
        ``evaluate()`` to read back via ``last_round_*``.

        Returns ``(round_time, round_total_bytes)`` for the caller's ``metrics``
        dict. When ``results`` is empty, only ``last_round_client_metrics`` is
        reset and ``(0.0, 0)`` is returned — the caller should not set
        ``round_time``/``round_bytes`` on ``metrics`` in that case (matching the
        pre-refactor behavior of skipping those keys entirely).
        """
        # Aggregate client-side training metrics (weighted mean, simple mean for
        # epochs_trained), reported by the client hooks in fit() metrics.
        self.last_round_client_metrics = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples > 0:
            numeric_keys = set()
            for _, fit_res in results:
                for k, v in fit_res.metrics.items():
                    if isinstance(v, (int, float)) and k not in (
                        "partition_id",
                        "bytes_uploaded",
                    ):
                        numeric_keys.add(k)

            for k in numeric_keys:
                if k == "epochs_trained":
                    simple_sum = sum(float(fit_res.metrics.get(k, 0.0)) for _, fit_res in results)
                    self.last_round_client_metrics[f"client/avg_{k}"] = simple_sum / len(results)
                else:
                    weighted_sum = sum(
                        float(fit_res.metrics.get(k, 0.0)) * fit_res.num_examples
                        for _, fit_res in results
                    )
                    self.last_round_client_metrics[f"client/avg_{k}"] = (
                        weighted_sum / total_examples
                    )

        if not results:
            return 0.0, 0

        # Model download size in bytes (hook may compress it, e.g. FedKD's SVD path).
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model_size_bytes = strategy.hook.download_size_bytes(strategy, ndarrays)
        else:
            model_size_bytes = 0

        round_delays = []
        round_bytes_uploaded = 0
        round_bytes_downloaded = 0
        client_bytes_uploaded: list[int] = []

        exp_config = strategy.config.get("experiment", strategy.config)
        epochs = exp_config.get("local_epochs", 5)
        public_epochs = int(strategy.config.get("algorithm", {}).get("public_epochs", 5))
        num_public = require_num_public_samples(strategy.config)
        compute_scale = strategy.hook.compute_speed_scale()

        for client_proxy, fit_res in results:
            cid = int(fit_res.metrics.get("partition_id", -1))
            if cid < 0 or cid >= strategy.num_clients:
                cid = hash(client_proxy.cid) % strategy.num_clients

            bytes_uploaded = int(fit_res.metrics.get("bytes_uploaded", model_size_bytes))
            client_bytes_uploaded.append(bytes_uploaded)
            num_samples = fit_res.num_examples
            train_sample_count = strategy.hook.local_train_sample_count(
                num_samples=num_samples,
                epochs=epochs,
                num_public=num_public,
                public_epochs=public_epochs,
                server_round=server_round,
            )

            t_download, t_train, t_upload = strategy.network_simulator.simulate_client_delay(
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

        # Decouple client and server simulated delays
        client_sim_time = max(round_delays) if round_delays else 0.0

        # Server compute time: non-zero for hooks with server-side work (KD).
        server_sim_time = strategy.hook.server_sim_time(strategy, results, aggregated_parameters)

        round_time = client_sim_time + server_sim_time
        round_total_bytes = round_bytes_downloaded + round_bytes_uploaded

        self.last_round_bytes = round_total_bytes
        self.last_round_time = round_time
        self.last_client_time = client_sim_time
        self.last_server_time = server_sim_time

        if client_bytes_uploaded:
            arr = np.array(client_bytes_uploaded, dtype=np.float64)
            self.last_client_bytes_stats = {
                "communication/client_bytes_uploaded_mean": float(arr.mean()),
                "communication/client_bytes_uploaded_min": float(arr.min()),
                "communication/client_bytes_uploaded_max": float(arr.max()),
                "communication/client_bytes_uploaded_std": float(arr.std()),
            }
        else:
            self.last_client_bytes_stats = {}

        # Real wall-clock elapsed since the previous round's aggregate_fit call.
        now = time.perf_counter()
        self.last_wall_time = now - self._last_wall_ts
        self._last_wall_ts = now

        return round_time, round_total_bytes

    def log(
        self,
        round_num: int,
        metrics: dict[str, Any],
    ) -> None:
        """Log key metrics for a communication round using hierarchical namespaces."""
        # Accumulate communication bytes
        round_bytes = metrics.get("communication/round_bytes", 0)
        self.cumulative_bytes += round_bytes
        cumulative_kb = self.cumulative_bytes / 1024.0
        cumulative_mb = cumulative_kb / 1024.0

        if "communication/cumulative_bytes" not in metrics:
            metrics["communication/cumulative_bytes"] = self.cumulative_bytes
        if "communication/cumulative_mb" not in metrics:
            metrics["communication/cumulative_mb"] = cumulative_mb

        # Accumulate physical simulated time
        round_time = metrics.get("system/round_time_sec", 0.0)
        self.cumulative_time += round_time
        if "system/cumulative_time_sec" not in metrics:
            metrics["system/cumulative_time_sec"] = self.cumulative_time

        # Accumulate client and server times
        client_time = metrics.get("system/client_sim_time_sec", 0.0)
        server_time = metrics.get("system/server_sim_time_sec", 0.0)
        self.cumulative_client_time += client_time
        self.cumulative_server_time += server_time
        if "system/cumulative_client_time_sec" not in metrics:
            metrics["system/cumulative_client_time_sec"] = self.cumulative_client_time
        if "system/cumulative_server_time_sec" not in metrics:
            metrics["system/cumulative_server_time_sec"] = self.cumulative_server_time

        # Accumulate real wall-clock time (not simulated)
        wall_time = metrics.get("system/wall_time_sec", 0.0)
        self.cumulative_wall_time += wall_time
        if "system/cumulative_wall_time_sec" not in metrics:
            metrics["system/cumulative_wall_time_sec"] = self.cumulative_wall_time

        # Print clean summary to console
        test_acc = metrics.get("test/accuracy", 0.0)
        test_loss = metrics.get("test/loss", 0.0)
        logger.info(
            f"Round {round_num:3d} | Test Acc: {test_acc * 100:6.2f}% | "
            f"Test Loss: {test_loss:6.4f} | Comm: {cumulative_mb:7.3f} MB | "
            f"Sim Time: {self.cumulative_time:8.2f}s"
        )

        if self.enabled and self.run is not None:
            self.run.log(metrics, step=round_num)

        # Log locally
        self._write_local_logs(metrics)

    def _write_local_logs(self, metrics: dict[str, Any]) -> None:
        # 1. Write to JSONL
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write to local JSONL log: {exc}")

        # 2. Write to CSV with a stable schema captured on first write.
        #    Extra keys beyond the schema are silently dropped; missing keys
        #    are written as empty strings to keep columns aligned.
        try:
            if self._csv_fieldnames is None:
                canonical = _COMMON_CSV_FIELDNAMES + self._hook_metric_keys
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
        """Close the WandB run."""
        if self.enabled and self.run is not None:
            self.run.finish()
            logger.info("WandB run finished.")

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
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
