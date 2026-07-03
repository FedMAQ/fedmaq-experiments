"""Telemetry manager for tracking metrics and logging to Weights & Biases."""

import csv
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import wandb

logger = logging.getLogger(__name__)

try:
    from hydra.core.hydra_config import HydraConfig

    _HYDRA_AVAILABLE = True
except Exception:
    _HYDRA_AVAILABLE = False


class TelemetryManager:
    """Manages telemetry logging to WandB, local files, and console."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        exp_config = config.get("experiment", config)
        self.enabled = exp_config.get("telemetry", {}).get("wandb_enabled", True)
        self.project = exp_config.get("telemetry", {}).get(
            "project", "fedmaq-experiments"
        )
        self.run_name = exp_config.get("telemetry", {}).get("run_name", None)
        self.run = None

        # Communication and time accumulators
        self.cumulative_bytes: int = 0
        self.cumulative_time: float = 0.0
        self.cumulative_client_time: float = 0.0
        self.cumulative_server_time: float = 0.0

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
            logger.info(
                f"WandB run initialized: {self.run.name if self.run else 'offline'}"
            )
        except Exception as exc:
            logger.warning(
                f"Could not initialize WandB: {exc}. Telemetry will be console-only."
            )
            self.enabled = False

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
                self._csv_fieldnames = sorted(metrics.keys())

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
