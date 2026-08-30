"""Opt-in persistence for the raw payloads used by transport accounting."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def payload_capture_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether raw payload capture is enabled for a configuration."""
    experiment = config.get("experiment", config)
    telemetry = experiment.get("telemetry", {})
    return bool(telemetry.get("log_payloads", False))


class PayloadArchive:
    """Persist per-round upload and download payloads beneath a run directory."""

    def __init__(self, log_dir: str | Path) -> None:
        self.payloads_dir = Path(log_dir) / "payloads"

    def record_round(
        self,
        server_round: int,
        framed_uploads: Mapping[int, bytes],
        downloads: Mapping[int, Sequence[bytes]],
    ) -> None:
        """Persist non-empty upload and download captures for one round."""
        from fedmaq.baselines.transport import unpack_payloads

        upload_payloads = {
            cid: unpack_payloads(framed)
            for cid, framed in framed_uploads.items()
            if isinstance(framed, bytes) and framed
        }
        download_payloads = {cid: payloads for cid, payloads in downloads.items() if payloads}
        if upload_payloads:
            self._write(
                server_round,
                upload_payloads,
                f"round_{server_round:04d}.pkl",
                "payloads",
            )
        if download_payloads:
            self._write(
                server_round,
                download_payloads,
                f"download_round_{server_round:04d}.pkl",
                "download payloads",
            )

    def _write(
        self,
        server_round: int,
        payloads: Mapping[int, Sequence[bytes]],
        filename: str,
        description: str,
    ) -> None:
        try:
            self.payloads_dir.mkdir(parents=True, exist_ok=True)
            with open(self.payloads_dir / filename, "wb") as file:
                pickle.dump(payloads, file)
        except Exception as exc:  # noqa: BLE001 - capture must never abort a round
            logger.warning("Failed to write round %s %s: %s", server_round, description, exc)


__all__ = ["PayloadArchive", "payload_capture_enabled"]
