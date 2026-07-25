"""Per-run provenance manifest.

Implements manuscript §4.3.1's confirmation-phase requirement that configuration
files be "treated as read-only once a confirmation run is launched, and their
content hashed into the run manifest for verification", and §4.3.1/ch6 §6.2's
promise that the frozen configuration is recoverable from a tagged commit.

The hash is taken over the *resolved* config rather than over the ``conf/`` files
on disk, so a run launched with ``--multirun`` overrides is identified by what it
actually executed, not by what the defaults happened to say. Two runs sharing a
``config_sha256`` ran the same configuration; §4.3.1's "one run corresponds to
exactly one configuration hash and one seed" is checkable against this field.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "run_manifest.json"


def config_sha256(cfg_dict: dict[str, Any]) -> str:
    """Stable content hash of a resolved config.

    ``sort_keys`` makes the digest independent of Hydra's composition order, so
    the same configuration reached by different override spellings hashes alike.
    """
    canonical = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    """Commit and cleanliness of the tree that produced the run.

    ``dirty`` is the load-bearing field: a confirmation run from a dirty tree is
    not reproducible from its commit alone, which §4.3.1's freeze forbids.
    Failures are recorded rather than raised, so a run outside a git checkout
    still produces a manifest.
    """

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tag": _git("describe", "--tags", "--exact-match") or None,
        "dirty": None if status is None else bool(status),
    }


def resolve_algorithm_config_name() -> str | None:
    """Which ``conf/algorithm/*.yaml`` Hydra composed for this run.

    ``algorithm.name`` does not answer this. Every §4.3.7 ablation arm declares
    ``name: fedmaq`` so they all dispatch the same hook, which means the manifest
    cannot say *which arm* a run is without the composed config's file name.
    §5.4 asks the reader to confirm arm parity "from the run manifests", so the
    manifest has to carry the identity that check keys on.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        choice = HydraConfig.get().runtime.choices.get("algorithm")
    except Exception:
        return None
    return str(choice) if choice is not None else None


def build_manifest(cfg_dict: dict[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    """Assemble the manifest record for one run."""
    if repo_root is None:
        # src/fedmaq/core/manifest.py -> repo root
        repo_root = Path(__file__).resolve().parents[3]

    algorithm = cfg_dict.get("algorithm", {}) or {}
    dataset = cfg_dict.get("dataset", {}) or {}
    heterogeneity = cfg_dict.get("heterogeneity", {}) or {}
    experiment = cfg_dict.get("experiment", {}) or {}

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        torch_version, cuda_version, gpu = None, None, None

    return {
        "config_sha256": config_sha256(cfg_dict),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "algorithm": algorithm.get("name"),
            "algorithm_config": cfg_dict.get("_algorithm_config_name")
            or resolve_algorithm_config_name(),
            "dataset": dataset.get("name"),
            "alpha": heterogeneity.get("alpha"),
            "seed": cfg_dict.get("seed"),
            "total_rounds": experiment.get("total_rounds"),
            "num_clients": experiment.get("num_clients"),
        },
        "git": _git_provenance(repo_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda": cuda_version,
            "gpu": gpu,
        },
        "config": cfg_dict,
    }


def write_run_manifest(cfg_dict: dict[str, Any], log_dir: Path) -> Path | None:
    """Write ``run_manifest.json`` beside the run's telemetry logs.

    Never raises: a provenance failure must not abort a multi-hour grid run. A
    missing manifest is visible at analysis time, whereas a crashed run is not
    recoverable.
    """
    try:
        manifest = build_manifest(cfg_dict)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / MANIFEST_FILENAME
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        if manifest["git"]["dirty"]:
            logger.warning(
                "Run manifest records a DIRTY working tree. Per manuscript "
                "§4.3.1 a confirmation run must be launched from a clean, "
                "tagged state; this run is not reproducible from its commit."
            )
        logger.info(f"Run manifest written: {path} (config_sha256={manifest['config_sha256'][:12]})")
        return path
    except Exception as exc:  # pragma: no cover - provenance must never break a run
        logger.warning(f"Failed to write run manifest: {exc}")
        return None


def resolve_log_dir() -> Path:
    """Hydra's per-run output directory, falling back to the process cwd.

    Mirrors ``TelemetryManager``'s resolution so the manifest lands beside
    ``experiment_log.jsonl`` rather than in a second location.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return Path(os.getcwd())
