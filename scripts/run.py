"""Hydra CLI entrypoint for the Flower simulation.

Thin wrapper: the simulation logic lives in :mod:`fedmaq.simulation` so it is
importable and testable in-process. This module only binds the Hydra config search
path (``../conf`` relative to this file) to the decorator-free :func:`fedmaq.simulation.run`,
under exclusive ownership of the run directory (:func:`scripts.run_guard.run_dir_guard`).
"""

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedmaq.simulation import run
from scripts.run_guard import LockHeldError, PriorEvidenceError, run_dir_guard

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint; delegates to the decorator-free :func:`run`."""
    try:
        with run_dir_guard(Path(HydraConfig.get().runtime.output_dir)):
            run(cfg)
    except (LockHeldError, PriorEvidenceError) as exc:
        raise SystemExit(f"[run] {exc}") from exc


if __name__ == "__main__":
    main()
