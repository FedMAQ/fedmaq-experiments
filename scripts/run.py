"""Hydra CLI entrypoint for the Flower simulation.

Thin wrapper: the simulation logic lives in :mod:`fedmaq.simulation` so it is
importable and testable in-process. This module only binds the Hydra config search
path (``../conf`` relative to this file) to the decorator-free :func:`fedmaq.simulation.run`.
"""

import logging

import hydra
from omegaconf import DictConfig

from fedmaq.simulation import run

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint; delegates to the decorator-free :func:`run`."""
    run(cfg)


if __name__ == "__main__":
    main()
