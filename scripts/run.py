"""Hydra CLI entrypoint for the Flower simulation.

Thin wrapper: the simulation logic lives in :mod:`fedmaq.simulation` so it is
importable and testable in-process. This module only binds the Hydra config search
path (``../conf`` relative to this file) to the decorator-free :func:`fedmaq.simulation.run`,
under exclusive ownership of the run directory (:func:`scripts.run_guard.run_dir_guard`).

The guard runs before ``hydra.main``, because Hydra rewrites ``.hydra/*.yaml`` and
opens the job log in the run directory before the task function starts. The run
directory is resolved once by composition and pinned as ``hydra.run.dir``, so a
timestamped default cannot resolve differently the second time.
"""

import logging
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedmaq.simulation import run
from scripts.run_guard import LockHeldError, PriorEvidenceError, run_dir_guard

logging.basicConfig(level=logging.INFO)

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
RUN_DIR_KEY = "hydra.run.dir"


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint; delegates to the decorator-free :func:`run`."""
    run(cfg)


def _resolved_run_dir(overrides: list[str]) -> Path:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    return Path(str(cfg.hydra.run.dir)).resolve()


def cli() -> None:
    """Own the run directory, then hand the pinned command line to Hydra."""
    overrides = sys.argv[1:]
    if any(arg.startswith("-") for arg in overrides):
        # Hydra's own flags (--help, --cfg, --multirun, ...) start no single run
        # here; sweeps go through run_matrix.py, which passes plain overrides.
        main()
        return
    run_dir = _resolved_run_dir(overrides)
    pinned = [arg for arg in overrides if arg.lstrip("+").split("=", 1)[0] != RUN_DIR_KEY]
    sys.argv = [sys.argv[0], *pinned, f"{RUN_DIR_KEY}={run_dir.as_posix()}"]
    try:
        with run_dir_guard(run_dir):
            main()
    except (LockHeldError, PriorEvidenceError) as exc:
        raise SystemExit(f"[run] {exc}") from exc


if __name__ == "__main__":
    cli()
