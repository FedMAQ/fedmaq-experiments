"""Final global-model persistence.

Manuscript §5.2.1 promises "t-SNE visualizations of the global model's
penultimate-layer feature space for FedMAQ against a heavily quantized
parameter-averaging baseline", and `chapter_5.tex` carries a visual frame for
them. Those plots are produced *after* the grid finishes, from the trained
global model — but until 2026-07-30 nothing in this repo wrote that model to
disk. A run ended at ``telemetry.finish()`` and its weights went out of scope
with the process, leaving round-metric CSVs, the WandB run, and
``run_manifest.json`` as the only artifacts. The figure would have been
unbuildable once the 183-run grid completed, and recoverable only by re-running
CIFAR-10 at $\\alpha = 0.1$ for FedMAQ plus a quantized baseline.

So every run now writes ``final_global_model.pt`` beside its telemetry. The
checkpoint is taken from inside ``evaluate_fn`` at the last round rather than
from strategy state after ``run_simulation`` returns: ``evaluate_fn`` holds
exactly the parameter vector the run's reported final accuracy was computed on,
whereas the strategy object lives in a Flower ServerApp whose state is not
reliably reachable from the driver once the simulation exits.

Cost is one state_dict per run — about 8.5 MB for MobileNetV2GN, roughly 1.6 GB
across the whole grid. Cheap against the alternative of discovering the gap
after dispatch.
"""

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("fedmaq")

FINAL_MODEL_FILENAME = "final_global_model.pt"


def write_final_global_model(
    model: Any,
    log_dir: Path,
    server_round: int,
    total_rounds: int,
) -> Path | None:
    """Persist the global model's ``state_dict`` on the final round.

    A no-op on every other round. Returns the written path, or ``None`` if this
    is not the final round or the write failed.

    Failure is logged and swallowed rather than raised: a run that trained for
    100 rounds should not be lost to a disk error in its last second, and a
    missing checkpoint is visible at analysis time in a way a crashed run is
    not. This mirrors the failure posture of ``manifest.write_run_manifest``.
    """
    if server_round != total_rounds:
        return None

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / FINAL_MODEL_FILENAME
        # Move to CPU first so the checkpoint loads on a machine without CUDA
        # (analysis and figure generation run on the local workstation, not the
        # datacenter allocation that produced the run).
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.error(f"Failed to write final global model to {log_dir}: {exc}")
        return None

    logger.info(f"Final global model written: {path} (round {server_round})")
    return path
