"""Shared client-side training primitives (ADR-0003).

Two narrow, independent pieces factored out of the near-duplicate training
loops in ``standard.py``/``feddistill.py``/``cfd.py``/``fedmd.py``:

* :func:`run_epochs` — the single-model batch-loop atom. ``fedkd.py`` does not
  use it (its joint student+teacher optimizer doesn't fit a single-model seam;
  see ADR-0003) and keeps its own hand-rolled loop.
* :func:`compress_and_reconstruct` — the delta-compress-reconstruct tail,
  shared only by ``standard.py`` and ``fedkd.py`` (the only two baselines that
  have this tail at all).

Both must preserve the exact pre-refactor per-batch operation order (RNG draw
order, optimizer construction, ``zero_grad -> forward -> loss -> backward ->
[on_after_backward] -> step``) — this is what the Step 2 golden-diff gate
(Decision 40) checks bit-exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass
class StepResult:
    """One batch's contribution, returned by a ``run_epochs`` ``step_fn``."""

    loss: torch.Tensor
    correct: int | None = None
    total: int | None = None
    extra_sums: dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Epoch-averaged results of a :func:`run_epochs` call."""

    avg_loss: float
    last_loss: float
    batches: int
    accuracy: float | None
    extra_avgs: dict[str, float]


def run_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_fn: Callable[[torch.Tensor, torch.Tensor], StepResult],
    device: torch.device | str,
    on_after_backward: Callable[[], None] | None = None,
) -> AggregatedMetrics:
    """Run ``epochs`` passes over ``loader``, training ``model`` via ``optimizer``.

    ``step_fn(images, labels) -> StepResult`` computes the loss term (and,
    optionally, correct/total counts and any extra per-batch scalars to
    average) for one batch; the caller owns everything about what "loss"
    means (plain CE, KD-regularized, KL-against-soft-targets, ...).

    ``labels`` may be a placeholder (e.g. an unused tensor) for loaders that
    don't carry them; ``step_fn`` is free to ignore it and omit
    ``correct``/``total`` from its ``StepResult``, in which case
    :attr:`AggregatedMetrics.accuracy` is ``None``.

    Preserves the pre-existing per-batch sequence exactly:
    ``zero_grad -> forward(via step_fn) -> backward -> [on_after_backward] -> step``.
    """
    model.train()

    loss_sum = 0.0
    last_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    has_accuracy = False
    extra_sums: dict[str, float] = {}

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            result = step_fn(images, labels)

            result.loss.backward()
            if on_after_backward is not None:
                on_after_backward()
            optimizer.step()

            last_loss = result.loss.item()
            loss_sum += last_loss
            batches += 1

            if result.correct is not None and result.total is not None:
                has_accuracy = True
                correct += result.correct
                total += result.total

            for key, value in result.extra_sums.items():
                extra_sums[key] = extra_sums.get(key, 0.0) + value

    avg_loss = loss_sum / batches if batches > 0 else 0.0
    accuracy = (correct / total if total > 0 else 0.0) if has_accuracy else None
    extra_avgs = {k: (v / batches if batches > 0 else 0.0) for k, v in extra_sums.items()}

    return AggregatedMetrics(
        avg_loss=avg_loss,
        last_loss=last_loss,
        batches=batches,
        accuracy=accuracy,
        extra_avgs=extra_avgs,
    )


def compress_and_reconstruct(
    original_params: list[np.ndarray],
    updated_params: list[np.ndarray],
    compressor_hook: Any,
) -> tuple[list[np.ndarray], int]:
    """Delta -> compress -> reconstruct tail shared by ``standard`` and ``fedkd``.

    ``w_new_reconstructed = w_old + compress(w_new - w_old)``. Returns the
    reconstructed parameters (what gets uploaded/aggregated) and the
    compressed byte size.
    """
    deltas = [u - o for u, o in zip(updated_params, original_params, strict=True)]
    compressed_deltas, byte_size = compressor_hook.compress(deltas)
    reconstructed_params = [
        o + cd for o, cd in zip(original_params, compressed_deltas, strict=True)
    ]
    return reconstructed_params, byte_size
