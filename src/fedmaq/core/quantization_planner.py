"""Owns FedMAQ's "how q is chosen" concept: probe -> EMA-smooth -> normalize -> assign.

Consolidates state that was previously scattered across ``FedMAQHook`` fields
(``_grad_norm_model``, ``_grad_norm_ema``, ``_round_client_q``, ``_last_grad_norms``,
``_last_assigned_q``) into one planner with one output value, :class:`QuantPlan`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from flwr.common import FitIns, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.config_defaults import RunContext
from fedmaq.core.models import set_model_parameters
from fedmaq.core.partitioning import get_client_loader

logger = logging.getLogger(__name__)

# Permissible bit-width set per manuscript §4.2: Q = {1,...,8, 16, 32}.
# 16/32-bit tiers are effectively "escape" precision levels for well-resourced
# clients; reachability depends on c_unit and configured memory range (see
# conf/algorithm/fedmaq.yaml).
DEFAULT_BIT_WIDTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32)

# A single stochastic gradient-norm probe: model + a client's (images, labels)
# batch in, per-client scalar norm out. The real adapter runs one forward+backward
# pass; tests inject a synthetic probe to drive plan_round without a dataset.
GradNormProbe = Callable[[nn.Module, torch.Tensor, torch.Tensor], float]


@dataclass(frozen=True)
class _QuantParams:
    """Parsed FedMAQ quantization hyperparameters for one plan_round call.

    Grouping these keeps the q-assignment helper's signature small and, via
    ``from_cfg``, preserves the F8 fail-loud contract: the algorithm-defining
    knobs (``q_min``/``q_max``/``c_unit``/``formulation``) are read with no
    default so a missing/renamed key raises up front, before any probe work.
    """

    q_min: int
    q_max: int
    c_unit: float
    formulation: int
    gamma1: float
    gamma2: float
    lambda_val: float
    tau_g: float
    tau_n: float
    bit_widths: tuple[int, ...]

    @classmethod
    def from_cfg(cls, alg_cfg: dict[str, Any]) -> _QuantParams:
        return cls(
            q_min=int(alg_cfg["q_min"]),
            q_max=int(alg_cfg["q_max"]),
            c_unit=float(alg_cfg["c_unit"]),
            formulation=int(alg_cfg["formulation"]),
            gamma1=float(alg_cfg.get("gamma1", 0.5)),
            gamma2=float(alg_cfg.get("gamma2", 0.5)),
            lambda_val=float(alg_cfg.get("lambda_val", 1.0)),
            tau_g=float(alg_cfg.get("tau_g", 0.5)),
            tau_n=float(alg_cfg.get("tau_n", 0.5)),
            bit_widths=tuple(int(b) for b in alg_cfg.get("bit_widths", DEFAULT_BIT_WIDTHS)),
        )


@dataclass(frozen=True)
class QuantPlan:
    """This round's per-client bit-width assignment plus the grad norms behind it.

    ``grad_norms`` are post-EMA-smoothing (if enabled) — the same values used to
    compute ``client_q`` — so callers reporting grad-norm telemetry and callers
    reading assigned q are looking at one consistent round snapshot.
    """

    client_q: dict[str, int]
    grad_norms: list[float]


def _snap_floor(value: float, bit_widths: tuple[int, ...]) -> int:
    """Snap ``value`` down to the largest permissible bit-width <= ``value``."""
    eligible = [b for b in bit_widths if b <= value]
    return max(eligible) if eligible else min(bit_widths)


def compute_fedmaq_q_k_t(
    c_k: float,
    c_unit: float,
    g_k: float,
    g_max: float,
    n_k: int,
    n_max: int,
    formulation: int,
    q_min: int,
    q_max: int,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    lambda_val: float = 1.0,
    tau_g: float = 0.5,
    tau_n: float = 0.5,
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS,
) -> int:
    """Compute client-specific quantization bit-width for FedMAQ.

    The final result is always a member of ``bit_widths`` (manuscript §4.2's
    permissible set Q), not an arbitrary continuous integer.
    """
    # Normalized signals
    tilde_g = g_k / g_max if g_max > 0.0 else 0.0
    tilde_n = n_k / n_max if n_max > 0.0 else 0.0

    # Tier 1 hard cap: Q_max = floor(c_k / c_unit), kept raw (unsnapped) so it can
    # be combined with the raw Tier-2 target below before a single floor-into-Q.
    q_k_max_raw = max(1.0, np.floor(c_k / c_unit))

    # Tier 2 soft quality target based on the formulation
    q_hat: float
    if formulation == 0:
        # Alternative 0: Resource-Only hard cap — no soft quality signal.
        # The soft target is always q_max; only Tier-1 constrains the final value.
        q_hat = q_max
    elif formulation == 1:
        # Alternative 1: Linear Sum
        term = gamma1 * tilde_g + gamma2 * tilde_n
        q_hat = q_min + np.round((q_max - q_min) * term)
    elif formulation == 2:
        # Alternative 2: Multiplicative
        term = (tilde_g**gamma1) * (tilde_n**gamma2)
        q_hat = q_min + np.round((q_max - q_min) * term)
    elif formulation == 3:
        # Alternative 3: Gradient-Primary, Data-Modulated
        modulator = (1.0 + lambda_val * tilde_n) / (1.0 + lambda_val)
        q_hat = q_min + np.round((q_max - q_min) * tilde_g * modulator)
    elif formulation == 4:
        # Alternative 4: Threshold-Based Staged Rule
        q_mid = int(np.round((q_max + q_min) / 2.0))
        if tilde_g >= tau_g and tilde_n >= tau_n:
            q_hat = q_max
        elif tilde_g >= tau_g or tilde_n >= tau_n:
            q_hat = q_mid
        else:
            q_hat = q_min
    else:
        q_hat = q_min

    # Clamp intermediate result to the configured [q_min, q_max] soft-target range.
    q_hat = max(float(q_min), min(float(q_max), float(q_hat)))
    # Combine raw Tier-1 cap and raw Tier-2 target via min(), then floor into the
    # permissible set Q exactly once: q_k^(t) = max{q in Q | q <= min(Q_k^max, q_hat_k^(t))}.
    # Memory-limited clients may receive fewer bits than q_min — intentional, the
    # physical bound wins over the soft quality target.
    return _snap_floor(min(q_k_max_raw, q_hat), bit_widths)


def _default_probe(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    """One stochastic single-batch gradient-norm probe: forward+backward, L2 norm."""
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    return torch.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)
    ).item()


class QuantizationPlanner:
    """Owns cross-round quantization state and produces one :class:`QuantPlan` per round.

    ``probe`` is the injectable adapter for the gradient-norm signal — the real
    adapter (default) runs a forward+backward pass on a client's batch; tests
    supply a synthetic probe to drive :meth:`plan_round` without a dataset.
    """

    def __init__(
        self,
        alg_name: str,
        model_fn: Callable[[str, int], nn.Module],
        probe: GradNormProbe = _default_probe,
    ) -> None:
        self._alg_name = alg_name
        self._model_fn = model_fn
        self._probe = probe
        self._grad_norm_model: nn.Module | None = None
        self._grad_norm_ema: dict[int, float] = {}

    def plan_round(
        self,
        parameters: Parameters,
        client_pids: list[int],
        client_cids: list[str],
        client_indices_dict: dict[int, list[int]],
        client_memory: dict[int, float] | list[float],
        ctx: RunContext,
        qp_cfg: dict[str, Any],
    ) -> QuantPlan:
        """Probe -> EMA-smooth -> normalize -> assign, for one round's sampled clients."""
        qp = _QuantParams.from_cfg(qp_cfg)
        temp_model = self._ensure_grad_norm_model(parameters, ctx)

        grad_norms, dataset_sizes = self._probe_grad_norms(
            temp_model, client_pids, ctx, client_indices_dict
        )
        grad_norms = self._smooth_grad_norms(client_pids, grad_norms, qp_cfg)

        client_q = self._assign_quantization(
            client_cids, client_pids, grad_norms, dataset_sizes, client_memory, qp
        )
        return QuantPlan(client_q=client_q, grad_norms=grad_norms)

    def _ensure_grad_norm_model(self, parameters: Parameters, ctx: RunContext) -> nn.Module:
        """Lazily build + cache the grad-norm probe model, then load ``parameters``."""
        if self._grad_norm_model is None:
            self._grad_norm_model = self._model_fn(ctx.dataset_name, ctx.num_classes)
            self._grad_norm_model.to(ctx.device)
        temp_model = self._grad_norm_model
        set_model_parameters(temp_model, parameters_to_ndarrays(parameters))
        temp_model.eval()
        return temp_model

    def _probe_grad_norms(
        self,
        temp_model: nn.Module,
        client_pids: list[int],
        ctx: RunContext,
        client_indices_dict: dict[int, list[int]],
    ) -> tuple[list[float], list[int]]:
        """One stochastic single-batch gradient-norm probe per sampled client.

        Returns the per-client raw grad norms (floored at 1e-8) and dataset sizes,
        aligned with ``client_pids``.
        """
        from fedmaq.core.strategy_hooks._partition import partition_dataset_size

        grad_norms: list[float] = []
        dataset_sizes: list[int] = []
        for pid in client_pids:
            n_k = partition_dataset_size(client_indices_dict, pid)
            dataset_sizes.append(n_k)

            loader = get_client_loader(
                dataset_name=ctx.dataset_name,
                client_id=pid,
                client_indices_dict=client_indices_dict,
                batch_size=ctx.batch_size,
                train=True,
            )
            try:
                images, labels = next(iter(loader))
                images, labels = images.to(ctx.device), labels.to(ctx.device)
                norm = self._probe(temp_model, images, labels)
            except ValueError:
                # F6: a shape/count mismatch (raised by set_model_parameters) is a
                # config/architecture bug, not a transient batch fault — fail loud.
                raise
            except Exception as exc:
                logger.warning(
                    f"Error computing gradient norm for client partition {pid}: {exc}. "
                    "Defaulting to 1e-8."
                )
                norm = 1e-8

            grad_norms.append(max(1e-8, norm))

        return grad_norms, dataset_sizes

    def _smooth_grad_norms(
        self,
        client_pids: list[int],
        grad_norms: list[float],
        alg_cfg: dict[str, Any],
    ) -> list[float]:
        """EMA-smooth per-client grad norms across rounds (Priority 3), if enabled.

        Mutates ``self._grad_norm_ema``. When ``grad_norm_ema`` is off, returns the
        raw norms unchanged.
        """
        if not alg_cfg.get("grad_norm_ema", False):
            return grad_norms
        beta = float(alg_cfg.get("grad_norm_beta", 0.7))
        smoothed_norms = []
        for pid, raw_norm in zip(client_pids, grad_norms, strict=True):
            if pid in self._grad_norm_ema:
                smoothed = beta * self._grad_norm_ema[pid] + (1.0 - beta) * raw_norm
            else:
                smoothed = raw_norm
            self._grad_norm_ema[pid] = smoothed
            smoothed_norms.append(smoothed)
        return smoothed_norms

    def _assign_quantization(
        self,
        client_cids: list[str],
        client_pids: list[int],
        grad_norms: list[float],
        dataset_sizes: list[int],
        client_memory: dict[int, float] | list[float],
        qp: _QuantParams,
    ) -> dict[str, int]:
        """Normalize the signals and compute each client's bit-width ``q``."""
        g_max = max(grad_norms) if grad_norms else 1e-8
        n_max = max(dataset_sizes) if dataset_sizes else 1

        client_q: dict[str, int] = {}
        for cid, pid, g_k, n_k in zip(
            client_cids, client_pids, grad_norms, dataset_sizes, strict=True
        ):
            c_k = float(client_memory[pid])
            q_k_t = compute_fedmaq_q_k_t(
                c_k=c_k,
                c_unit=qp.c_unit,
                g_k=g_k,
                g_max=g_max,
                n_k=n_k,
                n_max=n_max,
                formulation=qp.formulation,
                q_min=qp.q_min,
                q_max=qp.q_max,
                gamma1=qp.gamma1,
                gamma2=qp.gamma2,
                lambda_val=qp.lambda_val,
                tau_g=qp.tau_g,
                tau_n=qp.tau_n,
                bit_widths=qp.bit_widths,
            )
            client_q[cid] = q_k_t
            logger.info(
                f"FedMAQ - Client {cid} (partition {pid}): "
                f"c_k={c_k:.1f}MB, g_k={g_k:.4f} (tilde_g={g_k / g_max:.4f}), "
                f"n_k={n_k} (tilde_n={n_k / n_max:.4f}) -> "
                f"Final assigned q: {q_k_t}"
            )
        return client_q


def inject_client_q(
    client_instructions: list[tuple[ClientProxy, FitIns]],
    client_q: dict[str, int],
) -> list[tuple[ClientProxy, FitIns]]:
    """Rewrite each client's ``FitIns`` with its assigned ``q``, via a fresh instance.

    Instantiates new ``FitIns`` to prevent shared-reference overwrites — config
    dicts on the original instructions must not be mutated in place, since Flower
    may hold other references to them.
    """
    updated: list[tuple[ClientProxy, FitIns]] = []
    for client, fit_ins in client_instructions:
        new_fit_ins = FitIns(fit_ins.parameters, dict(fit_ins.config))
        new_fit_ins.config["q"] = client_q[client.cid]
        updated.append((client, new_fit_ins))
    return updated
