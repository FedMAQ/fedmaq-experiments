"""FedMAQ quantization planning."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from flwr.common import FitIns, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.config_defaults import RunContext
from fedmaq.core.models import set_model_parameters
from fedmaq.core.partitioning import get_client_loader
from fedmaq.core.randomness import derive_seed

logger = logging.getLogger(__name__)

DEFAULT_BIT_WIDTHS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 16)

GradNormProbe = Callable[[nn.Module, torch.Tensor, torch.Tensor], float]

Formulation = int | str
PowerMeanDegree = float | str

POWER_MEAN_FORMULATION = "power_mean"
MINIMUM_POWER_MEAN = "min"

FORMULATION_CONSTANTS: dict[Formulation, tuple[str, ...]] = {
    0: (),
    1: ("gamma1", "gamma2"),
    2: ("gamma1", "gamma2"),
    3: ("kappa",),
    4: ("tau_g", "tau_n"),
    POWER_MEAN_FORMULATION: ("p", "omega"),
}
SUPPORTED_FORMULATIONS = tuple(FORMULATION_CONSTANTS)


@dataclass(frozen=True)
class _QuantParams:
    """Parsed FedMAQ quantization hyperparameters for one plan_round call.

    Grouping these keeps the q-assignment helper's signature small and, via
    ``from_cfg``, preserves the F8 fail-loud contract: the algorithm-defining
    knobs (``q_min``/``q_max``/``c_unit``/``formulation``) are read with no
    default so a missing/renamed key raises up front, before any probe work.
    The Tier-2 constants extend the same contract conditionally, via
    :data:`FORMULATION_CONSTANTS` — only the ones the selected formulation
    consumes are required, since every fedmaq config carries all five while any
    given run reads at most two.
    """

    q_min: int
    q_max: int
    c_unit: float
    formulation: Formulation
    gamma1: float
    gamma2: float
    kappa: float
    tau_g: float
    tau_n: float
    p: PowerMeanDegree
    omega: float
    bit_widths: tuple[int, ...]
    resource_aware: bool

    @classmethod
    def from_cfg(cls, alg_cfg: dict[str, Any]) -> _QuantParams:
        raw_formulation = alg_cfg["formulation"]
        formulation: Formulation
        if raw_formulation == POWER_MEAN_FORMULATION:
            formulation = POWER_MEAN_FORMULATION
        else:
            formulation = int(raw_formulation)
        if formulation not in FORMULATION_CONSTANTS:
            raise ValueError(
                f"algorithm.formulation={formulation} is not one of "
                f"{SUPPORTED_FORMULATIONS}; refusing to plan a round."
            )
        required = FORMULATION_CONSTANTS[formulation]

        def constant(key: str, default: float) -> float:
            if key in required:
                return float(alg_cfg[key])
            return float(alg_cfg.get(key, default))

        return cls(
            q_min=int(alg_cfg["q_min"]),
            q_max=int(alg_cfg["q_max"]),
            c_unit=float(alg_cfg["c_unit"]),
            formulation=formulation,
            gamma1=constant("gamma1", 0.5),
            gamma2=constant("gamma2", 0.5),
            kappa=constant("kappa", 1.0),
            tau_g=constant("tau_g", 0.5),
            tau_n=constant("tau_n", 0.5),
            p=_parse_power_mean_degree(alg_cfg["p"] if "p" in required else alg_cfg.get("p", 0.0)),
            omega=_parse_power_mean_weight(
                alg_cfg["omega"] if "omega" in required else alg_cfg.get("omega", 0.5)
            ),
            bit_widths=tuple(int(b) for b in alg_cfg.get("bit_widths", DEFAULT_BIT_WIDTHS)),
            resource_aware=bool(alg_cfg.get("resource_aware", True)),
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
    client_q_max: dict[str, float] = field(default_factory=dict)
    client_q_hat: dict[str, float] = field(default_factory=dict)
    client_grad_norms: dict[str, float] = field(default_factory=dict)
    tier1_enabled: bool = True
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS


@dataclass(frozen=True)
class QuantizationDecision:
    """Auditable result of one client's Tier-1/Tier-2 precision decision."""

    q_k_max: float
    q_hat: float
    q: int


def _snap_floor(value: float, bit_widths: tuple[int, ...]) -> int:
    """Snap ``value`` down to the largest permissible bit-width <= ``value``."""
    eligible = [b for b in bit_widths if b <= value]
    return max(eligible) if eligible else min(bit_widths)


def _parse_power_mean_degree(value: Any) -> PowerMeanDegree:
    """Return an exact supported power-mean degree or fail before planning."""
    if value == MINIMUM_POWER_MEAN:
        return MINIMUM_POWER_MEAN
    try:
        degree = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"algorithm.p={value!r} must be finite or {MINIMUM_POWER_MEAN!r}") from exc
    if not math.isfinite(degree):
        raise ValueError(f"algorithm.p={value!r} must be finite or {MINIMUM_POWER_MEAN!r}")
    return degree


def _parse_power_mean_weight(value: Any) -> float:
    """Return a valid state-signal weight for the power-mean family."""
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"algorithm.omega={value!r} must lie in [0, 1]") from exc
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"algorithm.omega={value!r} must lie in [0, 1]")
    return weight


def _weighted_power_mean(tilde_g: float, tilde_n: float, p: PowerMeanDegree, omega: float) -> float:
    """Evaluate the chosen power-mean degree with exact named limits."""
    if omega == 0.0:
        return tilde_n
    if omega == 1.0:
        return tilde_g
    if p == MINIMUM_POWER_MEAN:
        return min(tilde_g, tilde_n)

    assert isinstance(p, float)
    if p == 0.0:
        return (tilde_g**omega) * (tilde_n ** (1.0 - omega))
    if p < 0.0 and (tilde_g == 0.0 or tilde_n == 0.0):
        return 0.0
    return (omega * tilde_g**p + (1.0 - omega) * tilde_n**p) ** (1.0 / p)


def compute_fedmaq_q_k_t_details(
    c_k: float,
    c_unit: float,
    g_k: float,
    g_max: float,
    n_k: int,
    n_max: int,
    formulation: Formulation,
    q_min: int,
    q_max: int,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    kappa: float = 1.0,
    tau_g: float = 0.5,
    tau_n: float = 0.5,
    p: PowerMeanDegree = 0.0,
    omega: float = 0.5,
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS,
    resource_aware: bool = True,
) -> QuantizationDecision:
    """Compute client-specific quantization bit-width for FedMAQ.

    The final result is always a member of ``bit_widths`` (manuscript §3.3.3's
    permissible set Q), not an arbitrary continuous integer.

    ``resource_aware=False`` lifts the Tier-1 memory ceiling so the Tier-2 soft
    target alone governs the assignment. This is Ablation Configuration 2
    (manuscript §4.3.7), which places FedMAQ in the memory-blind condition the
    reproducible baselines occupy. Modeled client capacity c_k is a synthetic
    simulation state (modeled with Raspberry Pi 5 as a client compute reference)
    governing the Tier-1 quantization ceiling q_{k,max} = floor(c_k / c_unit),
    strictly distinct from simulator host hardware RAM / host VRAM (such as NVIDIA
    L40S simulation host references).
    """
    tilde_g = g_k / g_max if g_max > 0.0 else 0.0
    tilde_n = n_k / n_max if n_max > 0.0 else 0.0

    q_k_max_raw = max(1.0, np.floor(c_k / c_unit))

    q_hat: float
    if formulation == 0:
        q_hat = q_max
    elif formulation == 1:
        term = gamma1 * tilde_g + gamma2 * tilde_n
        q_hat = q_min + np.round((q_max - q_min) * term)
    elif formulation == 2:
        term = (tilde_g**gamma1) * (tilde_n**gamma2)
        q_hat = q_min + np.round((q_max - q_min) * term)
    elif formulation == 3:
        modulator = (1.0 + kappa * tilde_n) / (1.0 + kappa)
        q_hat = q_min + np.round((q_max - q_min) * tilde_g * modulator)
    elif formulation == 4:
        q_mid = int(np.round((q_max + q_min) / 2.0))
        if tilde_g >= tau_g and tilde_n >= tau_n:
            q_hat = q_max
        elif tilde_g >= tau_g or tilde_n >= tau_n:
            q_hat = q_mid
        else:
            q_hat = q_min
    elif formulation == POWER_MEAN_FORMULATION:
        term = _weighted_power_mean(
            tilde_g,
            tilde_n,
            _parse_power_mean_degree(p),
            _parse_power_mean_weight(omega),
        )
        q_hat = q_min + np.round((q_max - q_min) * term)
    else:
        raise ValueError(f"formulation={formulation} is not one of {SUPPORTED_FORMULATIONS}")

    q_hat = max(float(q_min), min(float(q_max), float(q_hat)))
    combined = q_hat if not resource_aware else min(q_k_max_raw, q_hat)
    return QuantizationDecision(
        q_k_max=float(q_k_max_raw),
        q_hat=float(q_hat),
        q=_snap_floor(combined, bit_widths),
    )


def compute_fedmaq_q_k_t(
    c_k: float,
    c_unit: float,
    g_k: float,
    g_max: float,
    n_k: int,
    n_max: int,
    formulation: Formulation,
    q_min: int,
    q_max: int,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    kappa: float = 1.0,
    tau_g: float = 0.5,
    tau_n: float = 0.5,
    p: PowerMeanDegree = 0.0,
    omega: float = 0.5,
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS,
    resource_aware: bool = True,
) -> int:
    """Compute only the final bit-width, preserving the original planner seam."""
    return compute_fedmaq_q_k_t_details(
        c_k=c_k,
        c_unit=c_unit,
        g_k=g_k,
        g_max=g_max,
        n_k=n_k,
        n_max=n_max,
        formulation=formulation,
        q_min=q_min,
        q_max=q_max,
        gamma1=gamma1,
        gamma2=gamma2,
        kappa=kappa,
        tau_g=tau_g,
        tau_n=tau_n,
        p=p,
        omega=omega,
        bit_widths=bit_widths,
        resource_aware=resource_aware,
    ).q


def _default_probe(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    """One stochastic single-batch gradient-norm probe: forward+backward, L2 norm."""
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    squared_norms = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
    squared_norm = torch.stack(squared_norms).sum() if squared_norms else torch.tensor(0.0)
    return float(torch.sqrt(squared_norm).item())


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
        client_indices_dict: dict[str, list[int]] | dict[int, list[int]] | None,
        client_memory: dict[int, float] | list[float] | np.ndarray | None,
        ctx: RunContext,
        qp_cfg: dict[str, Any],
        seed_base: int,
        server_round: int,
    ) -> QuantPlan:
        """Probe -> EMA-smooth -> normalize -> assign, for one round's sampled clients."""
        qp = _QuantParams.from_cfg(qp_cfg)
        temp_model = self._ensure_grad_norm_model(parameters, ctx)

        grad_norms, dataset_sizes = self._probe_grad_norms(
            temp_model, client_pids, ctx, client_indices_dict, seed_base, server_round
        )
        grad_norms = self._smooth_grad_norms(client_pids, grad_norms, qp_cfg)

        client_q, client_q_max, client_q_hat = self._assign_quantization(
            client_cids, client_pids, grad_norms, dataset_sizes, client_memory, qp
        )
        return QuantPlan(
            client_q=client_q,
            grad_norms=grad_norms,
            client_grad_norms=dict(zip(client_cids, grad_norms, strict=True)),
            client_q_max=client_q_max,
            client_q_hat=client_q_hat,
            tier1_enabled=qp.resource_aware,
            bit_widths=qp.bit_widths,
        )

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
        client_indices_dict: dict[str, list[int]] | dict[int, list[int]] | None,
        seed_base: int,
        server_round: int,
    ) -> tuple[list[float], list[int]]:
        """One stochastic single-batch gradient-norm probe per sampled client.

        Returns the per-client raw grad norms (floored at 1e-8) and dataset sizes,
        aligned with ``client_pids``.
        """
        from fedmaq.core.strategy_hooks._partition import partition_dataset_size

        if client_indices_dict is None and client_pids:
            raise ValueError("gradient-norm probing requires client partition indices")
        grad_norms: list[float] = []
        dataset_sizes: list[int] = []
        for pid in client_pids:
            n_k = partition_dataset_size(client_indices_dict, pid)
            dataset_sizes.append(n_k)

            loader = get_client_loader(
                dataset_name=ctx.dataset_name,
                client_id=pid,
                client_indices_dict={
                    str(key): value for key, value in (client_indices_dict or {}).items()
                },
                batch_size=ctx.batch_size,
                train=True,
                seed=derive_seed("probe_loader", seed_base, pid, max(1, server_round)),
            )
            try:
                images, labels = next(iter(loader))
                images, labels = images.to(ctx.device), labels.to(ctx.device)
                norm = self._probe(temp_model, images, labels)
            except ValueError:
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
        client_memory: dict[int, float] | list[float] | np.ndarray | None,
        qp: _QuantParams,
    ) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
        """Normalize the signals and compute each client's bit-width ``q``."""
        g_max = max(grad_norms) if grad_norms else 1e-8
        n_max = max(dataset_sizes) if dataset_sizes else 1

        client_q: dict[str, int] = {}
        client_q_max: dict[str, float] = {}
        client_q_hat: dict[str, float] = {}
        for cid, pid, g_k, n_k in zip(
            client_cids, client_pids, grad_norms, dataset_sizes, strict=True
        ):
            c_k = float(client_memory[pid]) if client_memory is not None else 0.0
            decision = compute_fedmaq_q_k_t_details(
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
                kappa=qp.kappa,
                tau_g=qp.tau_g,
                tau_n=qp.tau_n,
                p=qp.p,
                omega=qp.omega,
                bit_widths=qp.bit_widths,
                resource_aware=qp.resource_aware,
            )
            client_q[cid] = decision.q
            client_q_max[cid] = decision.q_k_max
            client_q_hat[cid] = decision.q_hat
            logger.info(
                f"FedMAQ - Client {cid} (partition {pid}): "
                f"c_k={c_k:.1f}MB, g_k={g_k:.4f} (tilde_g={g_k / g_max:.4f}), "
                f"n_k={n_k} (tilde_n={n_k / n_max:.4f}) -> "
                f"Final assigned q: {decision.q}"
            )
        return client_q, client_q_max, client_q_hat


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
