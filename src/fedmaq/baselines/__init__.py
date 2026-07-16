"""SOTA FL baseline implementations (FedAvg, FedProx, DAdaQuant, ...).

Factory
-------
``get_compressor_hook`` is the canonical way to select a client-side
:class:`~fedmaq.core.client.CompressionHook` for a given algorithm name.
To add a new baseline with a custom compressor, register it in
``_COMPRESSOR_HOOKS`` below (constructor takes ``(alg_cfg, rng)``).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from flwr.app import RecordDict

from fedmaq.baselines.compression import (
    FedKDCompressionHook,
    compress_tensor,
    decompress_tensor,
)
from fedmaq.baselines.postprocess import FedMAQPostProcessCompressionHook
from fedmaq.baselines.quantization import (
    DAdaQuantCompressionHook,
    FedPAQCompressionHook,
)
from fedmaq.core.client import CompressionHook

__all__ = [
    "FedKDCompressionHook",
    "FedMAQPostProcessCompressionHook",
    "compress_tensor",
    "decompress_tensor",
    "DAdaQuantCompressionHook",
    "FedPAQCompressionHook",
    "get_compressor_hook",
]

# Algorithm name -> compressor hook constructor, each taking
# (alg_cfg, rng, state) where alg_cfg is cfg.algorithm as a plain dict, rng is
# a seeded NumPy generator (stochastic rounding reproducibility for
# DAdaQuant/FedMAQ) or None for a default unseeded generator, and state is the
# per-client persistent RecordDict (only used by FedMAQ's post-processing hook).
_COMPRESSOR_HOOKS: dict[
    str,
    Callable[
        [dict[str, Any], np.random.Generator | None, RecordDict | None],
        CompressionHook,
    ],
] = {
    "fedpaq": lambda alg_cfg, rng, state: FedPAQCompressionHook(q=int(alg_cfg.get("q", 8))),
    "dadaquant": lambda alg_cfg, rng, state: DAdaQuantCompressionHook(
        q=int(alg_cfg.get("q_min", 1)),
        rng=rng,
    ),
    # FedMAQ and FedMAQ-Lite's q is a true bit-width from the manuscript's discrete set
    # {1,...,8,16,32} (see compute_fedmaq_q_k_t), so they must use FedPAQ's
    # bit-width-faithful symmetric quantizer.
    # Dispatch below overrides this with FedMAQPostProcessCompressionHook when
    # ``alg_cfg["post_process"]`` is true (primary benchmarking grid only).
    "fedmaq": lambda alg_cfg, rng, state: FedPAQCompressionHook(q=int(alg_cfg.get("q_min", 2))),
    "fedmaq_lite": lambda alg_cfg, rng, state: FedPAQCompressionHook(
        q=int(alg_cfg.get("q_min", 2))
    ),
    "fedkd": lambda alg_cfg, rng, state: FedKDCompressionHook(
        energy=float(alg_cfg.get("tmin", 0.5)),
        min_rank_frac=float(alg_cfg.get("min_rank_frac", 0.0)),
    ),
}


def get_compressor_hook(
    alg_name: str,
    alg_cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
    state: RecordDict | None = None,
) -> CompressionHook:
    """Factory: return the appropriate :class:`CompressionHook` for ``alg_name``.

    Parameters
    ----------
    alg_name:
        Algorithm identifier from the Hydra config (``cfg.algorithm.name``).
    alg_cfg:
        Algorithm sub-config dict (``cfg.algorithm`` as a plain dict).
    rng:
        Seeded NumPy generator for stochastic rounding reproducibility
        (DAdaQuant / FedMAQ).  If None a default unseeded generator is used.
    state:
        Per-client persistent :class:`flwr.app.RecordDict` (``Context.state``).
        Only consumed when dispatching to :class:`FedMAQPostProcessCompressionHook`
        (``alg_name in {"fedmaq", "fedmaq_lite"}`` and ``alg_cfg["post_process"]`` is true);
        ignored otherwise.

    Returns
    -------
    CompressionHook
        An identity hook for algorithms without client-side compression
        (fedavg, fedprox, fedmd, fedavg_kd, feddistill: uncompressed float32).
    """
    if alg_name in {"fedmaq", "fedmaq_lite"} and alg_cfg.get("post_process"):
        return FedMAQPostProcessCompressionHook(q=int(alg_cfg.get("q_min", 2)), state=state)
    hook_ctor = _COMPRESSOR_HOOKS.get(alg_name)
    if hook_ctor is None:
        return CompressionHook()
    return hook_ctor(alg_cfg, rng, state)
