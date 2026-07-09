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

from fedmaq.baselines.compression import (
    FedKDCompressionHook,
    compress_tensor,
    decompress_tensor,
)
from fedmaq.baselines.quantization import (
    DAdaQuantCompressionHook,
    FedPAQCompressionHook,
)
from fedmaq.core.client import CompressionHook

__all__ = [
    "FedKDCompressionHook",
    "compress_tensor",
    "decompress_tensor",
    "DAdaQuantCompressionHook",
    "FedPAQCompressionHook",
    "get_compressor_hook",
]

# Algorithm name -> compressor hook constructor, each taking
# (alg_cfg, rng) where alg_cfg is cfg.algorithm as a plain dict and rng is
# a seeded NumPy generator (stochastic rounding reproducibility for
# DAdaQuant/FedMAQ) or None for a default unseeded generator.
_COMPRESSOR_HOOKS: dict[
    str, Callable[[dict[str, Any], np.random.Generator | None], CompressionHook]
] = {
    "fedpaq": lambda alg_cfg, rng: FedPAQCompressionHook(q=int(alg_cfg.get("q", 8))),
    "dadaquant": lambda alg_cfg, rng: DAdaQuantCompressionHook(
        q=int(alg_cfg.get("q_min", 1)),
        rng=rng,
    ),
    # FedMAQ's q is a true bit-width from the manuscript's discrete set
    # {1,...,8,16,32} (see compute_fedmaq_q_k_t), so it must use FedPAQ's
    # bit-width-faithful symmetric quantizer, not DAdaQuant's levels-per-sign
    # semantics (which would badly misinterpret e.g. q=16 as 33 levels).
    "fedmaq": lambda alg_cfg, rng: FedPAQCompressionHook(
        q=int(alg_cfg.get("q_min", 2))
    ),
    "fedkd": lambda alg_cfg, rng: FedKDCompressionHook(
        energy=float(alg_cfg.get("tmin", 0.5))
    ),
}


def get_compressor_hook(
    alg_name: str,
    alg_cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
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

    Returns
    -------
    CompressionHook
        An identity hook for algorithms without client-side compression
        (fedavg, fedprox, fedmd, fedavg_kd, feddistill: uncompressed float32).
    """
    hook_ctor = _COMPRESSOR_HOOKS.get(alg_name)
    if hook_ctor is None:
        return CompressionHook()
    return hook_ctor(alg_cfg, rng)
