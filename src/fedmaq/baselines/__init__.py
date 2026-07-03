"""SOTA FL baseline implementations (FedAvg, FedProx, DAdaQuant, ...).

Factory
-------
``get_compressor_hook`` is the canonical way to select a client-side
:class:`~fedmaq.core.client.CompressionHook` for a given algorithm name.
When adding a new baseline with a custom compressor, add a branch here.
"""

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
        An identity hook for algorithms without client-side compression.
    """
    if alg_name == "fedpaq":
        return FedPAQCompressionHook(q=int(alg_cfg.get("q", 8)))
    elif alg_name in {"dadaquant", "fedmaq"}:
        return DAdaQuantCompressionHook(
            q=int(alg_cfg.get("q_min", 1)),
            rng=rng,
        )
    elif alg_name == "fedkd":
        return FedKDCompressionHook(energy=float(alg_cfg.get("tmin", 0.5)))
    # fedavg, fedprox, fedmd, fedavg_kd: identity (uncompressed float32)
    return CompressionHook()
