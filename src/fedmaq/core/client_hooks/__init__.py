"""Client-side fit-strategy registry — maps algorithm names to ClientFitStrategy.

To add a new baseline with custom local training:
1. Implement a :class:`ClientFitStrategy` subclass in its own module.
2. Register it in ``_FIT_STRATEGIES`` below.
3. Algorithms that use the default FedAvg-style local loop need no entry — they
   fall back to :class:`StandardFit`.
"""

from __future__ import annotations

from fedmaq.core.client_hooks.base import ClientFitStrategy
from fedmaq.core.client_hooks.cfd import CFDFit
from fedmaq.core.client_hooks.feddistill import FedDistillFit
from fedmaq.core.client_hooks.fedkd import FedKDFit
from fedmaq.core.client_hooks.fedmd import FedMDFit
from fedmaq.core.client_hooks.standard import DAdaQuantFit, FedMAQFit, StandardFit

__all__ = [
    "ClientFitStrategy",
    "StandardFit",
    "DAdaQuantFit",
    "FedMAQFit",
    "FedMDFit",
    "FedKDFit",
    "FedDistillFit",
    "CFDFit",
    "get_fit_strategy",
]

# Algorithms absent from this map use StandardFit (the default FedAvg local loop).
_FIT_STRATEGIES: dict[str, type[ClientFitStrategy]] = {
    "fedmd": FedMDFit,
    "fedkd": FedKDFit,
    "dadaquant": DAdaQuantFit,
    "fedmaq": FedMAQFit,
    "feddistill": FedDistillFit,
    "cfd": CFDFit,
}


def get_fit_strategy(alg_name: str) -> ClientFitStrategy:
    """Return the :class:`ClientFitStrategy` for ``alg_name`` (StandardFit if absent)."""
    return _FIT_STRATEGIES.get(alg_name, StandardFit)()
