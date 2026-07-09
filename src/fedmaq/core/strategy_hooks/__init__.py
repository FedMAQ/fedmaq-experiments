"""Strategy hook registry — maps algorithm names to StrategyHook instances.

To add a new baseline:
1. Implement a ``StrategyHook`` subclass in its own module (e.g. ``fednew.py``).
2. Register its class in ``_STRATEGY_HOOKS`` below (constructor takes ``config``).
3. Register any client-side hooks in ``core/client_hooks`` and
   ``baselines/__init__.py:get_compressor_hook``.
4. Update ``HANDOFF.md`` and ``baseline_registry.md``.

Algorithms with no registered hook fall back to :class:`PassthroughHook` (the
FedAvg-family default) with a warning. Algorithms listed in ``_UNPORTED`` raise a
clear error at construction time rather than a confusing mid-round failure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fedmaq.core.strategy_hooks.base import StrategyHook
from fedmaq.core.strategy_hooks.cfd import CFDHook
from fedmaq.core.strategy_hooks.dadaquant import DAdaQuantHook
from fedmaq.core.strategy_hooks.fedavg_kd import FedAvgKDHook
from fedmaq.core.strategy_hooks.feddistill import FedDistillHook
from fedmaq.core.strategy_hooks.fedkd import FedKDHook
from fedmaq.core.strategy_hooks.fedmaq import FedMAQHook
from fedmaq.core.strategy_hooks.fedmd import FedMDHook
from fedmaq.core.strategy_hooks.passthrough import PassthroughHook

logger = logging.getLogger(__name__)

__all__ = [
    "StrategyHook",
    "PassthroughHook",
    "FedKDHook",
    "FedMAQHook",
    "DAdaQuantHook",
    "FedMDHook",
    "FedAvgKDHook",
    "FedDistillHook",
    "CFDHook",
    "get_strategy_hook",
]

# Algorithm name -> hook constructor (each takes the full config dict).
_STRATEGY_HOOKS: dict[str, Callable[[dict[str, Any]], StrategyHook]] = {
    "fedmaq": FedMAQHook,
    "dadaquant": DAdaQuantHook,
    "fedkd": FedKDHook,
    "fedmd": FedMDHook,
    "fedavg_kd": FedAvgKDHook,
}

# Registered configs whose hook is not yet implemented. Selecting one fails at
# construction time (clear message) instead of raising mid-round.
_UNPORTED: dict[str, str] = {
    "feddistill": "FedDistill (Task 10, ~Sep 2026)",
    "cfd": "CFD (Task 11, ~Oct 2026)",
}


def get_strategy_hook(alg_name: str, config: dict[str, Any]) -> StrategyHook:
    """Return the :class:`StrategyHook` for ``alg_name``.

    Unknown algorithm names fall back to :class:`PassthroughHook` (with a warning)
    so FedAvg-family algorithms that only customise the client side need no
    server-side registration. Unported algorithms raise ``NotImplementedError``.
    """
    if alg_name in _UNPORTED:
        raise NotImplementedError(
            f"Algorithm '{alg_name}' is registered but not yet implemented "
            f"({_UNPORTED[alg_name]}). Select a different algorithm."
        )
    hook_cls = _STRATEGY_HOOKS.get(alg_name)
    if hook_cls is None:
        logger.warning(
            f"No strategy hook registered for algorithm '{alg_name}'; falling back "
            "to PassthroughHook (FedAvg-family default). Verify algorithm.name."
        )
        return PassthroughHook()
    return hook_cls(config)
