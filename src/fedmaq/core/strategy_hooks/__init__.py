"""Strategy hook registry — maps algorithm names to StrategyHook instances.

To add a new baseline:
1. Implement a ``StrategyHook`` subclass in its own module (e.g. ``fednew.py``).
2. Add an ``elif alg_name == "fednew"`` branch in ``get_strategy_hook`` below.
3. Register any client-side hooks in ``baselines/__init__.py:get_compressor_hook``.
4. Update ``HANDOFF.md`` and ``baseline_registry.md``.
"""

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


def get_strategy_hook(alg_name: str, config: dict[str, Any]) -> StrategyHook:
    """Return the appropriate :class:`StrategyHook` for ``alg_name``.

    Unknown algorithm names fall back to :class:`PassthroughHook` so that
    new FedAvg-family algorithms (that only customise the client side) require
    no server-side hook registration.
    """
    if alg_name == "fedmaq":
        return FedMAQHook(config)
    elif alg_name == "dadaquant":
        return DAdaQuantHook(config)
    elif alg_name == "fedkd":
        return FedKDHook(config)
    elif alg_name == "fedmd":
        return FedMDHook(config)
    elif alg_name == "fedavg_kd":
        return FedAvgKDHook(config)
    elif alg_name == "feddistill":
        return FedDistillHook(config)
    elif alg_name == "cfd":
        return CFDHook(config)
    else:
        return PassthroughHook()
