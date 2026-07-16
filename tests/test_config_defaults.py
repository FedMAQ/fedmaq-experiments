"""Regression tests for the centralized fallback defaults (config_defaults.py).

These fallbacks fire only when a hook is built from a config that omits the key
(e.g. minimal/test configs); in real runs Hydra supplies every key, so a smoke
run cannot exercise these branches. These tests pin the fallback values so a
future edit to the constants can't silently change hook behavior on absent keys.
"""

from __future__ import annotations

from fedmaq.core import config_defaults as cd
from fedmaq.core.strategy_hooks.cfd import CFDHook


def test_constants_match_expected_literals() -> None:
    # Values these constants replaced across the KD strategy hooks. Changing any
    # of these silently changes fallback behavior everywhere they are used.
    assert cd.SERVER_COMPUTE_SPEED == 2000.0
    assert cd.BATCH_SIZE == 64
    assert cd.DATASET_NAME == "mnist"
    assert cd.NUM_CLASSES == 10


def test_cfdhook_uses_dataset_fallbacks_on_empty_config() -> None:
    # Empty config -> the dead fallback branch fires. This is the only hook that
    # reads the dataset defaults in __init__, so it exercises them directly.
    hook = CFDHook(config={})
    assert hook.dataset_name == cd.DATASET_NAME
    assert hook.num_classes == cd.NUM_CLASSES
