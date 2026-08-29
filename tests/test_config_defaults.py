"""Regression tests for the centralized fallback defaults (config_defaults.py).

These fallbacks fire only when a hook is built from a config that omits the key
(e.g. minimal/test configs); in real runs Hydra supplies every key, so a smoke
run cannot exercise these branches. These tests pin the fallback values so a
future edit to the constants can't silently change hook behavior on absent keys.
"""

from __future__ import annotations

import pytest

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


def test_require_num_public_samples_returns_configured_value() -> None:
    cfg = {"experiment": {"num_public_samples": 3000}}
    assert cd.require_num_public_samples(cfg) == 3000


@pytest.mark.parametrize("cfg", [{}, {"experiment": {}}])
def test_require_num_public_samples_fails_loud_on_missing_key(cfg: dict) -> None:
    # F12: no silent 200 fallback. A missing key must raise, not corrupt the
    # public-proxy pool size, since canonical |D_pub| is 3000.
    with pytest.raises(KeyError, match="num_public_samples"):
        cd.require_num_public_samples(cfg)


def test_run_context_resolves_nested_configuration_without_algorithm_state() -> None:
    context = cd.resolve_run_context(
        {
            "dataset": {"name": "cifar100", "num_classes": 100},
            "experiment": {
                "batch_size": 32,
                "num_public_samples": 3000,
                "server_compute_speed": 9000.0,
            },
            "device": "cpu",
            "seed": 7,
            "algorithm": {"name": "fedmaq", "q_min": 2},
        }
    )

    assert context.dataset_name == "cifar100"
    assert context.num_classes == 100
    assert context.batch_size == 32
    assert context.device.type == "cpu"
    assert context.seed == 7
    assert context.num_public_samples == 3000
    assert context.server_compute_speed == 9000.0
    assert context.algorithm_name == "fedmaq"
    assert not hasattr(context, "alg_cfg")


def test_run_context_accepts_flat_and_partially_specified_configuration() -> None:
    flat = cd.resolve_run_context(
        {
            "dataset_name": "femnist",
            "num_classes": 62,
            "batch_size": 16,
            "seed": 11,
            "algorithm_name": "fedprox",
        }
    )
    partial = cd.resolve_run_context({"dataset": {"name": "mnist"}})

    assert (flat.dataset_name, flat.num_classes, flat.batch_size) == ("femnist", 62, 16)
    assert flat.seed == 11
    assert flat.num_public_samples is None
    assert flat.algorithm_name == "fedprox"
    assert flat.server_compute_speed == cd.SERVER_COMPUTE_SPEED
    assert partial.dataset_name == "mnist"
    assert partial.num_classes == cd.NUM_CLASSES
    assert partial.batch_size == cd.BATCH_SIZE
    assert partial.device.type


def test_run_context_uses_explicit_defaults_for_missing_values() -> None:
    context = cd.resolve_run_context({})

    assert context.seed == 42
    assert context.server_compute_speed == cd.SERVER_COMPUTE_SPEED
    assert context.algorithm_name == ""
    assert context.num_public_samples is None


def test_algorithm_configuration_is_resolved_separately_from_run_context() -> None:
    nested = cd.resolve_algorithm_config({"algorithm": {"name": "fedprox", "mu": 0.2}})
    flat = cd.resolve_algorithm_config({"name": "fedprox", "mu": 0.3})

    assert nested == {"name": "fedprox", "mu": 0.2}
    assert flat == {"name": "fedprox", "mu": 0.3}
