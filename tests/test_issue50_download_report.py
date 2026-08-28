"""Coverage for the typed download-byte report introduced by issue #50."""

from __future__ import annotations

import numpy as np
import pytest

from fedmaq.baselines.transport import UploadReport, measure_bytes
from fedmaq.core.strategy_hooks import (
    CFDHook,
    DAdaQuantHook,
    FedAvgKDHook,
    FedDistillHook,
    FedKDHook,
    FedMAQHook,
    FedMDHook,
    PassthroughHook,
    StrategyHook,
)


def _concrete_subclasses(cls: type[StrategyHook]) -> set[type[StrategyHook]]:
    subclasses: set[type[StrategyHook]] = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(_concrete_subclasses(subclass))
    return subclasses


_ISSUE46_STRATEGY_HOOKS = {
    CFDHook,
    DAdaQuantHook,
    FedAvgKDHook,
    FedDistillHook,
    FedKDHook,
    FedMAQHook,
    FedMDHook,
    PassthroughHook,
}


def test_strategy_hook_subclass_enumeration_matches_issue46_tripwire():
    """Every concrete strategy hook is included, including inherited paths."""
    assert _concrete_subclasses(StrategyHook) == _ISSUE46_STRATEGY_HOOKS


def _make_hook(hook_cls: type[StrategyHook]) -> StrategyHook:
    if hook_cls is CFDHook:
        return hook_cls({"dataset": {"name": "mnist", "num_classes": 4}})
    if hook_cls is FedKDHook:
        return hook_cls({"algorithm": {}})
    if hook_cls is FedDistillHook:
        return hook_cls({"dataset": {"num_classes": 4}})
    if hook_cls is FedMDHook:
        return hook_cls({})
    if hook_cls is DAdaQuantHook:
        return hook_cls({})
    if hook_cls is FedAvgKDHook:
        return hook_cls({})
    if hook_cls is FedMAQHook:
        return hook_cls({})
    if hook_cls is PassthroughHook:
        return hook_cls()
    raise AssertionError(f"No coverage fixture for strategy hook {hook_cls.__name__}")


@pytest.mark.parametrize(
    "hook_cls",
    sorted(_concrete_subclasses(StrategyHook), key=lambda cls: cls.__name__),
    ids=lambda hook_cls: hook_cls.__name__,
)
def test_every_strategy_hook_returns_a_well_formed_download_report(
    hook_cls: type[StrategyHook],
):
    """Walk the class hierarchy so an override outside the registry is covered."""
    hook = _make_hook(hook_cls)
    report = hook.download_size_bytes(
        None,
        [
            np.array([-2.0, 0.0, 2.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        ],
    )

    assert isinstance(report, UploadReport)
    assert report.secondary_bytes is None
    if hook_cls is CFDHook:
        assert report.measured_bytes == 0
        assert report.payload_bytes == 0
        assert report.payloads == ()
        return

    assert report.measured_bytes > 0
    assert report.payload_bytes > 0
    assert report.payload_bytes == sum(len(payload) for payload in report.payloads)
    assert report.measured_bytes == sum(measure_bytes(payload) for payload in report.payloads)
    assert len(report.payloads) == 2


def test_cfd_download_report_preserves_its_independent_published_coder():
    """CFD reports its own coder's bytes without pretending to use measure_bytes."""
    hook = CFDHook({"dataset": {"name": "mnist", "num_classes": 4}})
    hook._last_downstream_bytes = 37

    report = hook.download_size_bytes(None, [])

    assert report == UploadReport(
        measured_bytes=37,
        payload_bytes=37,
        secondary_bytes=None,
        payloads=(),
    )
