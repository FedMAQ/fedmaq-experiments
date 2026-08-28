"""Coverage for the typed upload-byte report introduced by issue #49."""

from __future__ import annotations

import numpy as np
import pytest

import fedmaq.baselines as baselines
from fedmaq.baselines.compression import FedKDCompressionHook
from fedmaq.baselines.postprocess import FedMAQPostProcessCompressionHook
from fedmaq.baselines.quantization import DAdaQuantCompressionHook, FedPAQCompressionHook
from fedmaq.baselines.transport import UploadReport, measure_bytes
from fedmaq.core.client import CompressionHook


def _concrete_subclasses(cls: type[CompressionHook]) -> set[type[CompressionHook]]:
    subclasses: set[type[CompressionHook]] = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(_concrete_subclasses(subclass))
    return subclasses


def _load_compressor_modules() -> None:
    """Load every baseline module before walking the source-derived hierarchy."""
    import importlib
    import pkgutil

    for module in pkgutil.walk_packages(baselines.__path__, f"{baselines.__name__}."):
        importlib.import_module(module.name)


_load_compressor_modules()

_ISSUE46_COMPRESSORS = {
    FedPAQCompressionHook,
    DAdaQuantCompressionHook,
    FedKDCompressionHook,
    FedMAQPostProcessCompressionHook,
}


def test_compressor_subclass_enumeration_matches_issue46_tripwire():
    """The four concrete subclasses plus the base identity hook match #46."""
    assert _concrete_subclasses(CompressionHook) == _ISSUE46_COMPRESSORS


def _make_hook(hook_cls: type[CompressionHook]) -> CompressionHook:
    if hook_cls is CompressionHook:
        return hook_cls()
    if hook_cls is FedPAQCompressionHook:
        return hook_cls(q=4, rng=np.random.default_rng(0))
    if hook_cls is DAdaQuantCompressionHook:
        return hook_cls(q=4, rng=np.random.default_rng(0))
    if hook_cls is FedKDCompressionHook:
        return hook_cls(energy=0.9)
    if hook_cls is FedMAQPostProcessCompressionHook:
        return hook_cls(q=4, rng=np.random.default_rng(0))
    raise AssertionError(f"No coverage fixture for compressor subclass {hook_cls.__name__}")


@pytest.mark.parametrize(
    "hook_cls",
    [CompressionHook, *_concrete_subclasses(CompressionHook)],
    ids=lambda hook_cls: hook_cls.__name__,
)
def test_every_compressor_subclass_returns_a_well_formed_upload_report(
    hook_cls: type[CompressionHook],
):
    """Walk the class hierarchy so an override outside the registry cannot be missed."""
    hook = _make_hook(hook_cls)
    _, report = hook.compress(
        [
            np.array([-2.0, 0.0, 2.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        ]
    )

    assert isinstance(report, UploadReport)
    assert report.measured_bytes > 0
    assert report.payload_bytes > 0
    assert len(report.payloads) == 2
    assert sum(measure_bytes(payload) for payload in report.payloads) == report.measured_bytes
    if isinstance(hook, DAdaQuantCompressionHook):
        assert report.secondary_bytes is not None
    else:
        assert report.secondary_bytes is None
