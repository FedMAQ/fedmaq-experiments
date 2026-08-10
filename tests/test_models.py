"""Regression tests for model factories and MobileNetV2GN scaling."""

from __future__ import annotations

import torch

from fedmaq.core.models import (
    MobileNetV2GN,
    SimpleCNN,
    get_client_model,
    get_kd_student_model,
    get_model,
    get_server_model_factory,
)


def _nparams(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_full_model_param_count_unchanged() -> None:
    assert _nparams(MobileNetV2GN(in_channels=3, num_classes=10)) == 2_236_682
    assert _nparams(MobileNetV2GN(in_channels=3, num_classes=100)) == 2_351_972


def test_cifar_kd_student_is_smaller_mobilenetv2gn() -> None:
    student = get_kd_student_model("cifar10", 10)
    full = get_model("cifar10", 10)
    assert isinstance(student, MobileNetV2GN)
    ratio = _nparams(student) / _nparams(full)
    assert 0.2 < ratio < 0.4


def test_width_half_student_forward_pass_and_groupnorm_valid() -> None:
    student = get_kd_student_model("cifar10", 10)
    out = student(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_fedkd_holds_the_only_compact_student() -> None:
    assert isinstance(get_client_model("fedkd", "cifar10", 10), MobileNetV2GN)
    assert isinstance(get_client_model("fedmaq", "cifar10", 10), MobileNetV2GN)
    server_factory = get_server_model_factory("fedmaq")
    assert isinstance(server_factory("cifar10", 10), MobileNetV2GN)
    assert isinstance(get_client_model("fedmaq", "femnist", 62), SimpleCNN)
