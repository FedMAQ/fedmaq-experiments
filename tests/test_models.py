"""Regression tests for the model factory and MobileNetV2GN width scaling.

Pins two invariants touched by the FedKD student switch (path B):
  * the full-model (width_mult=1.0) parameter count is unchanged, so every
    baseline that trains the full MobileNetV2GN is unaffected;
  * the CIFAR KD student is a genuinely smaller, still-depthwise-separable
    width-0.5 MobileNetV2GN (not the old SimpleCNN), and GroupNorm stays valid
    at the reduced width.
"""

from __future__ import annotations

import torch

from fedmaq.core.models import (
    MobileNetV2GN,
    get_kd_student_model,
    get_model,
)


def _nparams(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_full_model_param_count_unchanged() -> None:
    # The channel-rounding change must be a no-op at width_mult=1.0; these are
    # the counts every baseline trains on.
    assert _nparams(MobileNetV2GN(in_channels=3, num_classes=10)) == 2_236_682
    assert _nparams(MobileNetV2GN(in_channels=3, num_classes=100)) == 2_351_972


def test_cifar_kd_student_is_smaller_mobilenetv2gn() -> None:
    student = get_kd_student_model("cifar10", 10)
    full = get_model("cifar10", 10)
    assert isinstance(student, MobileNetV2GN)  # same depthwise-separable family
    ratio = _nparams(student) / _nparams(full)
    assert 0.2 < ratio < 0.4  # genuinely smaller (~0.26x), not ~1.0x like SimpleCNN


def test_width_half_student_forward_pass_and_groupnorm_valid() -> None:
    # Instantiation would raise if any GroupNorm channel count were indivisible;
    # exercise a forward pass to confirm the reduced-width graph runs.
    student = get_kd_student_model("cifar10", 10)
    out = student(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
