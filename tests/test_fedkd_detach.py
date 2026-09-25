"""Regression tests for the FedKD KL-target detach switch (ADR-0028)."""

from __future__ import annotations

import torch

from fedmaq.core.client_hooks.fedkd import mutual_kl_terms


def _underflowing_logits() -> tuple[torch.Tensor, torch.Tensor]:
    # A 200-logit gap drives one softmax probability to exactly 0 in float32,
    # the state that diverged the first-study FedKD runs.
    student = torch.tensor([[0.0, -200.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.5, -200.0]], requires_grad=True)
    return student, teacher


def _grads(detach: bool) -> tuple[torch.Tensor, torch.Tensor]:
    student, teacher = _underflowing_logits()
    kl_t_to_s, kl_s_to_t = mutual_kl_terms(student, teacher, 1.0, detach_targets=detach)
    (kl_t_to_s + kl_s_to_t).backward()
    return student.grad, teacher.grad


def test_zero_probability_target_gives_nan_gradient_without_detach():
    student_grad, teacher_grad = _grads(detach=False)
    assert torch.isnan(student_grad).any() or torch.isnan(teacher_grad).any()


def test_zero_probability_target_gives_finite_gradient_with_detach():
    student_grad, teacher_grad = _grads(detach=True)
    assert torch.isfinite(student_grad).all()
    assert torch.isfinite(teacher_grad).all()


def test_detach_leaves_the_loss_value_unchanged():
    torch.manual_seed(0)
    student = torch.randn(4, 10)
    teacher = torch.randn(4, 10)
    for temperature in (1.0, 2.0):
        off = mutual_kl_terms(student, teacher, temperature, detach_targets=False)
        on = mutual_kl_terms(student, teacher, temperature, detach_targets=True)
        assert torch.equal(off[0], on[0])
        assert torch.equal(off[1], on[1])


def test_switch_defaults_off():
    student, teacher = _underflowing_logits()
    kl_t_to_s, kl_s_to_t = mutual_kl_terms(student, teacher, 1.0)
    (kl_t_to_s + kl_s_to_t).backward()
    assert torch.isnan(student.grad).any() or torch.isnan(teacher.grad).any()
