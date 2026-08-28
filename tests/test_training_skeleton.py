from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core.client_hooks.training_skeleton import StepResult, run_epochs


class _RecordingModel(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.events = events

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.events.append("forward")
        return self.layer(images)


class _RecordingOptimizer(torch.optim.SGD):
    def __init__(self, parameters, events: list[str]) -> None:
        super().__init__(parameters, lr=0.1)
        self.events = events

    def zero_grad(self, *args, **kwargs) -> None:
        self.events.append("zero_grad")
        super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.events.append("step")
        return super().step(*args, **kwargs)


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(4, 1), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )


def test_run_epochs_preserves_batch_sequence_and_reports_final_loss() -> None:
    events: list[str] = []
    model = _RecordingModel(events)
    optimizer = _RecordingOptimizer(model.parameters(), events)
    losses = iter((1.0, 2.0, 3.0, 4.0))

    def step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
        del labels
        outputs = model(images)
        events.append("loss")
        loss = outputs.sum() * 0 + next(losses)
        loss.register_hook(lambda _gradient: events.append("backward"))
        return StepResult(loss=loss)

    def on_after_backward() -> None:
        events.append("after_backward")

    result = run_epochs(
        model=model,
        loader=_loader(),
        optimizer=optimizer,
        epochs=2,
        step_fn=step_fn,
        device="cpu",
        on_after_backward=on_after_backward,
    )

    expected_batch = ["zero_grad", "forward", "loss", "backward", "after_backward", "step"]
    assert events == expected_batch * 4
    assert result.batches == 4
    assert result.avg_loss == 2.5
    assert result.last_loss == 4.0
    assert result.accuracy is None


def test_recording_post_backward_callback_does_not_change_optimizer_step() -> None:
    torch.manual_seed(0)
    baseline = _RecordingModel([])
    observed = _RecordingModel([])
    observed.load_state_dict(copy.deepcopy(baseline.state_dict()))

    def step_fn(model: torch.nn.Module):
        def step(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
            del labels
            return StepResult(loss=model(images).square().mean())

        return step

    baseline_result = run_epochs(
        model=baseline,
        loader=_loader(),
        optimizer=torch.optim.SGD(baseline.parameters(), lr=0.1),
        epochs=2,
        step_fn=step_fn(baseline),
        device="cpu",
    )
    observed_gradients: list[tuple[torch.Tensor, ...]] = []

    def observe() -> None:
        observed_gradients.append(
            tuple(parameter.grad.detach().clone() for parameter in observed.parameters())
        )

    observed_result = run_epochs(
        model=observed,
        loader=_loader(),
        optimizer=torch.optim.SGD(observed.parameters(), lr=0.1),
        epochs=2,
        step_fn=step_fn(observed),
        device="cpu",
        on_after_backward=observe,
    )

    assert len(observed_gradients) == 4
    assert observed_result == baseline_result
    for observed_parameter, baseline_parameter in zip(
        observed.parameters(), baseline.parameters(), strict=True
    ):
        torch.testing.assert_close(observed_parameter, baseline_parameter)
