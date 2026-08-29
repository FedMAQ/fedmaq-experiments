from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core.client import (
    CompressionHook,
    FedProxLossHook,
    GenericClient,
    LossHook,
    read_training_metrics,
)
from fedmaq.core.client_hooks.standard import StandardFit
from fedmaq.core.models import get_model_parameters


def test_base_loss_hook_has_no_optional_training_metrics() -> None:
    assert read_training_metrics(LossHook()) == {}


def test_fedprox_exposes_read_only_training_metrics() -> None:
    hook = FedProxLossHook()
    hook.last_ce = 1.25
    hook.last_prox = 0.75

    metrics = read_training_metrics(hook)

    assert metrics == {"f14_ce_loss": 1.25, "f14_prox_penalty": 0.75}
    try:
        metrics["new"] = 1.0  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("training metrics must be read-only")


def test_disabled_or_absent_capability_emits_no_metrics() -> None:
    class DisabledHook:
        def training_metrics(self):
            return None

    assert read_training_metrics(DisabledHook()) == {}
    assert read_training_metrics(SimpleNamespace()) == {}


def test_malformed_capability_is_ignored() -> None:
    class MalformedHook:
        def training_metrics(self):
            return {"f14_ce_loss": object()}

    assert read_training_metrics(MalformedHook()) == {}


def test_standard_fit_emits_capability_metrics_after_training() -> None:
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.tensor([0, 1])),
        batch_size=2,
    )
    model = torch.nn.Linear(2, 2)
    client = GenericClient(
        cid="0",
        trainloader=loader,
        testloader=loader,
        model=model,
        loss_hook=FedProxLossHook(mu=0.1),
        compressor_hook=CompressionHook(),
        config={"seed": 0, "experiment": {"local_epochs": 1, "learning_rate": 0.01}},
    )

    _, _, metrics = StandardFit().fit(client, get_model_parameters(model), {"server_round": 1})

    assert metrics["f14_ce_loss"] > 0.0
    assert metrics["f14_prox_penalty"] == 0.0
    assert "f14_grad_norm" in metrics
    assert "f14_gn_affine_norm" in metrics
