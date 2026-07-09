"""Abstract base class for client-side fit strategies.

Each concrete :class:`ClientFitStrategy` encapsulates one algorithm's local
training/prediction procedure, keeping :class:`~fedmaq.core.client.GenericClient`
free of ``if alg_name == ...`` dispatch. The ``fit``/``evaluate`` methods receive
the live ``GenericClient`` so they can read its model, data loaders, hooks, and
config without duplicating that state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from fedmaq.core.models import set_model_parameters

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient


class ClientFitStrategy(ABC):
    """Per-algorithm local ``fit``/``evaluate`` procedure for ``GenericClient``."""

    @abstractmethod
    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Run one round of local training; return (params, num_examples, metrics)."""
        ...

    def evaluate(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the (incoming) global model on the client's test set."""
        return standard_evaluate(client, parameters, config)


def standard_evaluate(
    client: GenericClient,
    parameters: list[np.ndarray],
    config: dict[str, Any],
    load_parameters: bool = True,
) -> tuple[float, int, dict[str, Any]]:
    """Shared test-set evaluation loop (top-1 accuracy + cross-entropy loss)."""
    if load_parameters:
        set_model_parameters(client.model, parameters)

    client.model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in client.testloader:
            images, labels = images.to(client.device), labels.to(client.device)
            outputs = client.model(images)
            loss_sum += criterion(outputs, labels).item() * len(labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = loss_sum / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return float(loss), total, {"accuracy": float(accuracy)}
