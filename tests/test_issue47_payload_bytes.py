from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import Code, Status, ndarrays_to_parameters
from flwr.common.typing import FitRes
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core.client import CompressionHook, GenericClient, LossHook
from fedmaq.core.client_hooks.fedmd import FedMDFit
from fedmaq.core.models import SimpleCNN
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager


def _loader(num_samples: int = 4, num_classes: int = 4) -> DataLoader:
    data = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(data, labels), batch_size=2)


def test_fedmd_reports_raw_prediction_bytes_as_payload_bytes(tmp_path):
    loader = _loader()
    config = {
        "experiment": {
            "local_epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "persistence_dir": str(tmp_path),
        },
        "algorithm": {
            "name": "fedmd",
            "public_pretrain_epochs": 1,
            "private_pretrain_epochs": 1,
            "public_epochs": 1,
        },
        "dataset": {"name": "mnist", "num_classes": 4},
    }
    client = GenericClient(
        cid="0",
        trainloader=loader,
        testloader=loader,
        model=SimpleCNN(in_channels=1, num_classes=4),
        loss_hook=LossHook(),
        compressor_hook=CompressionHook(),
        config=config,
        public_loader=loader,
    )

    _, _, metrics = FedMDFit().fit(client, [], {"server_round": 1})

    expected_payload_bytes = np.zeros((len(loader.dataset), 4), dtype=np.float32).nbytes
    assert metrics["bytes_uploaded"] == expected_payload_bytes
    assert metrics["payload_bytes"] == expected_payload_bytes


class _Proxy:
    cid = "0"


def _fit_res(metrics: dict[str, int]) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=1,
        metrics=metrics,
    )


def _strategy(algorithm: str, monkeypatch) -> TelemetryFedAvg:
    monkeypatch.setattr("fedmaq.core.telemetry._HYDRA_AVAILABLE", False)
    config = {
        "experiment": {
            "num_clients": 1,
            "num_public_samples": 4,
            "telemetry": {"wandb_enabled": False},
        },
        "algorithm": {"name": algorithm},
        "dataset": {"name": "mnist", "num_classes": 4},
    }
    return TelemetryFedAvg(
        telemetry_manager=TelemetryManager(config),
        config=config,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
    )


@pytest.mark.parametrize("algorithm", ["fedmd", "cfd"])
def test_round_payload_bytes_preserve_special_arm_upload_totals(algorithm, monkeypatch):
    strategy = _strategy(algorithm, monkeypatch)
    fit_res = _fit_res(
        {"partition_id": 0, "bytes_uploaded": 123, "payload_bytes": 123}
    )

    strategy.telemetry_manager.record_fit_round(
        strategy, server_round=1, results=[(_Proxy(), fit_res)], aggregated_parameters=None
    )

    assert strategy.telemetry_manager.snapshot_for_round(1).round_payload_bytes == 123


def test_round_payload_bytes_requires_explicit_client_report(monkeypatch):
    strategy = _strategy("fedmd", monkeypatch)
    fit_res = _fit_res({"partition_id": 0, "bytes_uploaded": 123})

    with pytest.raises(KeyError, match="payload_bytes"):
        strategy.telemetry_manager.record_fit_round(
            strategy, server_round=1, results=[(_Proxy(), fit_res)], aggregated_parameters=None
        )
