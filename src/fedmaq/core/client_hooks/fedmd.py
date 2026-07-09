"""FedMD client-side fit strategy (transfer-learning pre-train + digest/revisit)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from fedmaq.core.client_hooks.base import ClientFitStrategy, standard_evaluate

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient


class FedMDFit(ClientFitStrategy):
    """FedMD: local models are persisted on disk and exchange public-set predictions.

    Round 1 pre-trains on public then private data; later rounds digest the server's
    averaged public predictions (L1) and revisit the private set (cross-entropy).
    Clients return their public-set predictions, not model weights.
    """

    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        persistence_dir = client.config.get("experiment", {}).get(
            "persistence_dir", ".data_partitions/fedmd_models"
        )
        model_dir = Path(persistence_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"client_{client.cid}.pth"

        # 1. Load weights if file exists, else pre-train
        if model_path.exists():
            client.model.load_state_dict(
                torch.load(model_path, map_location=client.device)
            )
        else:
            # Run pre-training (Transfer Learning Phase)
            alg_cfg = client.config.get("algorithm", {})
            pub_pretrain_epochs = int(alg_cfg.get("public_pretrain_epochs", 10))
            priv_pretrain_epochs = int(alg_cfg.get("private_pretrain_epochs", 10))
            exp_config = client.config.get("experiment", client.config)
            lr = client._get_decayed_lr(config)
            weight_decay = float(exp_config.get("weight_decay", 0.0))
            momentum = float(exp_config.get("momentum", alg_cfg.get("momentum", 0.9)))

            optimizer = torch.optim.SGD(
                client.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
            criterion = nn.CrossEntropyLoss()

            # a. Pre-train on public dataset
            if client.public_loader is not None:
                client.model.train()
                for _ in range(pub_pretrain_epochs):
                    for images, labels in client.public_loader:
                        images, labels = (
                            images.to(client.device),
                            labels.to(client.device),
                        )
                        optimizer.zero_grad()
                        outputs = client.model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

            # b. Pre-train on private dataset
            client.model.train()
            for _ in range(priv_pretrain_epochs):
                for images, labels in client.trainloader:
                    images, labels = (
                        images.to(client.device),
                        labels.to(client.device),
                    )
                    optimizer.zero_grad()
                    outputs = client.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Save initial weights
            torch.save(client.model.state_dict(), model_path)

        # 2. Check if we received predictions (soft targets) from the server
        # (if server_round > 1)
        server_round = config.get("server_round", 1)
        if (
            server_round > 1
            and len(parameters) == 1
            and client.public_loader is not None
        ):
            avg_predictions = parameters[0]
            alg_cfg = client.config.get("algorithm", {})
            public_epochs = int(alg_cfg.get("public_epochs", 5))
            exp_config = client.config.get("experiment", client.config)
            lr = client._get_decayed_lr(config)
            weight_decay = float(exp_config.get("weight_decay", 0.0))
            momentum = float(exp_config.get("momentum", alg_cfg.get("momentum", 0.9)))

            optimizer = torch.optim.SGD(
                client.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
            l1_criterion = nn.L1Loss()

            # Digest Phase: L1 loss against public soft targets
            client.model.train()
            for _ in range(public_epochs):
                start_idx = 0
                for images, _ in client.public_loader:
                    images = images.to(client.device)
                    batch_len = len(images)
                    batch_targets = torch.tensor(
                        avg_predictions[start_idx : start_idx + batch_len],
                        dtype=torch.float32,
                        device=client.device,
                    )
                    start_idx += batch_len

                    optimizer.zero_grad()
                    outputs = client.model(images)
                    loss = l1_criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()

            # Revisit Phase: cross entropy loss on private dataset
            private_epochs = int(
                config.get("epochs", exp_config.get("local_epochs", 5))
            )
            ce_criterion = nn.CrossEntropyLoss()
            client.model.train()
            for _ in range(private_epochs):
                for images, labels in client.trainloader:
                    images, labels = (
                        images.to(client.device),
                        labels.to(client.device),
                    )
                    optimizer.zero_grad()
                    outputs = client.model(images)
                    loss = ce_criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Save updated weights
            torch.save(client.model.state_dict(), model_path)

        # 3. Compute predictions on public dataset to send back to server
        predictions = []
        if client.public_loader is not None:
            client.model.eval()
            with torch.no_grad():
                for images, _ in client.public_loader:
                    images = images.to(client.device)
                    outputs = client.model(images)
                    predictions.append(outputs.cpu().numpy())
            predictions = np.concatenate(predictions, axis=0)
        else:
            num_classes = client.config.get("dataset", {}).get("num_classes", 10)
            predictions = np.zeros((1, num_classes), dtype=np.float32)

        byte_size = predictions.nbytes
        return (
            [predictions],
            len(client.trainloader.dataset),
            {
                "bytes_uploaded": byte_size,
                "partition_id": int(client.cid),
                "local_loss": 0.0,
            },
        )

    def evaluate(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        persistence_dir = client.config.get("experiment", {}).get(
            "persistence_dir", ".data_partitions/fedmd_models"
        )
        model_path = Path(persistence_dir) / f"client_{client.cid}.pth"
        if model_path.exists():
            client.model.load_state_dict(
                torch.load(model_path, map_location=client.device)
            )
        return standard_evaluate(client, parameters, config, load_parameters=False)
