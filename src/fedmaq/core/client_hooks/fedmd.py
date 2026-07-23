"""FedMD client-side fit strategy (transfer-learning pre-train + digest/revisit)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from fedmaq.core.client_hooks.base import ClientFitStrategy, standard_evaluate
from fedmaq.core.client_hooks.training_skeleton import StepResult, run_epochs

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

        loss_sum = 0.0
        correct = 0
        total_samples = 0
        batches = 0
        epochs_trained = 0

        # 1. Load weights if file exists, else pre-train
        if model_path.exists():
            client.model.load_state_dict(torch.load(model_path, map_location=client.device))
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

            def no_metrics_step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
                outputs = client.model(images)
                return StepResult(loss=criterion(outputs, labels))

            # a. Pre-train on public dataset
            if client.public_loader is not None:
                run_epochs(
                    model=client.model,
                    loader=client.public_loader,
                    optimizer=optimizer,
                    epochs=pub_pretrain_epochs,
                    step_fn=no_metrics_step_fn,
                    device=client.device,
                )

            # b. Pre-train on private dataset. loss_sum/batches/correct/total_samples
            # are shared with the revisit phase below (same running accumulator as
            # the original code), so they're tracked manually here rather than via
            # run_epochs' own per-call averaging -- combining two already-divided
            # averages would reassociate the float sum and risk a non-bit-exact
            # result under Decision 40's golden-diff gate.
            def priv_pretrain_step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
                nonlocal loss_sum, batches, correct, total_samples
                outputs = client.model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()
                return StepResult(loss=loss)

            run_epochs(
                model=client.model,
                loader=client.trainloader,
                optimizer=optimizer,
                epochs=priv_pretrain_epochs,
                step_fn=priv_pretrain_step_fn,
                device=client.device,
            )
            epochs_trained += priv_pretrain_epochs

            # Save initial weights
            torch.save(client.model.state_dict(), model_path)

        # 2. Check if we received predictions (soft targets) from the server
        # (if server_round > 1)
        server_round = config.get("server_round", 1)
        if server_round > 1 and len(parameters) == 1 and client.public_loader is not None:
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

            # Digest Phase: L1 loss against public soft targets. The batch offset
            # into avg_predictions resets each epoch, so (like CFD's distill phase)
            # this is driven one epoch at a time rather than via a single
            # epochs=public_epochs call.
            def make_digest_step_fn(start_box: list[int]):
                def step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
                    batch_len = len(images)
                    batch_targets = torch.tensor(
                        avg_predictions[start_box[0] : start_box[0] + batch_len],
                        dtype=torch.float32,
                        device=client.device,
                    )
                    start_box[0] += batch_len
                    outputs = client.model(images)
                    return StepResult(loss=l1_criterion(outputs, batch_targets))

                return step_fn

            for _ in range(public_epochs):
                run_epochs(
                    model=client.model,
                    loader=client.public_loader,
                    optimizer=optimizer,
                    epochs=1,
                    step_fn=make_digest_step_fn([0]),
                    device=client.device,
                )

            # Revisit Phase: cross entropy loss on private dataset. Shares
            # loss_sum/batches/correct/total_samples with the priv-pretrain phase
            # above (see comment there re: manual accumulation for bit-exactness).
            private_epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
            ce_criterion = nn.CrossEntropyLoss()

            def revisit_step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
                nonlocal loss_sum, batches, correct, total_samples
                outputs = client.model(images)
                loss = ce_criterion(outputs, labels)
                loss_sum += loss.item()
                batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()
                return StepResult(loss=loss)

            run_epochs(
                model=client.model,
                loader=client.trainloader,
                optimizer=optimizer,
                epochs=private_epochs,
                step_fn=revisit_step_fn,
                device=client.device,
            )
            epochs_trained += private_epochs

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
        avg_train_loss = loss_sum / batches if batches > 0 else 0.0
        avg_train_acc = correct / total_samples if total_samples > 0 else 0.0

        return (
            [predictions],
            len(client.trainloader.dataset),
            {
                "bytes_uploaded": byte_size,
                "partition_id": int(client.cid),
                "local_loss": avg_train_loss,
                "train_loss": avg_train_loss,
                "train_acc": avg_train_acc,
                "epochs_trained": epochs_trained,
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
            client.model.load_state_dict(torch.load(model_path, map_location=client.device))
        return standard_evaluate(client, parameters, config, load_parameters=False)
