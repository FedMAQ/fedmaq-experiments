"""Generic Flower Client implementation with customizable hooks for loss and compression."""

from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedmaq.core.models import (
    DEVICE,
    get_kd_teacher_model,
    get_model_parameters,
    set_model_parameters,
)


class LossHook:
    """Base class for customizing local training loss functions."""

    def on_train_begin(self, model: nn.Module) -> None:
        """Hook called before the first training batch."""
        pass

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute the local loss."""
        return criterion(outputs, targets)


class FedProxLossHook(LossHook):
    """Loss hook adding proximal L2 regularization for FedProx."""

    def __init__(self, mu: float = 0.01) -> None:
        self.mu = mu
        self.global_params: list[torch.Tensor] = []

    def on_train_begin(self, model: nn.Module) -> None:
        # Save a frozen copy of the initial global weights
        self.global_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        loss = criterion(outputs, targets)
        proximal_term = 0.0
        params = [p for p in model.parameters() if p.requires_grad]
        for p, gp in zip(params, self.global_params):
            proximal_term += torch.sum((p - gp) ** 2)
        return loss + (self.mu / 2.0) * proximal_term


class CompressionHook:
    """Base class for compressing client model updates (deltas)."""

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas and return (compressed_deltas, byte_size)."""
        # Default: Identity (uncompressed Float32 weights -> 4 bytes per element)
        byte_size = sum(d.nbytes for d in deltas)
        return deltas, byte_size


class GenericClient(fl.client.NumPyClient):
    """Extensible client wrapping a PyTorch model and executing local epochs."""

    def __init__(
        self,
        cid: str,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_hook: LossHook,
        compressor_hook: CompressionHook,
        config: dict[str, Any],
        public_loader: torch.utils.data.DataLoader | None = None,
    ) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.loss_hook = loss_hook
        self.compressor_hook = compressor_hook
        self.config = config
        self.public_loader = public_loader
        self.device = DEVICE
        self.model.to(self.device)

    def get_properties(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"cid": self.cid}

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        alg_name = self.config.get("algorithm", {}).get("name", "")

        if alg_name == "fedmd":
            persistence_dir = self.config.get("experiment", {}).get(
                "persistence_dir", ".data_partitions/fedmd_models"
            )
            model_dir = Path(persistence_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"client_{self.cid}.pth"

            # 1. Load weights if file exists, else pre-train
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Run pre-training (Transfer Learning Phase)
                alg_cfg = self.config.get("algorithm", {})
                pub_pretrain_epochs = int(alg_cfg.get("public_pretrain_epochs", 10))
                priv_pretrain_epochs = int(alg_cfg.get("private_pretrain_epochs", 10))
                exp_config = self.config.get("experiment", self.config)
                lr = float(config.get("lr", exp_config.get("learning_rate", 0.01)))
                weight_decay = float(exp_config.get("weight_decay", 0.0))

                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=0.5,
                )
                criterion = nn.CrossEntropyLoss()

                # a. Pre-train on public dataset
                if self.public_loader is not None:
                    self.model.train()
                    for epoch in range(pub_pretrain_epochs):
                        for images, labels in self.public_loader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            optimizer.zero_grad()
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                # b. Pre-train on private dataset
                self.model.train()
                for epoch in range(priv_pretrain_epochs):
                    for images, labels in self.trainloader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                # Save initial weights
                torch.save(self.model.state_dict(), model_path)

            # 2. Check if we received predictions (soft targets) from the server
            # (if server_round > 1)
            server_round = config.get("server_round", 1)
            if server_round > 1 and len(parameters) == 1 and self.public_loader is not None:
                avg_predictions = parameters[0]
                alg_cfg = self.config.get("algorithm", {})
                public_epochs = int(alg_cfg.get("public_epochs", 5))
                exp_config = self.config.get("experiment", self.config)
                lr = float(config.get("lr", exp_config.get("learning_rate", 0.01)))
                weight_decay = float(exp_config.get("weight_decay", 0.0))

                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=0.5,
                )
                l1_criterion = nn.L1Loss()

                # Digest Phase: L1 loss against public soft targets
                self.model.train()
                for epoch in range(public_epochs):
                    start_idx = 0
                    for images, _ in self.public_loader:
                        images = images.to(self.device)
                        batch_len = len(images)
                        batch_targets = torch.tensor(
                            avg_predictions[start_idx : start_idx + batch_len],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        start_idx += batch_len

                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = l1_criterion(outputs, batch_targets)
                        loss.backward()
                        optimizer.step()

                # Revisit Phase: cross entropy loss on private dataset
                private_epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
                ce_criterion = nn.CrossEntropyLoss()
                self.model.train()
                for epoch in range(private_epochs):
                    for images, labels in self.trainloader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = ce_criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                # Save updated weights
                torch.save(self.model.state_dict(), model_path)

            # 3. Compute predictions on public dataset to send back to server
            predictions = []
            if self.public_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for images, _ in self.public_loader:
                        images = images.to(self.device)
                        outputs = self.model(images)
                        predictions.append(outputs.cpu().numpy())
                predictions = np.concatenate(predictions, axis=0)
            else:
                num_classes = self.config.get("dataset", {}).get("num_classes", 10)
                predictions = np.zeros((1, num_classes), dtype=np.float32)

            byte_size = predictions.nbytes
            return (
                [predictions],
                len(self.trainloader.dataset),
                {
                    "bytes_uploaded": byte_size,
                    "partition_id": int(self.cid),
                    "local_loss": 0.0,
                },
            )

        elif alg_name == "fedkd":
            persistence_dir = self.config.get("experiment", {}).get(
                "persistence_dir", ".data_partitions/fedkd_models"
            )
            model_dir = Path(persistence_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            teacher_path = model_dir / f"teacher_{self.cid}.pth"

            # 1. Instantiate teacher model based on dataset
            dataset_name = self.config.get("dataset", {}).get("name", "")
            num_classes = int(self.config.get("dataset", {}).get("num_classes", 10))
            teacher_model = get_kd_teacher_model(dataset_name, num_classes)
            teacher_model.to(self.device)

            # 2. Load teacher weights if file exists, otherwise keep random initialization
            if teacher_path.exists():
                teacher_model.load_state_dict(torch.load(teacher_path, map_location=self.device))

            # 3. Load global student parameters
            set_model_parameters(self.model, parameters)

            # 4. Setup Joint Optimizer
            exp_config = self.config.get("experiment", self.config)
            lr = float(config.get("lr", exp_config.get("learning_rate", 0.01)))
            weight_decay = float(exp_config.get("weight_decay", 0.0))
            epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))

            optimizer = torch.optim.SGD(
                list(self.model.parameters()) + list(teacher_model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            ce_criterion = nn.CrossEntropyLoss()
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            temperature = float(self.config.get("algorithm", {}).get("temperature", 2.0))

            # 5. Local Training: Student-Teacher Mutual Distillation
            self.model.train()
            teacher_model.train()
            for epoch in range(epochs):
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    # Forward pass
                    outputs_s = self.model(images)
                    outputs_t = teacher_model(images)

                    # Task Loss
                    loss_s_task = ce_criterion(outputs_s, labels)
                    loss_t_task = ce_criterion(outputs_t, labels)

                    # Soft predictions for KL divergence
                    outputs_s_log_soft = F.log_softmax(outputs_s / temperature, dim=1)
                    outputs_t_log_soft = F.log_softmax(outputs_t / temperature, dim=1)
                    outputs_s_soft = F.softmax(outputs_s / temperature, dim=1)
                    outputs_t_soft = F.softmax(outputs_t / temperature, dim=1)

                    # Mutual Knowledge Distillation Loss (scaled by temperature^2)
                    kl_t_to_s = kl_criterion(outputs_s_log_soft, outputs_t_soft) * (temperature**2)
                    kl_s_to_t = kl_criterion(outputs_t_log_soft, outputs_s_soft) * (temperature**2)

                    # Adaptive scaling: divide by sum of task losses
                    denom = loss_s_task + loss_t_task + 1e-6
                    loss_kd_s = kl_t_to_s / denom
                    loss_kd_t = kl_s_to_t / denom

                    # Joint optimization loss
                    loss_s = loss_s_task + loss_kd_s
                    loss_t = loss_t_task + loss_kd_t
                    total_loss = loss_s + loss_t

                    total_loss.backward()
                    optimizer.step()

            # 6. Save updated teacher model parameters
            torch.save(teacher_model.state_dict(), teacher_path)

            # 7. Extract updated student model parameters and compute delta
            updated_params = get_model_parameters(self.model)
            deltas = [u - o for u, o in zip(updated_params, parameters)]

            # 8. Update compressor hook with dynamic energy if provided in configuration
            if "energy" in config:
                if hasattr(self.compressor_hook, "energy"):
                    self.compressor_hook.energy = float(config["energy"])

            # 9. Compress updates
            compressed_deltas, byte_size = self.compressor_hook.compress(deltas)

            # Reconstruct parameter update: w_new_reconstructed = w_old + compressed_deltas
            reconstructed_params = [o + cd for o, cd in zip(parameters, compressed_deltas)]

            return (
                reconstructed_params,
                len(self.trainloader.dataset),
                {
                    "bytes_uploaded": byte_size,
                    "partition_id": int(self.cid),
                    "local_loss": 0.0,
                },
            )

        # Default FL path (FedAvg, FedProx, etc.)
        # Load incoming server weights
        set_model_parameters(self.model, parameters)

        # Update compressor hook with dynamic q if provided in configuration
        if "q" in config:
            if hasattr(self.compressor_hook, "q"):
                self.compressor_hook.q = int(config["q"])

        # Compute local loss before training if DAdaQuant is enabled
        local_loss = 0.0
        if alg_name == "dadaquant":
            self.model.eval()
            loss_sum = 0.0
            total_samples = 0
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss_sum += criterion(outputs, labels).item() * len(labels)
                    total_samples += len(labels)
            local_loss = loss_sum / total_samples if total_samples > 0 else 0.0

        # Retrieve round configurations (with defaults from config)
        exp_config = self.config.get("experiment", self.config)
        lr = float(config.get("lr", exp_config.get("learning_rate", 0.01)))
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
        weight_decay = float(exp_config.get("weight_decay", 0.0))

        # Setup training
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.loss_hook.on_train_begin(self.model)

        last_loss = 0.0
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_hook.compute_loss(self.model, outputs, labels, criterion)
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

        # Extract updated parameters
        updated_params = get_model_parameters(self.model)

        # Compute delta = w_new - w_old
        deltas = [u - o for u, o in zip(updated_params, parameters)]

        # Compress updates
        compressed_deltas, byte_size = self.compressor_hook.compress(deltas)

        # Reconstruct parameter update: w_new_reconstructed = w_old + compressed_deltas
        reconstructed_params = [o + cd for o, cd in zip(parameters, compressed_deltas)]

        return (
            reconstructed_params,
            len(self.trainloader.dataset),
            {
                "bytes_uploaded": byte_size,
                "partition_id": int(self.cid),
                "local_loss": float(last_loss) if alg_name == "fedmaq" else local_loss,
            },
        )

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        # Load weights
        alg_name = self.config.get("algorithm", {}).get("name", "")
        if alg_name == "fedmd":
            persistence_dir = self.config.get("experiment", {}).get(
                "persistence_dir", f".data_partitions/{alg_name}_models"
            )
            model_path = Path(persistence_dir) / f"client_{self.cid}.pth"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            set_model_parameters(self.model, parameters)

        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss_sum += criterion(outputs, labels).item() * len(labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return float(loss), total, {"accuracy": float(accuracy)}
