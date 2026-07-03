"""Standard PyTorch model architectures and helpers for FedMAQ."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DEVICE is evaluated once at import time based on the process's visible CUDA devices.
# In Flower's simulation, worker processes inherit the same CUDA environment, so this
# is consistent across the lifetime of a single simulation run.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    """LeNet-5 style Simple CNN for MNIST/FMNIST/FEMNIST/CIFAR10."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Determine flat features based on channel size:
        # 3 channels implies 32x32 CIFAR, 1 channel implies 28x28 MNIST
        flat_features = 64 * 8 * 8 if in_channels == 3 else 64 * 7 * 7
        self.fc1 = nn.Linear(flat_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyCNN(nn.Module):
    """A smaller CNN for MNIST/FMNIST/FEMNIST/CIFAR10 to act as student model for FedKD."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        # Determine flat features based on channel size:
        # 3 channels implies 32x32 CIFAR, 1 channel implies 28x28 MNIST
        flat_features = 32 * 8 * 8 if in_channels == 3 else 32 * 7 * 7
        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    """Basic Block for ResNet18 with Group Normalization."""

    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, num_groups: int = 32
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(min(num_groups, planes), planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(min(num_groups, planes), planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    min(num_groups, self.expansion * planes), self.expansion * planes
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18GN(nn.Module):
    """ResNet-18 architecture with Group Normalization instead of Batch Normalization."""

    def __init__(
        self, in_channels: int = 3, num_classes: int = 10, num_groups: int = 32
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        # Small 3x3 conv at the start (standard for CIFAR-10/100 32x32 resolution)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, 64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(
        self, block: Any, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [
        val.cpu().detach().numpy() for val in model.parameters() if val.requires_grad
    ]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load parameters (NumPy arrays) into the model."""
    params = [p for p in model.parameters() if p.requires_grad]
    for p, w in zip(params, parameters):
        p.data = torch.from_numpy(w).to(p.device)


def get_model(dataset_name: str, num_classes: int) -> nn.Module:
    """Factory function to get appropriate model architecture for a dataset."""
    name_lower = dataset_name.lower()
    if name_lower in ["mnist", "fmnist", "femnist"]:
        in_channels = 1
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif name_lower in ["cifar10", "cifar-10", "cifar100", "cifar-100"]:
        in_channels = 3
        return ResNet18GN(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported dataset for model selection: {dataset_name}")


def get_kd_student_model(dataset_name: str, num_classes: int) -> nn.Module:
    """Retrieve student model for knowledge distillation baselines (FedKD/FedMAQ)."""
    dataset_name_lower = dataset_name.lower()
    if "cifar" in dataset_name_lower:
        return SimpleCNN(in_channels=3, num_classes=num_classes)
    else:
        return TinyCNN(in_channels=1, num_classes=num_classes)


def get_kd_teacher_model(dataset_name: str, num_classes: int) -> nn.Module:
    """Retrieve teacher model for knowledge distillation baselines (FedKD/FedMAQ)."""
    dataset_name_lower = dataset_name.lower()
    if "cifar" in dataset_name_lower:
        return ResNet18GN(in_channels=3, num_classes=num_classes)
    else:
        return SimpleCNN(in_channels=1, num_classes=num_classes)


def get_client_model(alg_name: str, dataset_name: str, num_classes: int) -> nn.Module:
    """Factory: return the appropriate local model for a given algorithm.

    FedKD and FedMAQ use a smaller student model (TinyCNN / SimpleCNN) for clients;
    all other algorithms use the full standard model.
    """
    if alg_name in {"fedkd", "fedmaq"}:
        return get_kd_student_model(dataset_name, num_classes)
    return get_model(dataset_name, num_classes)
