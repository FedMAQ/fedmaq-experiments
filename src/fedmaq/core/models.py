"""Standard PyTorch model architectures and helpers for FedMAQ."""

from collections.abc import Callable

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
        self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int
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


class InvertedResidualGN(nn.Module):
    """MobileNetV2 inverted residual block with GroupNorm instead of BatchNorm.

    Follows the expand → depthwise → project bottleneck design from
    Sandler et al. (2018), replacing all BatchNorm layers with GroupNorm
    for compatibility with non-IID federated learning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers: list[nn.Module] = []
        # Pointwise expansion (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.GroupNorm(min(num_groups, hidden_dim), hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )
        # Depthwise convolution
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.GroupNorm(min(num_groups, hidden_dim), hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        )
        # Pointwise linear projection (no activation)
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2GN(nn.Module):
    """MobileNetV2 with GroupNorm instead of BatchNorm for federated learning.

    Standard MobileNetV2 architecture (Sandler et al., 2018) adapted for:
    - **GroupNorm**: Replaces BatchNorm to avoid batch-statistics divergence
      under non-IID client data in federated settings.
    - **CIFAR resolution**: Uses stride-1 initial convolution (vs. stride-2
      for ImageNet) to preserve spatial resolution at 32×32 input.

    ~2.3M parameters at width_mult=1.0, ~4.9× smaller than ResNet18GN (~11.17M).
    """

    # Standard MobileNetV2 inverted residual settings:
    # (expand_ratio, output_channels, num_blocks, stride)
    _INVERTED_RESIDUAL_SETTINGS: list[tuple[int, int, int, int]] = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # stride 1 for CIFAR 32×32 (ImageNet uses 2)
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        num_groups: int = 8,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        input_channels = max(int(32 * width_mult), 8)
        last_channels = max(int(1280 * width_mult), 8)

        # Initial convolution — stride 1 for CIFAR 32×32 (ImageNet uses stride 2)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, input_channels, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, input_channels), input_channels),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks
        for t, c, n, s in self._INVERTED_RESIDUAL_SETTINGS:
            output_channels = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidualGN(
                        input_channels, output_channels, stride, t, num_groups
                    )
                )
                input_channels = output_channels

        # Final convolution
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channels, last_channels, 1, bias=False),
                nn.GroupNorm(min(num_groups, last_channels), last_channels),
                nn.ReLU6(inplace=True),
            )
        )

        self.classifier = nn.Linear(last_channels, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [
        val.cpu().detach().numpy() for val in model.parameters() if val.requires_grad
    ]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load parameters (NumPy arrays) into the model.

    Raises ``ValueError`` on a length or per-tensor shape mismatch rather than
    silently truncating (which would happen with a plain ``zip``), so that loading
    an incompatible parameter list into the wrong architecture fails loudly.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model expects {len(params)} trainable tensors "
            f"but received {len(parameters)}. Likely a model-architecture mismatch."
        )
    for p, w in zip(params, parameters, strict=True):
        if tuple(p.shape) != tuple(w.shape):
            raise ValueError(
                f"Parameter shape mismatch: model tensor {tuple(p.shape)} vs "
                f"incoming {tuple(w.shape)}."
            )
        p.data = torch.from_numpy(w).to(p.device)


# Registry of CIFAR-compatible model constructors, keyed by lowercase name.
# Each value is a callable(in_channels, num_classes) -> nn.Module.
_CIFAR_MODELS: dict[str, type[nn.Module]] = {
    "mobilenetv2gn": MobileNetV2GN,
    "resnet18gn": ResNet18GN,
}

# Default CIFAR model architecture. Changed from ResNet18GN (~11.17M params) to
# MobileNetV2GN (~2.3M params) for edge-realistic federated learning experiments.
# See STATUS.md §2 for rationale.
DEFAULT_CIFAR_MODEL: str = "mobilenetv2gn"


def get_model(
    dataset_name: str, num_classes: int, model_name: str | None = None
) -> nn.Module:
    """Factory function to get appropriate model architecture for a dataset.

    Args:
        dataset_name: Dataset identifier (e.g. "cifar10", "mnist").
        num_classes: Number of output classes.
        model_name: Optional override for CIFAR model architecture.
            Supported: "mobilenetv2gn" (default), "resnet18gn".
            Ignored for MNIST/FEMNIST datasets (always SimpleCNN).
    """
    name_lower = dataset_name.lower()
    if name_lower in ["mnist", "fmnist", "femnist"]:
        return SimpleCNN(in_channels=1, num_classes=num_classes)
    elif name_lower in ["cifar10", "cifar-10", "cifar100", "cifar-100"]:
        key = (model_name or DEFAULT_CIFAR_MODEL).lower()
        model_cls = _CIFAR_MODELS.get(key)
        if model_cls is None:
            raise ValueError(
                f"Unknown CIFAR model '{model_name}'. Available: {list(_CIFAR_MODELS.keys())}"
            )
        return model_cls(in_channels=3, num_classes=num_classes)
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
        return MobileNetV2GN(in_channels=3, num_classes=num_classes)
    else:
        return SimpleCNN(in_channels=1, num_classes=num_classes)


def get_client_model(alg_name: str, dataset_name: str, num_classes: int) -> nn.Module:
    """Factory: return the appropriate local model for a given algorithm.

    FedKD and FedMAQ-Lite use a smaller student model (TinyCNN / SimpleCNN) for clients;
    all other algorithms (including FedMAQ) use the full standard model.
    """
    if alg_name in {"fedkd", "fedmaq_lite"}:
        return get_kd_student_model(dataset_name, num_classes)
    return get_model(dataset_name, num_classes)


def get_server_model_factory(alg_name: str) -> Callable[[str, int], nn.Module]:
    """Single source of truth for the ``(dataset_name, num_classes) -> nn.Module``
    factory the server uses for a given algorithm's grad-norm probe and
    self-distillation student/teacher.

    FedMAQ-Lite operates on the KD student architecture (SimpleCNN/TinyCNN);
    FedMAQ (and any other server-KD caller) uses the standard model. ``FedMAQHook``
    must call this rather than re-deriving the choice inline — previously the two
    inline copies keyed on ``fedmaq_lite`` only while :func:`get_client_model`
    keyed on ``{fedkd, fedmaq_lite}``, a latent divergence. It is behavior-equivalent
    here because ``fedkd`` never reaches ``FedMAQHook`` (it routes to ``FedKDHook``);
    this factory intentionally scopes to its actual callers.
    """
    if alg_name == "fedmaq_lite":
        return get_kd_student_model
    return get_model
