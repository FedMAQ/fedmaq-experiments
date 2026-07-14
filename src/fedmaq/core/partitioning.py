"""Dataset loading and deterministic partitioning with server-side public reserve.

Supported partition modes:
- ``dirichlet``: Artificial non-IID skew via Dirichlet(alpha) distribution. Used for
  CIFAR-10 and CIFAR-100.
- ``writer``: Writer-based natural partitioning for FEMNIST. EMNIST byclass preserves
  writer locality within each class via natural ordering; we approximate writer
  partitions by dividing each class's samples into ``num_clients`` equal chunks along
  the natural ordering axis, yielding non-IID distributions without artificial skewing.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST

# Base paths
DATA_DIR = Path("data").resolve()
CACHE_DIR = Path(".data_partitions").resolve()

# Standard normalizations for torchvision datasets
TRANSFORMS = {
    "mnist": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "fmnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    "femnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    "cifar10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "cifar100": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2761)),
        ]
    ),
}


@lru_cache(maxsize=8)
def _load_dataset_cached(dataset_name: str, train: bool) -> Dataset:
    """Cached internal loader — call ``load_dataset`` instead.

    The cache avoids repeated torchvision disk scans when the same dataset is
    requested multiple times (e.g. once per sampled client in FedMAQ gradient
    norm computation).  ``maxsize=8`` covers four datasets × two splits.

    .. note::
        Test monkeypatching must target ``load_dataset`` (the public function),
        not this cached helper, to avoid stale cache entries across tests.
    """
    return load_dataset(dataset_name, train)


def load_dataset(dataset_name: str, train: bool = True) -> Dataset:
    """Download and return torchvision dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    name_lower = dataset_name.lower()

    if name_lower == "mnist":
        return MNIST(DATA_DIR, train=train, download=True, transform=TRANSFORMS["mnist"])
    elif name_lower == "fmnist":
        return FashionMNIST(DATA_DIR, train=train, download=True, transform=TRANSFORMS["fmnist"])
    elif name_lower == "femnist":
        # FEMNIST is approximated via EMNIST with 'byclass' split.
        # Writer locality is preserved in the natural ordering within each class.
        return EMNIST(
            DATA_DIR,
            split="byclass",
            train=train,
            download=True,
            transform=TRANSFORMS["femnist"],
        )
    elif name_lower in ["cifar10", "cifar-10"]:
        return CIFAR10(DATA_DIR, train=train, download=True, transform=TRANSFORMS["cifar10"])
    elif name_lower in ["cifar100", "cifar-100"]:
        return CIFAR100(DATA_DIR, train=train, download=True, transform=TRANSFORMS["cifar100"])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_labels(dataset: Dataset) -> np.ndarray:
    """Extract labels array from a PyTorch Dataset."""
    if hasattr(dataset, "targets"):
        # CIFAR datasets and MNIST-like targets list/tensor
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        # Fallback loop (slow, but safe for generic wrappers)
        targets = [dataset[i][1] for i in range(len(dataset))]

    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy()
    return np.array(targets)


def _generate_dirichlet_partition(
    class_indices: dict[int, np.ndarray],
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """Partition using Dirichlet distribution for statistical heterogeneity."""
    client_indices: dict[str, list[int]] = {str(k): [] for k in range(num_clients)}
    for remaining_idx in class_indices.values():
        rng.shuffle(remaining_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(remaining_idx)).astype(int)
        # Fix rounding error
        diff = len(remaining_idx) - proportions.sum()
        for _ in range(diff):
            proportions[rng.integers(num_clients)] += 1
        start = 0
        for k in range(num_clients):
            end = start + proportions[k]
            client_indices[str(k)].extend(remaining_idx[start:end].tolist())
            start = end
    return client_indices


def _generate_writer_partition(
    class_indices: dict[int, np.ndarray],
    num_clients: int,
) -> dict[str, list[int]]:
    """Writer-based natural partition for FEMNIST.

    EMNIST byclass preserves writer locality within each class via natural ordering.
    We approximate writer partitions by dividing each class's samples into
    ``num_clients`` equal chunks along the natural ordering axis, yielding non-IID
    distributions without artificial Dirichlet skewing.

    This approximation is calibrated against LEAF FEMNIST by setting
    ``num_clients`` to approximately the number of real writers (default: 200).
    """
    client_indices: dict[str, list[int]] = {str(k): [] for k in range(num_clients)}
    for remaining_idx in class_indices.values():
        # Preserve natural ordering — writer locality is encoded in EMNIST's structure
        chunks = np.array_split(remaining_idx, num_clients)
        for k, chunk in enumerate(chunks):
            client_indices[str(k)].extend(chunk.tolist())
    return client_indices


def generate_partition_indices(
    dataset_name: str,
    num_clients: int,
    alpha: float = 1.0,
    num_public_samples: int = 200,
    seed: int = 42,
    partition: str = "dirichlet",
) -> tuple[list[int], dict[str, list[int]]]:
    """Generate or retrieve cached partition indices (deterministic).

    Args:
        dataset_name: Torchvision dataset identifier.
        num_clients: Number of federated clients / simulated writers.
        alpha: Dirichlet concentration parameter (unused for ``partition="writer"``).
        num_public_samples: Samples reserved for the server-side public proxy pool.
        seed: RNG seed for reproducibility.
        partition: ``"dirichlet"`` (default) or ``"writer"`` (FEMNIST natural partition).

    Returns:
        public_indices: Indices reserved for server-side public pool.
        client_indices: Map of str(client_id) -> list of sample indices.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if partition == "writer":
        cache_name = (
            f"{dataset_name.lower()}_clients_{num_clients}_"
            f"writer_pub_{num_public_samples}_seed_{seed}.json"
        )
    else:
        cache_name = (
            f"{dataset_name.lower()}_clients_{num_clients}_"
            f"alpha_{alpha}_pub_{num_public_samples}_seed_{seed}.json"
        )
    cache_file = CACHE_DIR / cache_name

    if cache_file.is_file():
        with open(cache_file) as f:
            cache_data = json.load(f)
        return cache_data["public_indices"], cache_data["client_indices"]

    # Generate indices
    dataset = load_dataset(dataset_name, train=True)
    labels = get_dataset_labels(dataset)
    num_classes = len(np.unique(labels))

    rng = np.random.default_rng(seed)

    # Step 1: Slice off server public pool (balanced across classes)
    public_indices: list[int] = []
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    # Distribute num_public_samples as evenly as possible across classes: base
    # count per class plus one extra for the first `remainder` classes, so the
    # pool totals exactly num_public_samples (when enough samples are available)
    # instead of silently dropping num_public_samples % num_classes samples.
    base_per_class = num_public_samples // num_classes
    remainder = num_public_samples % num_classes
    for c in range(num_classes):
        target = base_per_class + (1 if c < remainder else 0)
        n_available = len(class_indices[c])
        n_select = min(target, n_available)
        selected = rng.choice(class_indices[c], size=n_select, replace=False)
        public_indices.extend(selected.tolist())
        class_indices[c] = np.setdiff1d(class_indices[c], selected)

    # Top up from classes with remaining capacity if any class was short on
    # samples, so the pool still reaches num_public_samples when feasible.
    shortfall = num_public_samples - len(public_indices)
    if shortfall > 0:
        for c in range(num_classes):
            if shortfall <= 0:
                break
            n_available = len(class_indices[c])
            if n_available == 0:
                continue
            n_select = min(shortfall, n_available)
            selected = rng.choice(class_indices[c], size=n_select, replace=False)
            public_indices.extend(selected.tolist())
            class_indices[c] = np.setdiff1d(class_indices[c], selected)
            shortfall -= n_select

    # Step 2: Partition remaining data
    if partition == "writer":
        client_indices = _generate_writer_partition(class_indices, num_clients)
    else:
        client_indices = _generate_dirichlet_partition(class_indices, num_clients, alpha, rng)

    # Save to cache
    cache_data = {"public_indices": public_indices, "client_indices": client_indices}
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    return public_indices, client_indices


def get_client_loader(
    dataset_name: str,
    client_id: int,
    client_indices_dict: dict[str, list[int]],
    batch_size: int = 64,
    train: bool = True,
) -> torch.utils.data.DataLoader:
    """Return PyTorch DataLoader for a specific client partition."""
    dataset = _load_dataset_cached(dataset_name, train)
    indices = client_indices_dict[str(client_id)]
    client_subset = Subset(dataset, indices)
    return torch.utils.data.DataLoader(client_subset, batch_size=batch_size, shuffle=train)


def get_server_loaders(
    dataset_name: str, public_indices: list[int], batch_size: int = 64
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return public unlabeled server dataset loader and central evaluation test loader."""
    train_dataset = _load_dataset_cached(dataset_name, True)
    public_subset = Subset(train_dataset, public_indices)
    public_loader = torch.utils.data.DataLoader(public_subset, batch_size=batch_size, shuffle=False)

    test_dataset = _load_dataset_cached(dataset_name, False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return public_loader, test_loader
