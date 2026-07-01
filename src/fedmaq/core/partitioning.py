"""Dataset loading and deterministic Dirichlet partitioning with server-side public reserve."""

import json
import os
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


def load_dataset(dataset_name: str, train: bool = True) -> Dataset:
    """Download and return torchvision dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    name_lower = dataset_name.lower()

    if name_lower == "mnist":
        return MNIST(DATA_DIR, train=train, download=True, transform=TRANSFORMS["mnist"])
    elif name_lower == "fmnist":
        return FashionMNIST(DATA_DIR, train=train, download=True, transform=TRANSFORMS["fmnist"])
    elif name_lower == "femnist":
        # FEMNIST is approximated via EMNIST with 'byclass' split
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


def generate_partition_indices(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    num_public_samples: int = 500,
    seed: int = 42,
) -> tuple[list[int], dict[str, list[int]]]:
    """Generate or retrieve cached partition indices (deterministic).

    Returns:
        public_indices: Indices reserved for server-side public pool
        client_indices: Map of str(client_id) -> list of sample indices
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
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

    # Set seed for determinism
    rng = np.random.default_rng(seed)

    # Step 1: Slice off server public pool (Option A)
    # We sample indices uniformly across classes to keep the public pool balanced
    public_indices = []
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    samples_per_class = num_public_samples // num_classes
    for c in range(num_classes):
        selected = rng.choice(class_indices[c], size=samples_per_class, replace=False)
        public_indices.extend(selected.tolist())
        class_indices[c] = np.setdiff1d(class_indices[c], selected)

    # Step 2: Partition remaining data among clients using Dirichlet distribution
    client_indices = {str(k): [] for k in range(num_clients)}

    for c in range(num_classes):
        remaining_idx = class_indices[c]
        rng.shuffle(remaining_idx)

        # Draw client distribution proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(remaining_idx)).astype(int)

        # Fix rounding error
        diff = len(remaining_idx) - proportions.sum()
        for _ in range(diff):
            proportions[rng.integers(num_clients)] += 1

        # Allocate indices
        start = 0
        for k in range(num_clients):
            end = start + proportions[k]
            client_indices[str(k)].extend(remaining_idx[start:end].tolist())
            start = end

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
    dataset = load_dataset(dataset_name, train=train)
    indices = client_indices_dict[str(client_id)]
    client_subset = Subset(dataset, indices)
    return torch.utils.data.DataLoader(client_subset, batch_size=batch_size, shuffle=train)


def get_server_loaders(
    dataset_name: str, public_indices: list[int], batch_size: int = 64
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return public unlabeled server dataset loader and central evaluation test loader."""
    train_dataset = load_dataset(dataset_name, train=True)
    public_subset = Subset(train_dataset, public_indices)
    public_loader = torch.utils.data.DataLoader(public_subset, batch_size=batch_size, shuffle=False)

    test_dataset = load_dataset(dataset_name, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return public_loader, test_loader
