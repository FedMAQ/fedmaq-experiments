"""Dataset loading and deterministic partitioning with server-side public reserve.

Supported partition modes:
- ``dirichlet``: Artificial non-IID skew via Dirichlet(alpha) distribution. Used for
  CIFAR-10 and CIFAR-100.
- ``writer``: Writer-based natural partitioning for FEMNIST. Each client is one real
  NIST Special Database 19 writer, taken from the LEAF FEMNIST federated split, so the
  heterogeneity is the dataset's own rather than an artificial skew.

.. note::
    ``writer`` previously approximated writers by chunking each class into equal parts
    along torchvision EMNIST's natural ordering. That approximation handed every client
    a slice of every class, which made the FEMNIST arm label-IID (total-variation
    distance to the global label distribution 0.0013) *and* gave every client an
    identical local dataset size. The latter silently disabled FedMAQ's data-richness
    signal on FEMNIST, since a constant :math:`|D_k|` carries no information. Real LEAF
    writers give TV distance 0.259 and a 17.2x spread in samples per client.
"""

import hashlib
import json
import os
import random
from collections.abc import Sized
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

PARTITION_SCHEMA_VERSION = 1


class PartitionIndices(NamedTuple):
    """Deterministic partition assignment container."""

    public_indices: list[int]
    validation_indices: list[int]
    client_indices: dict[str, list[int]]


# Base paths
DATA_DIR = Path("data").resolve()
CACHE_DIR = Path(".data_partitions").resolve()

# LEAF FEMNIST (federated EMNIST) — the canonical writer-partitioned build, hosted by
# the TensorFlow Federated project. Used instead of torchvision's EMNIST because
# torchvision ships no writer identifiers, which is what forced the old approximation.
FEMNIST_LEAF_URL = "https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2"
FEMNIST_LEAF_DIR = DATA_DIR / "femnist_leaf"

# Standard normalizations for torchvision datasets
TRANSFORMS = {
    "mnist": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "fmnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    # FEMNIST normalization lives on LEAFFEMNIST itself, which reads memmapped uint8
    # arrays rather than PIL images and so cannot use a ToTensor-headed Compose.
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


def _prepare_femnist_leaf(split: str) -> tuple[Path, Path, Path]:
    """Materialize the LEAF FEMNIST split as memmap-friendly ``.npy`` arrays.

    The upstream HDF5 nests samples under one group per writer, which is awkward for
    the flat global index space the partitioner works in. We flatten it once into
    parallel image/label/writer arrays and cache them; subsequent runs just memmap.

    Pixels are stored as ``uint8`` with ink high. The upstream arrays are float32 in
    ``[0, 1]`` on a *white* (1.0) background, which is inverted relative to the
    MNIST/EMNIST convention the rest of this codebase and its normalization assume.
    """
    img_path = FEMNIST_LEAF_DIR / f"femnist_leaf_{split}_images.npy"
    lbl_path = FEMNIST_LEAF_DIR / f"femnist_leaf_{split}_labels.npy"
    wid_path = FEMNIST_LEAF_DIR / f"femnist_leaf_{split}_writers.npy"
    if img_path.is_file() and lbl_path.is_file() and wid_path.is_file():
        return img_path, lbl_path, wid_path

    import h5py  # local import: only the one-time conversion needs it

    h5_path = FEMNIST_LEAF_DIR / f"fed_emnist_{split}.h5"
    if not h5_path.is_file():
        _download_femnist_leaf()

    with h5py.File(h5_path, "r") as f:
        group = f["examples"]
        # Sorted for determinism: HDF5 key order is not guaranteed stable across builds.
        writer_keys = sorted(group.keys())
        total = sum(group[k]["label"].shape[0] for k in writer_keys)

        images = np.lib.format.open_memmap(
            img_path, mode="w+", dtype=np.uint8, shape=(total, 28, 28)
        )
        labels = np.empty(total, dtype=np.int64)
        writers = np.empty(total, dtype=np.int32)

        cursor = 0
        for widx, key in enumerate(writer_keys):
            px = group[key]["pixels"][:]
            lb = group[key]["label"][:]
            n = lb.shape[0]
            # Invert to ink-high, then quantize to uint8.
            images[cursor : cursor + n] = np.rint((1.0 - px) * 255.0).astype(np.uint8)
            labels[cursor : cursor + n] = lb
            writers[cursor : cursor + n] = widx
            cursor += n

        images.flush()

    np.save(lbl_path, labels)
    np.save(wid_path, writers)
    return img_path, lbl_path, wid_path


def _download_femnist_leaf() -> None:
    """Fetch and unpack the LEAF FEMNIST archive into :data:`FEMNIST_LEAF_DIR`."""
    import tarfile
    import urllib.request

    FEMNIST_LEAF_DIR.mkdir(parents=True, exist_ok=True)
    archive = FEMNIST_LEAF_DIR / "fed_emnist.tar.bz2"
    if not archive.is_file():
        urllib.request.urlretrieve(FEMNIST_LEAF_URL, archive)  # noqa: S310 (pinned https)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(FEMNIST_LEAF_DIR, filter="data")


class LEAFFEMNIST(Dataset):
    """LEAF FEMNIST with real NIST SD19 writer identifiers.

    Exposes ``targets`` and ``writer_ids`` as parallel arrays over a flat global index
    space, so :func:`generate_partition_indices` can carve a public proxy pool and then
    assign whole writers to clients using the same index arithmetic as the other
    datasets.
    """

    def __init__(self, train: bool = True) -> None:
        split = "train" if train else "test"
        img_path, lbl_path, wid_path = _prepare_femnist_leaf(split)
        self.images = np.load(img_path, mmap_mode="r")
        self.targets = np.load(lbl_path)
        self.writer_ids = np.load(wid_path)
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # copy(): the memmap is read-only, and torch.from_numpy rejects non-writable buffers
        img = torch.from_numpy(np.array(self.images[index])).unsqueeze(0).float().div_(255.0)
        return self.normalize(img), int(self.targets[index])


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
        # LEAF FEMNIST, which carries the real writer identifiers the ``writer``
        # partition mode needs. torchvision's EMNIST has no writer column.
        return LEAFFEMNIST(train=train)
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
        targets = [dataset[i][1] for i in range(len(cast(Sized, dataset)))]

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
    writer_ids: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
    num_val_writers: int | None = None,
) -> tuple[list[int], dict[str, list[int]]]:
    """Writer-based natural partition for FEMNIST: one real writer per client.

    Holds out a disjoint writer subset for server validation before assigning
    writers to clients. ``num_clients`` writers are drawn without replacement from
    those still holding samples after the public proxy pool and validation subset have
    been carved out. Samples belonging to unselected writers go unused.
    """
    if num_clients > 0 and writer_ids.size == 0:
        raise ValueError("writer partition requires a dataset exposing writer_ids")

    remaining = np.concatenate([idx for idx in class_indices.values() if idx.size > 0])
    remaining_writers = writer_ids[remaining]

    available = np.unique(remaining_writers)
    total_available = int(available.size)
    if num_val_writers is None:
        num_val_writers = min(
            max(1, int(round(num_clients * 0.1))), max(0, total_available - num_clients)
        )

    if total_available < num_clients + num_val_writers:
        raise ValueError(
            f"writer partition needs {num_clients} client writers + {num_val_writers} val writers "
            f"but only {total_available} remain after reserving the public pool"
        )

    # Deterministic selection: first val writers, then client writers from remainder
    val_writers = rng.choice(available, size=num_val_writers, replace=False)
    client_available = np.setdiff1d(available, val_writers)
    selected_clients = rng.choice(client_available, size=num_clients, replace=False)

    order = np.argsort(remaining_writers, kind="stable")
    sorted_writers = remaining_writers[order]

    # Extract validation sample indices
    val_indices: list[int] = []
    val_starts = np.searchsorted(sorted_writers, val_writers, side="left")
    val_ends = np.searchsorted(sorted_writers, val_writers, side="right")
    for start, end in zip(val_starts, val_ends, strict=True):
        val_indices.extend(remaining[order[start:end]].tolist())

    # Extract client sample indices
    client_starts = np.searchsorted(sorted_writers, selected_clients, side="left")
    client_ends = np.searchsorted(sorted_writers, selected_clients, side="right")
    client_indices = {
        str(k): remaining[order[start:end]].tolist()
        for k, (start, end) in enumerate(zip(client_starts, client_ends, strict=True))
    }

    return val_indices, client_indices


def canonical_partition_digest(cache_data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of canonical partition cache JSON."""
    canonical_bytes = json.dumps(cache_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def get_partition_cache_info(cache_file: Path | str) -> dict[str, Any] | None:
    """Return partition cache schema, relative path, and content digest, or None if missing."""
    path = Path(cache_file).resolve()
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        cache_data = json.load(f)
    digest = canonical_partition_digest(cache_data)
    try:
        rel_path = path.relative_to(DATA_DIR.parent).as_posix()
    except ValueError:
        rel_path = path.name
    return {
        "schema_version": int(cache_data.get("schema_version", PARTITION_SCHEMA_VERSION)),
        "path": rel_path,
        "sha256": digest,
    }


def generate_partition_indices(
    dataset_name: str,
    num_clients: int,
    alpha: float = 1.0,
    num_public_samples: int = 200,
    seed: int = 42,
    partition: str = "dirichlet",
    num_val_samples: int | None = None,
    num_val_writers: int | None = None,
    force: bool = False,
) -> PartitionIndices:
    """Generate or retrieve cached partition indices (deterministic).

    Holds out a stratified validation slice from the train pool before client
    partitioning for Dirichlet datasets, and a disjoint writer subset for FEMNIST
    writer partitioning.

    Args:
        dataset_name: Torchvision dataset identifier.
        num_clients: Number of federated clients / simulated writers.
        alpha: Dirichlet concentration parameter (unused for ``partition="writer"``).
        num_public_samples: Samples reserved for the server-side public proxy pool.
        seed: RNG seed for reproducibility.
        partition: ``"dirichlet"`` (default) or ``"writer"`` (FEMNIST natural partition).
        num_val_samples: Samples reserved for server validation (default 5,000 for CIFAR).
        num_val_writers: Writers reserved for server validation in writer mode.
        force: If True, ignore existing cache on disk and recompute.

    Returns:
        PartitionIndices(public_indices, validation_indices, client_indices)
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

    if cache_file.is_file() and not force:
        with open(cache_file) as f:
            cache_data = json.load(f)
        if (
            cache_data.get("schema_version") == PARTITION_SCHEMA_VERSION
            and "validation_indices" in cache_data
            and "public_indices" in cache_data
            and "client_indices" in cache_data
        ):
            return PartitionIndices(
                cache_data["public_indices"],
                cache_data["validation_indices"],
                cache_data["client_indices"],
            )

    # Generate indices
    dataset = load_dataset(dataset_name, train=True)
    labels = get_dataset_labels(dataset)
    num_classes = len(np.unique(labels))

    rng = np.random.default_rng(seed)

    public_indices: list[int] = []
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    base_per_class = num_public_samples // num_classes
    remainder = num_public_samples % num_classes
    for c in range(num_classes):
        target = base_per_class + (1 if c < remainder else 0)
        n_available = len(class_indices[c])
        n_select = min(target, n_available)
        selected = rng.choice(class_indices[c], size=n_select, replace=False)
        public_indices.extend(selected.tolist())
        class_indices[c] = np.setdiff1d(class_indices[c], selected)

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

    if partition == "writer":
        writer_ids = getattr(dataset, "writer_ids", None)
        if writer_ids is None:
            raise ValueError(
                f"partition='writer' requires writer identifiers, which "
                f"{dataset_name!r} does not expose"
            )
        val_indices, client_indices = _generate_writer_partition(
            class_indices, np.asarray(writer_ids), num_clients, rng, num_val_writers=num_val_writers
        )
    else:
        # Default validation holdout size: 5,000 for CIFAR or large datasets,
        # 10% for smaller mock datasets.
        if num_val_samples is None:
            if (
                dataset_name.lower() in ("cifar10", "cifar-10", "cifar100", "cifar-100")
                or len(labels) >= 50000
            ):
                num_val_samples = 5000
            else:
                num_val_samples = max(1, int(round(len(labels) * 0.10)))

        val_indices = []
        base_val_per_class = num_val_samples // num_classes
        val_remainder = num_val_samples % num_classes
        for c in range(num_classes):
            target = base_val_per_class + (1 if c < val_remainder else 0)
            n_available = len(class_indices[c])
            n_select = min(target, n_available)
            selected = rng.choice(class_indices[c], size=n_select, replace=False)
            val_indices.extend(selected.tolist())
            class_indices[c] = np.setdiff1d(class_indices[c], selected)

        val_shortfall = num_val_samples - len(val_indices)
        if val_shortfall > 0:
            for c in range(num_classes):
                if val_shortfall <= 0:
                    break
                n_available = len(class_indices[c])
                if n_available == 0:
                    continue
                n_select = min(val_shortfall, n_available)
                selected = rng.choice(class_indices[c], size=n_select, replace=False)
                val_indices.extend(selected.tolist())
                class_indices[c] = np.setdiff1d(class_indices[c], selected)
                val_shortfall -= n_select

        client_indices = _generate_dirichlet_partition(class_indices, num_clients, alpha, rng)

    # Compute realized post-holdout shard statistics & per-client class count matrix
    shard_sizes = [int(len(client_indices[str(k)])) for k in range(num_clients)]
    shard_stats = {
        "min": int(min(shard_sizes)) if shard_sizes else 0,
        "max": int(max(shard_sizes)) if shard_sizes else 0,
        "mean": float(np.mean(shard_sizes)) if shard_sizes else 0.0,
        "median": float(np.median(shard_sizes)) if shard_sizes else 0.0,
        "std": float(np.std(shard_sizes)) if shard_sizes else 0.0,
        "total": int(sum(shard_sizes)),
    }

    client_class_counts: list[list[int]] = []
    for k in range(num_clients):
        c_indices = client_indices[str(k)]
        if c_indices:
            client_labels = labels[c_indices]
            counts = [int(np.sum(client_labels == c)) for c in range(num_classes)]
        else:
            counts = [0] * num_classes
        client_class_counts.append(counts)

    # Save to cache
    cache_data = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "dataset": dataset_name,
        "num_clients": num_clients,
        "alpha": alpha if partition != "writer" else None,
        "num_public_samples": num_public_samples,
        "num_val_samples": len(val_indices),
        "seed": seed,
        "partition": partition,
        "public_indices": public_indices,
        "validation_indices": val_indices,
        "client_indices": client_indices,
        "shard_sizes": shard_sizes,
        "shard_stats": shard_stats,
        "client_class_counts": client_class_counts,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    return PartitionIndices(public_indices, val_indices, client_indices)


def _seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for deterministic multi-worker shuffling.

    No-op for the default ``num_workers=0`` (single-process) case, but keeps
    shuffling reproducible if worker processes are ever enabled.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_client_loader(
    dataset_name: str,
    client_id: int,
    client_indices_dict: dict[str, list[int]],
    batch_size: int = 64,
    train: bool = True,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    """Return PyTorch DataLoader for a specific client partition.

    When ``seed`` is given and ``train`` shuffling is on, a dedicated
    ``torch.Generator`` drives the shuffle so batch order is reproducible across
    runs and independent of global-RNG advancement.
    """
    dataset = _load_dataset_cached(dataset_name, train)
    indices = client_indices_dict[str(client_id)]
    client_subset = Subset(dataset, indices)
    generator = None
    if seed is not None and train:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        client_subset,
        batch_size=batch_size,
        shuffle=train,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def get_server_loaders(
    dataset_name: str,
    public_indices: list[int],
    validation_indices: list[int] | None = None,
    batch_size: int = 64,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return public unlabeled server dataset loader, validation loader, and central test loader."""
    train_dataset = _load_dataset_cached(dataset_name, True)
    public_subset = Subset(train_dataset, public_indices)
    public_loader = torch.utils.data.DataLoader(public_subset, batch_size=batch_size, shuffle=False)

    if validation_indices is not None:
        val_subset = Subset(train_dataset, validation_indices)
    else:
        val_subset = Subset(train_dataset, [])
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    test_dataset = _load_dataset_cached(dataset_name, False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return public_loader, val_loader, test_loader
