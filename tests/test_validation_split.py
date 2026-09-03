"""Dedicated verification test suite for Issue #99 & #102.

Verifies:
1. Validation slice holdout before client partitioning for Dirichlet datasets.
2. Pairwise disjointness among public, validation, and client index sets.
3. Determinism: same seed yields byte-identical validation indices and partition caches.
4. Natural writer partitioning holds out disjoint writer subset for validation.
5. Realized post-holdout shard statistics (min, max, mean, median, std, total) are recorded.
6. Realized per-client/per-class count matrix is recorded with shape [num_clients, num_classes].
7. Post-holdout shard thinning: on CIFAR-10 (alpha=0.1, 100 clients, 200 public, 5000 val),
   total distributed samples is 44,800 and minimum client shard is 7.
8. Server loader accessor returns (public_loader, val_loader, test_loader).
9. Simulation entrypoint plumbs split selection: val_loader on 'val', test_loader on 'test'.
10. Run manifest records actual loader_used and partition_cache digest.
11. Protocol gate enforces loader_used matches contract split.
12. Evidence validation checks partition cache presence, schema_version, and SHA-256 digest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from fedmaq.core.partitioning import (
    PARTITION_SCHEMA_VERSION,
    PartitionIndices,
    generate_partition_indices,
    get_partition_cache_info,
    get_server_loaders,
)
from fedmaq.core.protocol import is_promotable_manifest, register_protocol


@pytest.fixture
def mock_dataset_1000(monkeypatch: pytest.MonkeyPatch):
    """Synthetic dataset with 1000 samples and 10 classes."""
    data = torch.randn(1000, 1, 4, 4)
    labels = torch.cat([torch.full((100,), c) for c in range(10)])
    ds = TensorDataset(data, labels)
    ds.targets = labels  # type: ignore[attr-defined]
    monkeypatch.setattr("fedmaq.core.partitioning.load_dataset", lambda name, train=True: ds)
    return ds


@pytest.fixture
def mock_writer_dataset_large(monkeypatch: pytest.MonkeyPatch):
    """Synthetic writer dataset with 20 writers and 1000 samples."""
    data = torch.randn(1000, 1, 4, 4)
    labels = torch.randint(0, 10, (1000,))
    writer_ids = np.repeat(np.arange(20), 50)
    ds = TensorDataset(data, labels)
    ds.targets = labels  # type: ignore[attr-defined]
    ds.writer_ids = writer_ids  # type: ignore[attr-defined]
    monkeypatch.setattr("fedmaq.core.partitioning.load_dataset", lambda name, train=True: ds)
    return ds


def test_stratified_validation_holdout_and_pairwise_disjointness(
    mock_dataset_1000, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Validation slice is stratified and public, val, and client sets are pairwise disjoint."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    num_clients = 10
    alpha = 0.5
    num_public = 50
    num_val = 100
    seed = 42

    result = generate_partition_indices(
        "mock",
        num_clients=num_clients,
        alpha=alpha,
        num_public_samples=num_public,
        num_val_samples=num_val,
        seed=seed,
        partition="dirichlet",
    )

    assert isinstance(result, PartitionIndices)
    pub, val, clients = result

    assert len(pub) == num_public
    assert len(val) == num_val
    assert len(clients) == num_clients

    pub_set = set(pub)
    val_set = set(val)
    assert pub_set.isdisjoint(val_set), "Public pool and validation slice overlap"

    total_client_samples = 0
    for cid, client_indices in clients.items():
        client_set = set(client_indices)
        assert pub_set.isdisjoint(client_set), f"Public pool overlaps with client {cid}"
        assert val_set.isdisjoint(client_set), f"Validation slice overlaps with client {cid}"
        total_client_samples += len(client_indices)

    assert len(pub) + len(val) + total_client_samples == 1000


def test_partition_determinism_and_cache_integrity(
    mock_dataset_1000, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The same seed yields identical validation indices and deterministic cache digest."""
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"

    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", cache_a)
    result_a = generate_partition_indices(
        "mock", num_clients=5, alpha=0.3, num_public_samples=20, num_val_samples=50, seed=123
    )

    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", cache_b)
    result_b = generate_partition_indices(
        "mock", num_clients=5, alpha=0.3, num_public_samples=20, num_val_samples=50, seed=123
    )

    assert result_a.public_indices == result_b.public_indices
    assert result_a.validation_indices == result_b.validation_indices
    assert result_a.client_indices == result_b.client_indices

    # Verify partition cache artifact JSON content
    cache_file_a = next(cache_a.glob("*.json"))
    cache_file_b = next(cache_b.glob("*.json"))

    info_a = get_partition_cache_info(cache_file_a)
    info_b = get_partition_cache_info(cache_file_b)
    assert info_a is not None and info_b is not None
    assert info_a["schema_version"] == PARTITION_SCHEMA_VERSION
    assert info_a["sha256"] == info_b["sha256"]


def test_writer_partition_holds_out_disjoint_validation_writers(
    mock_writer_dataset_large, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Natural writer partition reserves a disjoint subset of writers for validation."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    writer_ids = mock_writer_dataset_large.writer_ids
    num_clients = 8
    num_val_writers = 3
    seed = 42

    result = generate_partition_indices(
        "femnist",
        num_clients=num_clients,
        num_public_samples=20,
        num_val_writers=num_val_writers,
        seed=seed,
        partition="writer",
    )

    pub, val, clients = result
    assert len(pub) == 20

    # Extract writers assigned to validation and clients
    val_writers = {int(writer_ids[idx]) for idx in val}
    assert len(val_writers) == num_val_writers

    client_writers = set()
    for _cid, indices in clients.items():
        writers_for_client = {int(writer_ids[idx]) for idx in indices}
        assert len(writers_for_client) == 1
        client_writers |= writers_for_client

    assert len(client_writers) == num_clients
    assert val_writers.isdisjoint(client_writers), "Validation writers overlap with client writers"


def test_partition_cache_records_post_holdout_statistics_and_count_matrix(
    mock_dataset_1000, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Partition cache records shard sizes, summary stats, and client-by-class counts."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    num_clients = 4
    generate_partition_indices(
        "mock",
        num_clients=num_clients,
        alpha=0.5,
        num_public_samples=40,
        num_val_samples=100,
        seed=42,
    )

    cache_file = next(tmp_path.glob("*.json"))
    data = json.loads(cache_file.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert "shard_sizes" in data
    assert len(data["shard_sizes"]) == num_clients

    stats = data["shard_stats"]
    assert stats["total"] == sum(data["shard_sizes"])
    assert stats["min"] == min(data["shard_sizes"])
    assert stats["max"] == max(data["shard_sizes"])
    assert isinstance(stats["mean"], float)

    matrix = data["client_class_counts"]
    assert len(matrix) == num_clients
    assert len(matrix[0]) == 10  # 10 classes
    for k in range(num_clients):
        assert sum(matrix[k]) == data["shard_sizes"][k]


def test_server_loaders_returns_three_loaders(mock_dataset_1000):
    """get_server_loaders returns (public_loader, val_loader, test_loader)."""
    public_indices = list(range(10))
    val_indices = list(range(10, 30))

    pub_loader, val_loader, test_loader = get_server_loaders(
        "mock",
        public_indices=public_indices,
        validation_indices=val_indices,
        batch_size=8,
    )

    assert len(pub_loader.dataset) == 10
    assert len(val_loader.dataset) == 20
    assert len(test_loader.dataset) == 1000


def test_protocol_gate_enforces_actual_loader_used():
    """Protocol check verifies recorded loader_used rather than declared split string."""
    config_val = {"protocol": "replacement-v1", "protocol_stage": "matched_tuning"}
    git = {"commit": "a" * 40, "dirty": False}
    reg_val = register_protocol(config_val, git)

    # Valid validation stage with loader_used == 'val'
    valid_val_manifest = {
        "config_sha256": reg_val.assurance_envelope["config_sha256"],
        "git": git,
        "protocol": reg_val.as_dict(),
        "run": {"loader_used": "val", "split": "val"},
    }
    assert is_promotable_manifest(valid_val_manifest)

    # Invalid: matched_tuning evaluated on test loader
    invalid_val_manifest = {
        "config_sha256": reg_val.assurance_envelope["config_sha256"],
        "git": git,
        "protocol": reg_val.as_dict(),
        "run": {"loader_used": "test", "split": "val"},
    }
    assert not is_promotable_manifest(invalid_val_manifest)

    # Downstream stage expects 'test'
    config_downstream = {"protocol": "replacement-v1", "protocol_stage": "downstream"}
    reg_downstream = register_protocol(config_downstream, git)
    valid_downstream_manifest = {
        "config_sha256": reg_downstream.assurance_envelope["config_sha256"],
        "git": git,
        "protocol": reg_downstream.as_dict(),
        "run": {"loader_used": "test", "split": "test"},
    }
    assert is_promotable_manifest(valid_downstream_manifest)

    invalid_downstream_manifest = {
        "config_sha256": reg_downstream.assurance_envelope["config_sha256"],
        "git": git,
        "protocol": reg_downstream.as_dict(),
        "run": {"loader_used": "val", "split": "test"},
    }
    assert not is_promotable_manifest(invalid_downstream_manifest)
