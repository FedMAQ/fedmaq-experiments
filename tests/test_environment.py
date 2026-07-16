"""Unit and integration tests for the Phase 1 federated learning environment."""

import flwr as fl
import numpy as np
import pytest
import torch
import torch.nn as nn
from flwr.clientapp import ClientApp
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation
from torch.utils.data import TensorDataset

from fedmaq.core.client import CompressionHook, GenericClient, LossHook
from fedmaq.core.models import (
    MobileNetV2GN,
    ResNet18GN,
    SimpleCNN,
    get_model,
    get_model_parameters,
    set_model_parameters,
)
from fedmaq.core.partitioning import (
    generate_partition_indices,
    get_client_loader,
    get_server_loaders,
)
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager


@pytest.fixture
def mock_dataset(monkeypatch):
    """Fixture to mock torchvision dataset download and loading."""
    # Create 100 samples of 1-channel 28x28 images (MNIST-like)
    mock_data = torch.randn(100, 1, 28, 28)
    mock_labels = torch.randint(0, 10, (100,))
    mock_ds = TensorDataset(mock_data, mock_labels)
    # Add targets attribute for target extraction helper
    mock_ds.targets = mock_labels

    # Patch load_dataset to return the mock dataset
    monkeypatch.setattr("fedmaq.core.partitioning.load_dataset", lambda name, train=True: mock_ds)
    return mock_ds


def test_model_factory_and_parameters():
    """Test get_model factory and get/set parameter helpers."""
    model = get_model("mnist", num_classes=10)
    assert isinstance(model, SimpleCNN)

    cifar_model = get_model("cifar10", num_classes=10)
    assert isinstance(cifar_model, MobileNetV2GN)

    # ResNet18GN is still available via explicit model_name override
    resnet_model = get_model("cifar10", num_classes=10, model_name="resnet18gn")
    assert isinstance(resnet_model, ResNet18GN)

    # Test parameter helpers
    params = get_model_parameters(model)
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)

    # Modify parameters and load back
    new_params = [p * 2.0 for p in params]
    set_model_parameters(model, new_params)

    # Verify modification
    re_extracted = get_model_parameters(model)
    for p_new, p_re in zip(new_params, re_extracted, strict=True):
        np.testing.assert_allclose(p_new, p_re, rtol=1e-5)


def test_set_model_parameters_raises_on_mismatch():
    """set_model_parameters must fail loudly on a count or shape mismatch.

    Regression: a plain zip silently truncated, so loading e.g. a TinyCNN's
    parameters into a SimpleCNN corrupted the model instead of erroring.
    """
    from fedmaq.core.models import TinyCNN

    simple = get_model("mnist", num_classes=10)  # SimpleCNN
    simple_params = get_model_parameters(simple)

    # Count mismatch: drop a tensor.
    with pytest.raises(ValueError, match="count mismatch"):
        set_model_parameters(simple, simple_params[:-1])

    # Shape mismatch at matching count: TinyCNN has the same tensor count as
    # SimpleCNN for 1-channel inputs but different conv widths (16 vs 32).
    tiny_params = get_model_parameters(TinyCNN(in_channels=1, num_classes=10))
    assert len(tiny_params) == len(simple_params)
    with pytest.raises(ValueError, match="shape mismatch"):
        set_model_parameters(simple, tiny_params)


def test_mobilenetv2gn_architecture():
    """Verify MobileNetV2GN forward pass, parameter count, and multi-class support."""
    model = MobileNetV2GN(in_channels=3, num_classes=10)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ~2.3M params — sanity check it's in the right ballpark (not ResNet18GN's ~11M)
    assert 2_000_000 < total_params < 3_000_000, f"Unexpected param count: {total_params}"

    # Forward pass with CIFAR-10 shaped input
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)

    # CIFAR-100 variant
    model_100 = MobileNetV2GN(in_channels=3, num_classes=100)
    y_100 = model_100(x)
    assert y_100.shape == (2, 100)

    # get/set parameter round-trip
    params = get_model_parameters(model)
    model2 = MobileNetV2GN(in_channels=3, num_classes=10)
    set_model_parameters(model2, params)
    params2 = get_model_parameters(model2)
    for p1, p2 in zip(params, params2, strict=True):
        np.testing.assert_allclose(p1, p2, rtol=1e-5)

    # Invalid model_name raises
    with pytest.raises(ValueError, match="Unknown CIFAR model"):
        get_model("cifar10", num_classes=10, model_name="nonexistent")


def test_deterministic_dirichlet_partitioning(mock_dataset, tmp_path, monkeypatch):
    """Test Dirichlet partitioning with public reserve and local caching."""
    # Patch CACHE_DIR to temp directory for testing
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    num_clients = 3
    alpha = 0.5
    num_public = 10
    seed = 42

    # First run (generates cache)
    pub_idx1, client_dict1 = generate_partition_indices(
        "mnist", num_clients, alpha, num_public, seed
    )

    assert len(pub_idx1) == num_public
    assert len(client_dict1) == num_clients
    total_client_samples = sum(len(indices) for indices in client_dict1.values())
    assert len(pub_idx1) + total_client_samples == 100

    # Second run (retrieves from cache)
    pub_idx2, client_dict2 = generate_partition_indices(
        "mnist", num_clients, alpha, num_public, seed
    )

    # Verify determinism and cache retrieval
    assert pub_idx1 == pub_idx2
    for k in client_dict1.keys():
        assert client_dict1[k] == client_dict2[k]


def test_partition_seed_invariant_for_paired_arms(mock_dataset, tmp_path, monkeypatch):
    """Partition is a pure function of (dataset, num_clients, alpha, num_public,
    seed) with **no algorithm input** — the invariant the paired statistical test
    relies on. Two arms (e.g. FedMAQ vs FedAvg) run at the same seed must see
    byte-identical partitions, while distinct seeds must give distinct partitions.

    Unlike ``test_deterministic_dirichlet_partitioning`` (whose second call hits the
    JSON cache and so only proves the cache round-trips), this regenerates from
    scratch into a *separate* cache dir, proving the generation itself — not just
    the cache read — is reproducible. The two fresh cache dirs stand in for two
    independently launched paired arms.
    """
    num_clients, alpha, num_public, seed = 3, 0.5, 10, 42

    # Arm A and Arm B: same config + seed, independent cache dirs (fresh generation).
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path / "arm_a")
    pub_a, clients_a = generate_partition_indices("mnist", num_clients, alpha, num_public, seed)

    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path / "arm_b")
    pub_b, clients_b = generate_partition_indices("mnist", num_clients, alpha, num_public, seed)

    assert pub_a == pub_b, "Paired arms at the same seed diverged on the public pool"
    for k in clients_a:
        assert clients_a[k] == clients_b[k], f"Paired arms diverged on client {k}"

    # A different seed (independent replicate) must produce a different partition.
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path / "seed_7")
    _, clients_c = generate_partition_indices("mnist", num_clients, alpha, num_public, seed=7)
    assert any(clients_a[k] != clients_c[k] for k in clients_a), (
        "Distinct seeds produced identical partitions — replicates are not independent"
    )


def test_writer_based_partitioning(mock_dataset, tmp_path, monkeypatch):
    """Test writer-based natural partitioning (FEMNIST mode) with caching."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    num_clients = 3
    num_public = 10
    seed = 42

    # First run (generates cache)
    pub_idx1, client_dict1 = generate_partition_indices(
        "femnist",
        num_clients,
        num_public_samples=num_public,
        seed=seed,
        partition="writer",
    )

    assert len(pub_idx1) == num_public
    assert len(client_dict1) == num_clients

    # All samples should be accounted for (public + client partitions)
    total_client_samples = sum(len(v) for v in client_dict1.values())
    assert len(pub_idx1) + total_client_samples == 100

    # No overlap between public pool and any client partition
    pub_set = set(pub_idx1)
    for indices in client_dict1.values():
        assert pub_set.isdisjoint(set(indices)), "Public pool overlaps with client data"

    # Second run (retrieves from cache — writer cache key is distinct from dirichlet)
    pub_idx2, client_dict2 = generate_partition_indices(
        "femnist",
        num_clients,
        num_public_samples=num_public,
        seed=seed,
        partition="writer",
    )
    assert pub_idx1 == pub_idx2
    for k in client_dict1.keys():
        assert client_dict1[k] == client_dict2[k]


def test_public_pool_exact_size_with_remainder(tmp_path, monkeypatch):
    """Public pool must total exactly num_public_samples even when it doesn't
    divide evenly across classes (e.g. FEMNIST's 62 classes), instead of
    silently dropping num_public_samples % num_classes samples."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    # 7 classes, 20 samples each (140 total) — 17 % 7 != 0, exercising the remainder path.
    num_classes = 7
    samples_per_class = 20
    data = torch.randn(num_classes * samples_per_class, 1, 4, 4)
    labels = torch.cat([torch.full((samples_per_class,), c) for c in range(num_classes)])
    mock_ds = TensorDataset(data, labels)
    mock_ds.targets = labels
    monkeypatch.setattr("fedmaq.core.partitioning.load_dataset", lambda name, train=True: mock_ds)

    from fedmaq.core.partitioning import generate_partition_indices

    num_public = 17  # 17 // 7 = 2 base, remainder = 3
    pub_idx, client_dict = generate_partition_indices(
        "mnist", num_clients=2, alpha=0.5, num_public_samples=num_public, seed=42
    )

    assert len(pub_idx) == num_public


def test_client_server_loaders(mock_dataset):
    """Test retrieval of client and server PyTorch DataLoaders."""
    client_dict = {"0": [0, 1, 2], "1": [3, 4]}
    pub_idx = [5, 6, 7]

    train_loader = get_client_loader(
        "mnist", client_id=0, client_indices_dict=client_dict, batch_size=2, train=True
    )
    assert len(train_loader.dataset) == 3

    pub_loader, test_loader = get_server_loaders("mnist", pub_idx, batch_size=2)
    assert len(pub_loader.dataset) == 3
    assert len(test_loader.dataset) == 100  # patched test dataset has 100 samples


def test_simulation_dry_run(mock_dataset, tmp_path, monkeypatch):
    """Test 1-round CPU dry-run simulation of the client/server environment."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    # 1. Setup partitioning
    public_indices, client_indices_dict = generate_partition_indices(
        "mnist", num_clients=2, alpha=0.5, num_public_samples=10, seed=42
    )

    # 2. Config dict
    cfg_dict = {
        "num_clients": 2,
        "batch_size": 2,
        "local_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "total_rounds": 1,
        "seed": 42,
        "client_fraction": 1.0,
        "bandwidth_mbps": 10.0,
        "compute_samples_per_sec": 200.0,
        "num_public_samples": 10,
        "telemetry": {
            "wandb_enabled": False,
            "project": "test",
            "mode": "offline",
        },
        "algorithm": {"name": "fedavg"},
        "dataset": {"name": "mnist", "num_classes": 10},
    }

    # 3. Telemetry and components
    telemetry = TelemetryManager(cfg_dict)

    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        train_loader = get_client_loader(
            "mnist", partition_id, client_indices_dict, batch_size=2, train=True
        )
        model = get_model("mnist", num_classes=10)
        return GenericClient(
            cid=str(partition_id),
            trainloader=train_loader,
            testloader=train_loader,
            model=model,
            loss_hook=LossHook(),
            compressor_hook=CompressionHook(),
            config=cfg_dict,
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)

    strategy = None

    def server_fn(context):
        nonlocal strategy
        global_model = get_model("mnist", num_classes=10)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(global_model))
        _, test_loader = get_server_loaders("mnist", public_indices, batch_size=2)

        def evaluate_fn(server_round, parameters, config):
            return 0.5, {"accuracy": 0.9}

        strategy = TelemetryFedAvg(
            telemetry_manager=telemetry,
            config=cfg_dict,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=2,
            min_available_clients=2,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=1))

    server_app = ServerApp(server_fn=server_fn)

    # 4. Run simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=2,
    )

    # Check that telemetry recorded cumulative bytes and simulated time
    assert telemetry.cumulative_bytes > 0
    assert strategy.simulated_time > 0


def test_dadaquant_compression_hook():
    """Test DAdaQuantCompressionHook stochastic rounding and size estimation."""
    from fedmaq.baselines.quantization import DAdaQuantCompressionHook

    # Test size estimation
    hook = DAdaQuantCompressionHook(q=4)
    deltas = [np.ones((100,), dtype=np.float32)]
    compressed_deltas, byte_size = hook.compress(deltas)

    # 100 elements * 4 bits = 400 bits = 50 bytes + 4 bytes scale = 54 bytes
    assert byte_size == 54
    assert len(compressed_deltas) == 1
    assert compressed_deltas[0].shape == (100,)

    # Test stochastic rounding unbiased property
    np.random.seed(42)
    hook_unbiased = DAdaQuantCompressionHook(q=1)
    large_deltas = [np.full((10000,), 0.5, dtype=np.float32)]
    decompressed, _ = hook_unbiased.compress(large_deltas)
    mean_val = np.mean(decompressed[0])

    # Assert mean is close to 0.5 (within standard error)
    np.testing.assert_allclose(mean_val, 0.5, atol=0.03)


def test_dadaquant_strategy_allocation():
    """Test TelemetryFedAvg time and client adaptive logic for DAdaQuant."""
    cfg_dict = {
        "num_clients": 2,
        "batch_size": 2,
        "seed": 42,
        "experiment": {
            "num_clients": 2,
            "client_fraction": 1.0,
            "total_rounds": 10,
        },
        "algorithm": {
            "name": "dadaquant",
            "psi": 0.9,
            "phi": 3,
            "q_min": 4,
            "q_max": 8,
        },
    }

    client_indices_dict = {
        "0": list(range(100)),  # weight: 100/300 = 1/3
        "1": list(range(200)),  # weight: 200/300 = 2/3
    }

    telemetry = TelemetryManager(cfg_dict)
    strategy = TelemetryFedAvg(
        telemetry_manager=telemetry,
        config=cfg_dict,
        client_indices_dict=client_indices_dict,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    class MockClientProxy(fl.server.client_proxy.ClientProxy):
        def __init__(self, cid: str) -> None:
            super().__init__(cid)

        def get_properties(self, ins, timeout=None, group_id=None):
            from flwr.common import Code, GetPropertiesRes, Status

            return GetPropertiesRes(
                status=Status(code=Code.OK, message=""),
                properties={"cid": int(self.cid)},
            )

        def get_parameters(self, ins, timeout):
            return None

        def fit(self, ins, timeout):
            return None

        def evaluate(self, ins, timeout):
            return None

        def reconnect(self, ins, timeout):
            return None

    # Test round 1 client-adaptive configuration
    params = ndarrays_to_parameters([np.zeros((10,))])
    client_manager = fl.server.client_manager.SimpleClientManager()
    client_manager.register(MockClientProxy("0"))
    client_manager.register(MockClientProxy("1"))

    instructions = strategy.configure_fit(
        server_round=1, parameters=params, client_manager=client_manager
    )

    assert len(instructions) == 2
    # Verify q is in the config dict
    q_dict = {inst.cid: fit_ins.config["q"] for inst, fit_ins in instructions}

    # Larger client (cid "1") should have larger or equal quantization level than client "0"
    assert q_dict["1"] >= q_dict["0"]
    # Both should be positive
    assert q_dict["0"] >= 1
    assert q_dict["1"] >= 1

    # Simulate convergence to test time-adaptive q_t doubling
    # phi is 3, so we check convergence when we have at least phi + 1 (4) rounds of history
    # and rounds_since_increase >= 3.
    strategy.hook.moving_average_history = [1.0, 1.0, 1.0, 1.0]  # Plateau detected
    strategy.hook.last_quantization_increase_round = 0
    strategy.hook.q_t = 4

    # Run configure_fit for round 5 (checks history up to round 4)
    instructions = strategy.configure_fit(
        server_round=5, parameters=params, client_manager=client_manager
    )
    # Since a plateau is detected (latest loss 1.0 >= past loss 1.0), q_t should double from 4 to 8
    assert strategy.hook.q_t == 8
    assert strategy.hook.last_quantization_increase_round == 4


def test_fedmd_simulation_dry_run(mock_dataset, tmp_path, monkeypatch):
    """Test 2-round simulation of the FedMD baseline implementation."""
    # Patch CACHE_DIR to temp directory for testing
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    import shutil
    from pathlib import Path

    model_dir = Path(".data_partitions/fedmd_models")
    if model_dir.exists():
        shutil.rmtree(model_dir)

    try:
        # Setup partitioning
        public_indices, client_indices_dict = generate_partition_indices(
            "mnist", num_clients=2, alpha=0.5, num_public_samples=10, seed=42
        )

        cfg_dict = {
            "num_clients": 2,
            "batch_size": 2,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "total_rounds": 2,
            "seed": 42,
            "client_fraction": 1.0,
            "bandwidth_mbps": 10.0,
            "compute_samples_per_sec": 200.0,
            "num_public_samples": 10,
            "telemetry": {
                "wandb_enabled": False,
                "project": "test",
                "mode": "offline",
            },
            "algorithm": {
                "name": "fedmd",
                "public_pretrain_epochs": 1,
                "private_pretrain_epochs": 1,
                "public_epochs": 1,
            },
            "dataset": {"name": "mnist", "num_classes": 10},
        }

        telemetry = TelemetryManager(cfg_dict)

        def client_fn(context):
            partition_id = context.node_config["partition-id"]
            train_loader = get_client_loader(
                "mnist", partition_id, client_indices_dict, batch_size=2, train=True
            )
            public_loader, _ = get_server_loaders("mnist", public_indices, batch_size=2)
            model = get_model("mnist", num_classes=10)
            return GenericClient(
                cid=str(partition_id),
                trainloader=train_loader,
                testloader=train_loader,
                model=model,
                loss_hook=LossHook(),
                compressor_hook=CompressionHook(),
                config=cfg_dict,
                public_loader=public_loader,
            ).to_client()

        client_app = ClientApp(client_fn=client_fn)

        strategy = None

        def server_fn(context):
            nonlocal strategy
            global_model = get_model("mnist", num_classes=10)
            initial_parameters = ndarrays_to_parameters(get_model_parameters(global_model))
            _, test_loader = get_server_loaders("mnist", public_indices, batch_size=2)

            def evaluate_fn(server_round, parameters, config):
                # Simple ensemble eval simulation
                client_paths = list(model_dir.glob("client_*.pth")) if model_dir.exists() else []
                assert len(client_paths) <= 2
                return 0.5, {"accuracy": 0.9}

            strategy = TelemetryFedAvg(
                telemetry_manager=telemetry,
                config=cfg_dict,
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                min_fit_clients=2,
                min_available_clients=2,
                evaluate_fn=evaluate_fn,
                initial_parameters=initial_parameters,
            )
            return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=2))

        server_app = ServerApp(server_fn=server_fn)

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=2,
        )

        # Verify that client models were saved
        assert model_dir.exists()
        assert len(list(model_dir.glob("client_*.pth"))) == 2
        assert strategy.simulated_time > 0
        assert telemetry.cumulative_bytes > 0

    finally:
        # Clean up models
        if model_dir.exists():
            shutil.rmtree(model_dir)


def test_fedkd_compression_hook():
    """Test FedKDCompressionHook SVD compression and size estimation."""
    from fedmaq.baselines.compression import FedKDCompressionHook

    hook = FedKDCompressionHook(energy=0.5)

    u_true = np.arange(10, dtype=np.float32).reshape(10, 1)
    v_true = np.arange(5, dtype=np.float32).reshape(1, 5)
    rank1_matrix = u_true @ v_true

    deltas = [rank1_matrix]
    reconstructed, byte_size = hook.compress(deltas)

    assert len(reconstructed) == 1
    assert reconstructed[0].shape == (10, 5)
    # Rank is 1. U is (10,1), Sigma is (1,), V is (1,5). Total floats = 16. Bytes = 64.
    assert byte_size == 64
    np.testing.assert_allclose(reconstructed[0], rank1_matrix, atol=1e-5)


def test_fedkd_client_fit(mock_dataset, tmp_path, monkeypatch):
    """Test GenericClient fit step with FedKD active."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    persistence_dir = tmp_path / "fedkd_models"

    cfg_dict = {
        "num_clients": 2,
        "batch_size": 2,
        "local_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "seed": 42,
        "experiment": {
            "persistence_dir": str(persistence_dir),
            "local_epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
        "algorithm": {
            "name": "fedkd",
            "tmin": 0.1,
            "tmax": 0.9,
            "temperature": 2.0,
        },
        "dataset": {"name": "mnist", "num_classes": 10},
    }

    train_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=2)

    from fedmaq.core.models import TinyCNN

    model = TinyCNN(in_channels=1, num_classes=10)

    from fedmaq.baselines.compression import FedKDCompressionHook

    compressor_hook = FedKDCompressionHook(energy=0.5)

    client = GenericClient(
        cid="0",
        trainloader=train_loader,
        testloader=train_loader,
        model=model,
        loss_hook=LossHook(),
        compressor_hook=compressor_hook,
        config=cfg_dict,
    )

    initial_params = get_model_parameters(model)
    reconstructed, num_examples, metrics = client.fit(initial_params, {"energy": 0.5})

    assert num_examples == 100
    assert "bytes_uploaded" in metrics
    assert metrics["partition_id"] == 0

    teacher_file = persistence_dir / "teacher_0.pth"
    assert teacher_file.exists()


def test_dadaquant_fit_reports_pretrain_loss(mock_dataset):
    """DAdaQuant's local_loss must be the CE loss on the incoming model BEFORE training.

    This feeds the server-side plateau -> q_t doubling, so the eval-before-train
    ordering is behavior-critical and not exercised by short smoke runs (plateau
    needs phi+1 non-improving rounds).
    """
    from fedmaq.core.models import SimpleCNN, get_model_parameters

    train_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=4)
    model = SimpleCNN(in_channels=1, num_classes=10)
    initial_params = get_model_parameters(model)

    # Expected pre-training loss: CE over the loader on the untrained model.
    reference = SimpleCNN(in_channels=1, num_classes=10)
    set_model_parameters(reference, initial_params)
    reference.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            out = reference(images)
            loss_sum += criterion(out, labels).item() * len(labels)
            n += len(labels)
    expected = loss_sum / n

    cfg_dict = {
        "experiment": {"local_epochs": 1, "learning_rate": 0.01, "weight_decay": 0.0},
        "algorithm": {"name": "dadaquant", "q_min": 1, "q_max": 8},
        "dataset": {"name": "mnist", "num_classes": 10},
    }
    client = GenericClient(
        cid="0",
        trainloader=train_loader,
        testloader=train_loader,
        model=model,
        loss_hook=LossHook(),
        compressor_hook=CompressionHook(),
        config=cfg_dict,
    )
    _, _, metrics = client.fit(initial_params, {"q": 4})

    assert metrics["local_loss"] > 0.0
    # Reported loss is measured on the pre-training model, so it matches the
    # reference computed on the initial parameters (not the post-training loss).
    assert metrics["local_loss"] == pytest.approx(expected, rel=1e-4)


def test_fedkd_simulation_dry_run(mock_dataset, tmp_path, monkeypatch):
    """Test 2-round simulation of the FedKD baseline implementation."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    import shutil

    persistence_dir = tmp_path / "fedkd_models"

    try:
        public_indices, client_indices_dict = generate_partition_indices(
            "mnist", num_clients=2, alpha=0.5, num_public_samples=10, seed=42
        )

        cfg_dict = {
            "num_clients": 2,
            "batch_size": 2,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "total_rounds": 2,
            "seed": 42,
            "client_fraction": 1.0,
            "bandwidth_mbps": 10.0,
            "compute_samples_per_sec": 200.0,
            "telemetry": {
                "wandb_enabled": False,
                "project": "test",
                "mode": "offline",
            },
            "experiment": {
                "num_clients": 2,
                "client_fraction": 1.0,
                "total_rounds": 2,
                "persistence_dir": str(persistence_dir),
                "local_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "bandwidth_mbps": 10.0,
                "compute_samples_per_sec": 200.0,
                "num_public_samples": 10,
            },
            "algorithm": {
                "name": "fedkd",
                "tmin": 0.1,
                "tmax": 0.9,
                "temperature": 2.0,
            },
            "dataset": {"name": "mnist", "num_classes": 10},
        }

        telemetry = TelemetryManager(cfg_dict)

        def client_fn(context):
            partition_id = context.node_config["partition-id"]
            train_loader = get_client_loader(
                "mnist", partition_id, client_indices_dict, batch_size=2, train=True
            )
            from fedmaq.core.models import TinyCNN

            model = TinyCNN(in_channels=1, num_classes=10)
            from fedmaq.baselines.compression import FedKDCompressionHook

            compressor_hook = FedKDCompressionHook(energy=0.1)

            return GenericClient(
                cid=str(partition_id),
                trainloader=train_loader,
                testloader=train_loader,
                model=model,
                loss_hook=LossHook(),
                compressor_hook=compressor_hook,
                config=cfg_dict,
            ).to_client()

        client_app = ClientApp(client_fn=client_fn)

        strategy = None

        def server_fn(context):
            nonlocal strategy
            from fedmaq.core.models import TinyCNN

            global_model = TinyCNN(in_channels=1, num_classes=10)
            initial_parameters = ndarrays_to_parameters(get_model_parameters(global_model))
            _, test_loader = get_server_loaders("mnist", public_indices, batch_size=2)

            def evaluate_fn(server_round, parameters, config):
                return 0.5, {"accuracy": 0.9}

            strategy = TelemetryFedAvg(
                telemetry_manager=telemetry,
                config=cfg_dict,
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                min_fit_clients=2,
                min_available_clients=2,
                evaluate_fn=evaluate_fn,
                initial_parameters=initial_parameters,
            )
            return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=2))

        server_app = ServerApp(server_fn=server_fn)

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=2,
        )

        assert persistence_dir.exists()
        assert len(list(persistence_dir.glob("teacher_*.pth"))) == 2
        assert strategy.simulated_time > 0
        assert telemetry.cumulative_bytes > 0

    finally:
        if persistence_dir.exists():
            shutil.rmtree(persistence_dir)


def test_fedpaq_compression_hook():
    """Test FedPAQCompressionHook uniform symmetric quantization."""
    from fedmaq.baselines.quantization import FedPAQCompressionHook

    hook = FedPAQCompressionHook(q=8)
    deltas = [np.array([-2.0, 0.0, 2.0], dtype=np.float32)]
    compressed, byte_size = hook.compress(deltas)

    # Scale = 2.0
    # Level = (1 << 7) - 1 = 127
    # element_bits = 3 * 8 = 24 bits = 3 bytes + 4 bytes scale = 7 bytes
    assert byte_size == 7
    assert len(compressed) == 1
    np.testing.assert_allclose(compressed[0], np.array([-2.0, 0.0, 2.0], dtype=np.float32))


def test_fedpaq_no_nan_for_all_permissible_bit_widths():
    """FedPAQ quantization must never emit NaN for any q in the permissible set Q.

    Regression: at q=1 the old (1 << (q-1)) - 1 gave 0 positive levels and a
    0/0 NaN on dequantization, which is reachable via FedMAQ's Tier-1 memory cap.
    """
    from fedmaq.baselines.quantization import FedPAQCompressionHook

    rng = np.random.default_rng(0)
    deltas = [rng.standard_normal((4, 3)).astype(np.float32)]
    for q in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32):
        compressed, _ = FedPAQCompressionHook(q=q).compress(deltas)
        assert np.all(np.isfinite(compressed[0])), f"NaN/Inf produced at q={q}"


def test_fedpaq_q1_is_sign_quantization():
    """q=1 must reduce to sign quantization: nonzero elements map to +/- scale."""
    from fedmaq.baselines.quantization import FedPAQCompressionHook

    deltas = [np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)]
    compressed, _ = FedPAQCompressionHook(q=1).compress(deltas)
    # scale = max|d| = 3.0; every nonzero element -> sign(d)*scale, zeros stay 0.
    np.testing.assert_allclose(
        compressed[0], np.array([-3.0, -3.0, 0.0, 3.0, 3.0], dtype=np.float32)
    )
    assert set(np.unique(compressed[0]).tolist()) <= {-3.0, 0.0, 3.0}


def test_fedprox_loss_hook():
    """Test FedProxLossHook proximal L2 regularization."""
    from fedmaq.core.client import FedProxLossHook

    model = nn.Linear(10, 2)
    hook = FedProxLossHook(mu=0.1)
    hook.on_train_begin(model)

    # Check that global params are saved and detached
    assert len(hook.global_params) == 2
    assert not hook.global_params[0].requires_grad

    # Modify model parameters
    with torch.no_grad():
        list(model.parameters())[0].add_(1.0)

    outputs = model(torch.randn(5, 10))
    targets = torch.randint(0, 2, (5,))
    criterion = nn.CrossEntropyLoss()

    loss = hook.compute_loss(model, outputs, targets, criterion)
    # Proximal term should be non-zero because parameter is modified
    assert loss.item() > 0.0


def test_fedmaq_strategy_allocation():
    """Test TelemetryFedAvg memory cap and multi-adaptive formula calculations for FedMAQ."""
    cfg_dict = {
        "num_clients": 2,
        "batch_size": 2,
        "seed": 42,
        "experiment": {
            "num_clients": 2,
            "client_fraction": 1.0,
            "total_rounds": 10,
        },
        "algorithm": {
            "name": "fedmaq_lite",
            "q_min": 2,
            "q_max": 8,
            "c_unit": 2048.0,
            "formulation": 3,
            "lambda_val": 1.0,
        },
    }

    client_indices_dict = {
        "0": list(range(50)),  # weight: 50/150 = 1/3
        "1": list(range(100)),  # weight: 100/150 = 2/3
    }
    public_indices = list(range(10))

    telemetry = TelemetryManager(cfg_dict)
    strategy = TelemetryFedAvg(
        telemetry_manager=telemetry,
        config=cfg_dict,
        client_indices_dict=client_indices_dict,
        public_indices=public_indices,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    class MockClientProxy(fl.server.client_proxy.ClientProxy):
        def __init__(self, cid: str) -> None:
            super().__init__(cid)

        def get_properties(self, ins, timeout=None, group_id=None):
            # Return partition ID matching proxy cid
            from flwr.common import Code, GetPropertiesRes, Status

            return GetPropertiesRes(
                status=Status(code=Code.OK, message=""),
                properties={"cid": int(self.cid)},
            )

        def get_parameters(self, ins, timeout):
            return None

        def fit(self, ins, timeout):
            return None

        def evaluate(self, ins, timeout):
            return None

        def reconnect(self, ins, timeout):
            return None

    # Test round 1 client-adaptive configuration for FedMAQ
    # Mock parameters representing global weights
    from fedmaq.core.models import TinyCNN, get_model_parameters

    model = TinyCNN(in_channels=1, num_classes=10)
    params = ndarrays_to_parameters(get_model_parameters(model))

    client_manager = fl.server.client_manager.SimpleClientManager()
    client_manager.register(MockClientProxy("0"))
    client_manager.register(MockClientProxy("1"))

    # Mock the client loader data to avoid file reads in test
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    mock_ds = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))
    mock_loader = DataLoader(mock_ds, batch_size=2)

    import fedmaq.core.partitioning

    strategy.public_indices = public_indices

    # Patch client memory for controlled testing
    strategy.client_memory = np.array([2048.0, 16384.0])  # Client 0: Q_max=1, Client 1: Q_max=8

    # We patch get_client_loader in partitioning module since it is imported inside configure_fit
    original_loader = fedmaq.core.partitioning.get_client_loader
    fedmaq.core.partitioning.get_client_loader = lambda *args, **kwargs: mock_loader

    try:
        instructions = strategy.configure_fit(
            server_round=1, parameters=params, client_manager=client_manager
        )
    finally:
        fedmaq.core.partitioning.get_client_loader = original_loader

    assert len(instructions) == 2
    q_dict = {inst.cid: fit_ins.config["q"] for inst, fit_ins in instructions}

    # Check that memory hard cap is enforced
    # Client 0 memory capacity is 2048 MB, c_unit = 2048.0 -> Q_max = 1
    # Note that q_min is 2, so client 0 should be capped at min(Q_max, q_hat) -> min(1, q_hat) = 1
    assert q_dict["0"] == 1
    # Client 1 memory capacity is 16384 MB -> Q_max = 8. It should have a larger q
    assert q_dict["1"] >= 1


def test_fedmaq_simulation_dry_run(mock_dataset, tmp_path, monkeypatch):
    """Test 2-round simulation of the FedMAQ algorithm implementation."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    import shutil
    from pathlib import Path

    persistence_dir = Path(".data_partitions/fedmaq_models")
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)

    try:
        # Setup partitioning
        public_indices, client_indices_dict = generate_partition_indices(
            "mnist", num_clients=2, alpha=0.5, num_public_samples=10, seed=42
        )

        cfg_dict = {
            "num_clients": 2,
            "batch_size": 2,
            "local_epochs": 1,
            "seed": 42,
            "experiment": {
                "num_clients": 2,
                "client_fraction": 1.0,
                "total_rounds": 2,
                "batch_size": 2,
                "num_public_samples": 10,
                "local_epochs": 1,
                "persistence_dir": str(persistence_dir),
            },
            "dataset": {
                "name": "mnist",
                "num_classes": 10,
            },
            "algorithm": {
                "name": "fedmaq_lite",
                "q_min": 2,
                "q_max": 8,
                "c_unit": 2048.0,
                "formulation": 3,
                "lambda_val": 1.0,
                "temperature": 1.0,
                "kd_weight": 0.5,
            },
        }

        telemetry = TelemetryManager(cfg_dict)

        def client_fn(context: fl.app.Context) -> fl.client.Client:
            partition_id = context.node_config["partition-id"]
            train_loader = get_client_loader(
                "mnist", partition_id, client_indices_dict, batch_size=2, train=True
            )
            public_loader, _ = get_server_loaders("mnist", public_indices, batch_size=2)
            from fedmaq.core.models import TinyCNN

            model = TinyCNN(in_channels=1, num_classes=10)
            loss_hook = LossHook()
            from fedmaq.baselines.quantization import DAdaQuantCompressionHook

            compressor_hook = DAdaQuantCompressionHook(q=2)

            return GenericClient(
                cid=str(partition_id),
                trainloader=train_loader,
                testloader=train_loader,
                model=model,
                loss_hook=loss_hook,
                compressor_hook=compressor_hook,
                config=cfg_dict,
                public_loader=public_loader,
            ).to_client()

        client_app = ClientApp(client_fn=client_fn)

        strategy = None

        def server_fn(context: fl.app.Context) -> ServerAppComponents:
            nonlocal strategy
            from fedmaq.core.models import TinyCNN

            global_model = TinyCNN(in_channels=1, num_classes=10)
            initial_parameters = ndarrays_to_parameters(get_model_parameters(global_model))

            def evaluate_fn(server_round, parameters, config):
                return 0.5, {"accuracy": 0.9}

            strategy = TelemetryFedAvg(
                telemetry_manager=telemetry,
                config=cfg_dict,
                client_indices_dict=client_indices_dict,
                public_indices=public_indices,
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                min_fit_clients=2,
                min_available_clients=2,
                evaluate_fn=evaluate_fn,
                initial_parameters=initial_parameters,
            )
            return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=2))

        server_app = ServerApp(server_fn=server_fn)

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=2,
        )

        # Verify simulation output variables
        assert strategy.simulated_time > 0
        assert telemetry.cumulative_bytes > 0
        assert telemetry.jsonl_path.exists()
        assert telemetry.csv_path.exists()

    finally:
        if persistence_dir.exists():
            shutil.rmtree(persistence_dir)


def test_compute_fedmaq_q_k_t():
    """Test FedMAQ quantization helper formulas."""
    from fedmaq.core.strategy import compute_fedmaq_q_k_t

    # Test formulation 0: Resource-Only Hard Cap
    q_res = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=0,
        q_min=2,
        q_max=8,
    )
    # formulation 0 sets q_hat = q_max = 8. q_max_capped = floor(16384/2048) = 8. min(8, 8) = 8.
    assert q_res == 8

    q_res_capped = compute_fedmaq_q_k_t(
        c_k=4096.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=0,
        q_min=2,
        q_max=8,
    )
    # q_hat = 8, q_max_capped = 2. min(2, 8) = 2.
    assert q_res_capped == 2

    # Test formulation 1: Linear Sum
    q = compute_fedmaq_q_k_t(
        c_k=4096.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=1,
        q_min=2,
        q_max=8,
        gamma1=0.5,
        gamma2=0.5,
    )
    # tilde_g = 0.5, tilde_n = 0.5. term = 0.25 + 0.25 = 0.5. q_hat = 2 + round(6 * 0.5) = 5.
    # Q_max_capped = floor(4096/2048) = 2. So Q should be min(2, 5) = 2.
    assert q == 2

    # If c_k is large enough:
    q = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=1,
        q_min=2,
        q_max=8,
        gamma1=0.5,
        gamma2=0.5,
    )
    assert q == 5

    # Test formulation 2: Multiplicative
    q_mult = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=2,
        q_min=2,
        q_max=8,
        gamma1=0.5,
        gamma2=0.5,
    )
    # term = (0.5**0.5) * (0.5**0.5) = 0.5. q_hat = 2 + round(6 * 0.5) = 5. min(8, 5) = 5.
    assert q_mult == 5

    # Test formulation 3: Gradient-Primary, Data-Modulated
    q_mod = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=3,
        q_min=2,
        q_max=8,
        lambda_val=1.0,
    )
    # modulator = (1 + 1*0.5)/2 = 0.75. q_hat = 2 + round(6*0.5*0.75) = 2 + round(2.25) = 4.
    assert q_mod == 4

    # Test formulation 4: Threshold-Based Staged Rule
    # Case A: both thresholds cleared (tilde_g=0.5 >= 0.4, tilde_n=0.5 >= 0.4)
    q_th_a = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=4,
        q_min=2,
        q_max=8,
        tau_g=0.4,
        tau_n=0.4,
    )
    assert q_th_a == 8

    # Case B: one threshold cleared (tilde_g=0.5 >= 0.4, tilde_n=0.5 < 0.6)
    q_th_b = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=4,
        q_min=2,
        q_max=8,
        tau_g=0.4,
        tau_n=0.6,
    )
    # q_mid = round(10/2) = 5
    assert q_th_b == 5

    # Case C: neither threshold cleared (tilde_g=0.5 < 0.6, tilde_n=0.5 < 0.6)
    q_th_c = compute_fedmaq_q_k_t(
        c_k=16384.0,
        c_unit=2048.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=4,
        q_min=2,
        q_max=8,
        tau_g=0.6,
        tau_n=0.6,
    )
    assert q_th_c == 2


def test_fedmaq_q_k_t_snaps_to_permissible_bit_widths():
    """Manuscript §4.2's Q = {1,...,8,16,32} must always be respected, including
    when the soft-target range or memory cap would otherwise land off-set."""
    from fedmaq.core.strategy import compute_fedmaq_q_k_t
    from fedmaq.core.strategy_hooks.fedmaq import DEFAULT_BIT_WIDTHS

    # Soft target interpolates within [q_min, q_max] = [1, 20]; formulation 0 forces
    # q_hat = q_max = 20, which is not in Q and must floor to the largest member <= 20 (16).
    q = compute_fedmaq_q_k_t(
        c_k=1_000_000.0,  # effectively unconstrained memory
        c_unit=1.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=0,
        q_min=1,
        q_max=20,
    )
    assert q == 16
    assert q in DEFAULT_BIT_WIDTHS

    # A generous memory/c_unit ratio should be able to reach the 32-bit escape tier
    # once snapped down to the nearest permissible value <= the raw capacity ratio.
    q_capped = compute_fedmaq_q_k_t(
        c_k=40.0,
        c_unit=1.0,  # raw ratio = 40 -> nearest permissible value <= 40 is 32
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=0,
        q_min=1,
        q_max=32,
    )
    assert q_capped == 32

    # Every output across formulations/configs must always land in the permissible set.
    for formulation in range(5):
        result = compute_fedmaq_q_k_t(
            c_k=16384.0,
            c_unit=2048.0,
            g_k=0.7,
            g_max=1.0,
            n_k=150,
            n_max=200,
            formulation=formulation,
            q_min=2,
            q_max=8,
        )
        assert result in DEFAULT_BIT_WIDTHS


def test_fedmaq_q_k_t_floors_not_rounds_at_q_set_gap():
    """Regression: combine-then-floor-once must not round to the nearest Q member.

    q_hat=13 with an unconstrained memory cap: the old (buggy) nearest-snap logic
    picked 16 (|13-16|=3 < |13-8|=5), which can exceed a client's memory cap.
    The manuscript's floor(min(Q_k^max, q_hat)) semantics must instead give 8,
    the largest permissible member <= 13.
    """
    from fedmaq.core.strategy import compute_fedmaq_q_k_t

    q = compute_fedmaq_q_k_t(
        c_k=1_000_000.0,  # effectively unconstrained memory
        c_unit=1.0,
        g_k=0.5,
        g_max=1.0,
        n_k=100,
        n_max=200,
        formulation=0,
        q_min=1,
        q_max=13,
    )
    assert q == 8


def test_compute_dadaquant_client_q():
    """Test DAdaQuant client-adaptive quantization helper."""
    from fedmaq.core.strategy import compute_dadaquant_client_q

    sizes = [100, 200]
    q_t = 4
    q_is = compute_dadaquant_client_q(sizes, q_t)
    assert len(q_is) == 2
    # Client with larger dataset (index 1) should have greater or equal q_i.
    assert q_is[1] >= q_is[0]


def test_compute_dadaquant_client_q_clamps_to_range():
    """Per-client q_i must be clamped to [q_min, q_max] when bounds are supplied.

    Regression: a heavily skewed data share can drive sqrt(a/b)*w_i^(2/3) above
    the budget; without the upper clamp a client is assigned more levels than
    q_max allows.
    """
    from fedmaq.core.strategy import compute_dadaquant_client_q

    # Extreme skew: one dominant client would otherwise exceed q_max.
    sizes = [1, 10000]
    q_is = compute_dadaquant_client_q(sizes, q_t=4, q_min=2, q_max=8)
    assert all(2 <= q <= 8 for q in q_is), q_is
    # No upper bound by default (backward-compatible), only the floor of 1.
    q_is_unbounded = compute_dadaquant_client_q(sizes, q_t=4)
    assert all(q >= 1 for q in q_is_unbounded)


def test_network_simulator():
    """Test NetworkSimulator delay calculation."""
    import numpy as np

    from fedmaq.core.strategy import NetworkSimulator

    upload_bw = np.array([10.0])  # 10 Mbps
    download_bw = np.array([20.0])  # 20 Mbps
    comp_speed = np.array([100.0])  # 100 samples/sec

    sim = NetworkSimulator(upload_bw, download_bw, comp_speed, num_clients=1)

    t_download, t_train, t_upload = sim.simulate_client_delay(
        cid=0,
        model_size_bytes=1000000,  # 1 MB
        bytes_uploaded=500000,  # 0.5 MB
        train_sample_count=200 * 5,  # 200 samples * 5 epochs
        compute_scale=1.0,
    )

    # 1 MB download on 20 Mbps link:
    # 20 Mbps = 2.5 MB/s. 1 MB / 2.5 MB/s = 0.4s.
    assert abs(t_download - 0.4) < 1e-5

    # 0.5 MB upload on 10 Mbps link:
    # 10 Mbps = 1.25 MB/s. 0.5 MB / 1.25 MB/s = 0.4s.
    assert abs(t_upload - 0.4) < 1e-5

    # 200 samples * 5 epochs = 1000 samples. 1000 samples / 100 samples/sec = 10s.
    assert abs(t_train - 10.0) < 1e-5


def test_evaluation_metrics():
    """Test compute_precision_recall_f1 helper."""
    import numpy as np

    from fedmaq.core.evaluation import compute_precision_recall_f1

    all_preds = np.array([0, 1, 0, 1])
    all_labels = np.array([0, 1, 1, 0])
    precision, recall, f1 = compute_precision_recall_f1(all_preds, all_labels, num_classes=2)

    # Class 0: tp=1, fp=1, fn=1. prec=0.5, rec=0.5, f1=0.5.
    # Class 1: tp=1, fp=1, fn=1. prec=0.5, rec=0.5, f1=0.5.
    # Macro avg: prec=0.5, rec=0.5, f1=0.5.
    assert abs(precision - 0.5) < 1e-5
    assert abs(recall - 0.5) < 1e-5
    assert abs(f1 - 0.5) < 1e-5


def test_strategy_hook_registry():
    """get_strategy_hook: dict lookup, Passthrough fallback, CFD construction."""
    from fedmaq.core.strategy_hooks import (
        _UNPORTED,
        CFDHook,
        FedDistillHook,
        FedMAQHook,
        PassthroughHook,
        get_strategy_hook,
    )

    cfg = {"algorithm": {"name": "fedmaq"}}
    assert isinstance(get_strategy_hook("fedmaq", cfg), FedMAQHook)
    assert isinstance(get_strategy_hook("feddistill", {}), FedDistillHook)

    # Unknown FedAvg-family name falls back to PassthroughHook (client-only algos).
    assert isinstance(get_strategy_hook("fedavg", {}), PassthroughHook)

    # CFD is fully ported: constructs (no longer in _UNPORTED) with sane defaults.
    assert "cfd" not in _UNPORTED
    cfd_hook = get_strategy_hook("cfd", {"dataset": {"name": "mnist", "num_classes": 10}})
    assert isinstance(cfd_hook, CFDHook)
    assert cfd_hook.b_up == 1
    assert cfd_hook.b_down == 1


def test_feddistill_logit_tracker_no_nan():
    """LogitTracker must stay finite when a client is missing most classes.

    Under Dirichlet alpha=0.1 most clients hold only a few labels; counts are
    initialized to ones so unseen classes yield a finite all-zero row, not 0/0 NaN.
    """
    from fedmaq.core.client_hooks.feddistill import LogitTracker

    tracker = LogitTracker(num_labels=10)
    logits = torch.randn(6, 10)
    labels = torch.tensor([0, 0, 3, 3, 0, 3])  # only classes 0 and 3 present
    tracker.update(logits, labels)

    avg = tracker.avg()
    assert avg.shape == (10, 10)
    assert np.all(np.isfinite(avg)), "LogitTracker produced NaN/Inf"
    for missing in (1, 2, 4, 5, 6, 7, 8, 9):
        assert np.allclose(avg[missing], 0.0), f"class {missing} row should be zero"


def test_feddistill_bytes_shape_guard():
    """Deserializing a logit buffer with the wrong num_labels must fail loudly."""
    from fedmaq.core.client_hooks.feddistill import bytes_to_logits, logits_to_bytes

    buf = logits_to_bytes(np.zeros((3, 3), dtype=np.float32))
    assert bytes_to_logits(buf, 3).shape == (3, 3)
    with pytest.raises(ValueError, match="num_labels"):
        bytes_to_logits(buf, 4)  # 9 floats != 4^2


def test_feddistill_hook_aggregation_and_broadcast():
    """Server averages client logit matrices, passes weights through, rebroadcasts."""
    from flwr.common import Code, FitIns, Status, ndarrays_to_parameters
    from flwr.common.typing import FitRes

    from fedmaq.core.client_hooks.feddistill import bytes_to_logits, logits_to_bytes
    from fedmaq.core.strategy_hooks.feddistill import FedDistillHook

    hook = FedDistillHook({"dataset": {"num_classes": 3}})
    assert hook.global_logits is None

    def _fit_res(matrix):
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([]),
            num_examples=1,
            metrics={"client_logits": logits_to_bytes(matrix)},
        )

    m1 = np.ones((3, 3), dtype=np.float32)
    m2 = np.full((3, 3), 3.0, dtype=np.float32)
    results = [(None, _fit_res(m1)), (None, _fit_res(m2))]

    sentinel = ndarrays_to_parameters([np.zeros(2, dtype=np.float32)])
    out_params, _ = hook.aggregate_fit(None, 1, results, [], sentinel, {})
    # FedAvg-averaged weights are passed through untouched.
    assert out_params is sentinel
    # Consensus logits = elementwise mean of the client matrices.
    assert np.allclose(hook.global_logits, 2.0)
    assert np.all(np.isfinite(hook.global_logits))

    # configure_fit broadcasts the consensus matrix as bytes.
    fit_ins = FitIns(ndarrays_to_parameters([]), {})
    updated = hook.configure_fit(None, 2, ndarrays_to_parameters([]), None, [(None, fit_ins)])
    gl = updated[0][1].config["global_logits"]
    assert isinstance(gl, bytes)
    assert np.allclose(bytes_to_logits(gl, 3), 2.0)


def test_feddistill_two_round_reg_path(mock_dataset):
    """End-to-end 2-round FedDistill+: round 2 must exercise the logit-KD reg branch.

    Round 1 has no global logits (plain CE); only after the server aggregates and
    rebroadcasts does the KLDiv(log_softmax(z), softmax(global_logits[y])) term fire.
    """
    from flwr.common import Code, Status, ndarrays_to_parameters
    from flwr.common.typing import FitRes

    from fedmaq.core.client_hooks.feddistill import logits_to_bytes
    from fedmaq.core.models import SimpleCNN, get_model_parameters
    from fedmaq.core.strategy_hooks.feddistill import FedDistillHook

    train_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=4)
    model = SimpleCNN(in_channels=1, num_classes=10)
    params = get_model_parameters(model)
    cfg = {
        "experiment": {"local_epochs": 1, "learning_rate": 0.01, "weight_decay": 0.0},
        "algorithm": {"name": "feddistill", "reg_alpha": 1.0},
        "dataset": {"name": "mnist", "num_classes": 10},
    }
    client = GenericClient(
        cid="0",
        trainloader=train_loader,
        testloader=train_loader,
        model=model,
        loss_hook=LossHook(),
        compressor_hook=CompressionHook(),
        config=cfg,
    )

    # Round 1: no global logits -> plain CE, but still emits per-class logits.
    p1, n1, m1 = client.fit(params, {"server_round": 1})
    assert isinstance(m1["client_logits"], bytes)
    assert all(np.all(np.isfinite(p)) for p in p1)

    # Server aggregates -> global logits become available.
    hook = FedDistillHook({"dataset": {"num_classes": 10}})
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(p1),
        num_examples=n1,
        metrics=m1,
    )
    hook.aggregate_fit(None, 1, [(None, fit_res)], [], ndarrays_to_parameters(p1), {})
    assert hook.global_logits is not None
    assert np.all(np.isfinite(hook.global_logits))

    # Round 2: broadcasting the global logits must drive the reg path without error.
    p2, _, m2 = client.fit(
        p1, {"server_round": 2, "global_logits": logits_to_bytes(hook.global_logits)}
    )
    assert all(np.all(np.isfinite(p)) for p in p2)
    # Upload accounting = model weights + the 10x10 float32 logit matrix.
    assert m2["bytes_uploaded"] == sum(int(p.nbytes) for p in p2) + 10 * 10 * 4
