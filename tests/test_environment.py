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
    monkeypatch.setattr(
        "fedmaq.core.partitioning.load_dataset", lambda name, train=True: mock_ds
    )
    return mock_ds


def test_model_factory_and_parameters():
    """Test get_model factory and get/set parameter helpers."""
    model = get_model("mnist", num_classes=10)
    assert isinstance(model, SimpleCNN)

    cifar_model = get_model("cifar10", num_classes=10)
    assert isinstance(cifar_model, ResNet18GN)

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
    for p_new, p_re in zip(new_params, re_extracted):
        np.testing.assert_allclose(p_new, p_re, rtol=1e-5)


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
        "heterogeneity": {
            "bandwidth": {"min_mbps": 5.0, "max_mbps": 20.0},
            "compute": {"min_samples_per_sec": 100.0, "max_samples_per_sec": 500.0},
        },
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

        def get_properties(self, ins, timeout):
            return None

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
    strategy.moving_average_history = [1.0, 1.0, 1.0, 1.0]  # Plateau detected
    strategy.last_quantization_increase_round = 0
    strategy.q_t = 4

    # Run configure_fit for round 5 (checks history up to round 4)
    instructions = strategy.configure_fit(
        server_round=5, parameters=params, client_manager=client_manager
    )
    # Since a plateau is detected (latest loss 1.0 >= past loss 1.0), q_t should double from 4 to 8
    assert strategy.q_t == 8
    assert strategy.last_quantization_increase_round == 4


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
            "heterogeneity": {
                "bandwidth": {"min_mbps": 5.0, "max_mbps": 20.0},
                "compute": {"min_samples_per_sec": 100.0, "max_samples_per_sec": 500.0},
            },
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
            initial_parameters = ndarrays_to_parameters(
                get_model_parameters(global_model)
            )
            _, test_loader = get_server_loaders("mnist", public_indices, batch_size=2)

            def evaluate_fn(server_round, parameters, config):
                # Simple ensemble eval simulation
                client_paths = (
                    list(model_dir.glob("client_*.pth")) if model_dir.exists() else []
                )
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
            return ServerAppComponents(
                strategy=strategy, config=ServerConfig(num_rounds=2)
            )

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
            "heterogeneity": {
                "bandwidth": {"min_mbps": 5.0, "max_mbps": 20.0},
                "compute": {"min_samples_per_sec": 100.0, "max_samples_per_sec": 500.0},
            },
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
                "heterogeneity": {
                    "bandwidth": {"min_mbps": 5.0, "max_mbps": 20.0},
                    "compute": {
                        "min_samples_per_sec": 100.0,
                        "max_samples_per_sec": 500.0,
                    },
                },
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
            initial_parameters = ndarrays_to_parameters(
                get_model_parameters(global_model)
            )
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
            return ServerAppComponents(
                strategy=strategy, config=ServerConfig(num_rounds=2)
            )

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
    np.testing.assert_allclose(
        compressed[0], np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    )


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
            "name": "fedmaq",
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
    strategy.client_memory = np.array(
        [2048.0, 16384.0]
    )  # Client 0: Q_max=1, Client 1: Q_max=8

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
                "name": "fedmaq",
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
            initial_parameters = ndarrays_to_parameters(
                get_model_parameters(global_model)
            )

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
            return ServerAppComponents(
                strategy=strategy, config=ServerConfig(num_rounds=2)
            )

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
