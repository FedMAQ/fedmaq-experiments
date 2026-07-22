"""Phase 0 safety-net tests: Hydra config composition + in-process run(cfg) smoke.

These guard the orchestration path that the hand-rolled unit tests in
``test_environment.py`` never exercise: that every ``conf/algorithm/*.yaml`` composes
into a valid config, and that the decorator-free :func:`fedmaq.simulation.run` entry
point drives the real ``client_fn``/``server_fn`` wiring end-to-end.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import TensorDataset

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())

# Every selectable algorithm config (including the FedDistill/CFD stubs, whose YAML
# must still compose even though their hooks are not yet implemented).
ALGORITHM_CONFIGS = [
    "fedavg",
    "fedprox",
    "fedpaq",
    "dadaquant",
    "fedmd",
    "fedkd",
    "fedavg_kd",
    "fedmaq",
    "fedmaq_no_kd",
    "fedmaq_state_only",
    "fedmaq_data_only",
    "fedmaq_resource_only",
    "feddistill",
    "cfd",
]


@pytest.fixture
def mock_dataset(monkeypatch):
    """Mock torchvision dataset loading with 100 MNIST-like samples."""
    mock_data = torch.randn(100, 1, 28, 28)
    mock_labels = torch.randint(0, 10, (100,))
    mock_ds = TensorDataset(mock_data, mock_labels)
    mock_ds.targets = mock_labels
    monkeypatch.setattr("fedmaq.core.partitioning.load_dataset", lambda name, train=True: mock_ds)
    return mock_ds


@pytest.mark.parametrize("algorithm", ALGORITHM_CONFIGS)
def test_algorithm_config_composes(algorithm):
    """Every algorithm config must compose into a structurally valid experiment config.

    A malformed ``conf/algorithm/*.yaml`` (or a broken default/interpolation) is
    otherwise invisible to the suite, since the unit tests build cfg dicts inline.
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"algorithm={algorithm}"])

    # Composition wiring: the four config groups + a resolvable algorithm name.
    assert cfg.algorithm.name, f"{algorithm} config is missing algorithm.name"
    assert cfg.dataset.name
    assert cfg.experiment.num_clients > 0
    assert cfg.experiment.total_rounds > 0
    # Manuscript Table 4.1 anchors that must survive composition.
    assert cfg.experiment.batch_size == 64
    assert cfg.experiment.num_public_samples == 3000


def test_run_cfg_smoke_fedavg(mock_dataset, tmp_path, monkeypatch):
    """The extracted run(cfg) entry point drives a real 1-round simulation in-process.

    Unlike a subprocess ``scripts/run.py`` smoke (which only checks an exit code),
    this asserts on returned telemetry state, making orchestration regressions and
    empty-payload bugs observable.
    """
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    from fedmaq.simulation import run

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=fedavg",
                # Route to the lightweight SimpleCNN path so the 1x28x28 mock fits.
                "dataset.name=mnist",
                "dataset.num_classes=10",
                "experiment.num_clients=2",
                "experiment.total_rounds=1",
                "experiment.num_public_samples=10",
                "experiment.batch_size=2",
                "experiment.local_epochs=1",
                "experiment.client_fraction=1.0",
                "experiment.client_gpus=0.0",
            ],
        )

    telemetry = run(cfg)

    # The run completed and accounted for transmitted bytes over the wire.
    assert telemetry.cumulative_bytes > 0
    assert telemetry.jsonl_path.exists()
    assert np.isfinite(telemetry.cumulative_bytes)


def test_run_cfg_smoke_feddistill_two_rounds(mock_dataset, tmp_path, monkeypatch):
    """FedDistill+ over 2 rounds through the real Flower orchestration.

    Validates the bytes-over-Flower transport that the in-process unit tests bypass:
    clients emit per-class logits in FitRes.metrics, the server averages and
    rebroadcasts them via FitIns.config, and round 2 runs the logit-KD reg path.
    """
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    from fedmaq.simulation import run

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=feddistill",
                "dataset.name=mnist",
                "dataset.num_classes=10",
                "experiment.num_clients=2",
                "experiment.total_rounds=2",
                "experiment.num_public_samples=10",
                "experiment.batch_size=2",
                "experiment.local_epochs=1",
                "experiment.client_fraction=1.0",
                "experiment.client_gpus=0.0",
            ],
        )

    telemetry = run(cfg)

    assert telemetry.cumulative_bytes > 0
    assert np.isfinite(telemetry.cumulative_bytes)


def test_run_cfg_smoke_cfd_two_rounds(mock_dataset, tmp_path, monkeypatch):
    """CFD over 2 rounds through the real Flower orchestration.

    Validates the soft-label transport the in-process unit tests bypass: clients
    return quantized codes as ``parameters`` (not weights), the server dequantizes
    + averages + dual-distills its persistent server_model, and round 2 broadcasts
    quantized server labels so the client-side digest (KL) branch engages.
    """
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    from fedmaq.simulation import run

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=cfd",
                "dataset.name=mnist",
                "dataset.num_classes=10",
                "experiment.num_clients=2",
                "experiment.total_rounds=2",
                "experiment.num_public_samples=10",
                "experiment.batch_size=2",
                "experiment.local_epochs=1",
                "experiment.client_fraction=1.0",
                "experiment.client_gpus=0.0",
            ],
        )

    telemetry = run(cfg)

    # Round 1 has no downstream broadcast (untrained server model); round 2 does,
    # so cumulative bytes must still be positive and finite overall.
    assert telemetry.cumulative_bytes > 0
    assert np.isfinite(telemetry.cumulative_bytes)
