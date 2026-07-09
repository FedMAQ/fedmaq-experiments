"""Golden characterization tests for the simulated time/bytes model.

Simulated runtime and communication overhead are logged thesis metrics
(``evaluation-metrics.md``). Phase 3 of the refactor relocates the algorithm-specific
magic numbers baked into ``NetworkSimulator`` (FedKD compute penalty ``2.5``; FedMD
round-1 pretraining ``10`` epochs) behind hook methods / config keys. These tests pin
the exact numeric behavior so any drift after that refactor fails loudly rather than
silently corrupting a reported metric.

The server-side KD speed (``2000.0``) is pinned in Phase 3 against the extracted
``server_sim_time`` hook seam (it does not yet exist as an isolated unit).
"""

import numpy as np

from fedmaq.core.strategy import NetworkSimulator

# Fixed link/compute so every delay below is a deterministic golden number.
UPLOAD_BW = np.array([10.0])  # Mbps -> 1.25 MB/s
DOWNLOAD_BW = np.array([20.0])  # Mbps -> 2.5 MB/s
COMP_SPEED = np.array([100.0])  # samples/sec


def _sim() -> NetworkSimulator:
    return NetworkSimulator(UPLOAD_BW, DOWNLOAD_BW, COMP_SPEED, num_clients=1)


def test_fedkd_compute_penalty_golden():
    """FedKD halves effective compute speed by the 2.5x dual-model penalty."""
    _, t_train, _ = _sim().simulate_client_delay(
        cid=0,
        model_size_bytes=1_000_000,
        bytes_uploaded=500_000,
        num_samples=200,
        epochs=5,
        alg_name="fedkd",
    )
    # 200 samples * 5 epochs = 1000; effective speed = 100 / 2.5 = 40 -> 25.0 s.
    assert t_train == 25.0


def test_fedavg_no_compute_penalty_golden():
    """Baseline (no penalty) train time, as a contrast to the FedKD case."""
    _, t_train, _ = _sim().simulate_client_delay(
        cid=0,
        model_size_bytes=1_000_000,
        bytes_uploaded=500_000,
        num_samples=200,
        epochs=5,
        alg_name="fedavg",
    )
    # 1000 samples / 100 samples/sec = 10.0 s.
    assert t_train == 10.0


def test_fedmd_round1_pretrain_golden():
    """FedMD round 1 folds in 10+10 pretraining epochs on public+private data."""
    _, t_train, _ = _sim().simulate_client_delay(
        cid=0,
        model_size_bytes=1_000_000,
        bytes_uploaded=500_000,
        num_samples=200,
        epochs=5,
        alg_name="fedmd",
        public_epochs=5,
        num_public=200,
        server_round=1,
    )
    # (200*10 + 200*10 + 200*5 + 200*5) / 100 = 6000 / 100 = 60.0 s.
    assert t_train == 60.0


def test_fedmd_round2_no_pretrain_golden():
    """FedMD round >1 drops the one-time pretraining cost."""
    _, t_train, _ = _sim().simulate_client_delay(
        cid=0,
        model_size_bytes=1_000_000,
        bytes_uploaded=500_000,
        num_samples=200,
        epochs=5,
        alg_name="fedmd",
        public_epochs=5,
        num_public=200,
        server_round=2,
    )
    # (200*5 + 200*5) / 100 = 2000 / 100 = 20.0 s.
    assert t_train == 20.0


def test_transmission_delays_algorithm_independent_golden():
    """Download/upload delays depend only on bytes and bandwidth, not the algorithm."""
    for alg in ("fedavg", "fedkd", "fedmd"):
        t_download, _, t_upload = _sim().simulate_client_delay(
            cid=0,
            model_size_bytes=1_000_000,  # 1 MB / 2.5 MB/s = 0.4 s
            bytes_uploaded=500_000,  # 0.5 MB / 1.25 MB/s = 0.4 s
            num_samples=200,
            epochs=5,
            alg_name=alg,
        )
        assert abs(t_download - 0.4) < 1e-9
        assert abs(t_upload - 0.4) < 1e-9
