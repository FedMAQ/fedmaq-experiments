"""Golden characterization tests for the simulated time/bytes model.

Simulated runtime and communication overhead are logged thesis metrics
(``evaluation-metrics.md``). Phase 3 of the refactor relocated the algorithm-specific
magic numbers baked into ``NetworkSimulator`` (FedKD compute penalty ``2.5``; FedMD
round-1 pretraining ``10`` epochs; server-side KD speed ``2000.0``) behind
``StrategyHook`` methods / config keys.

These tests pin the exact numeric behavior of the *composed* algorithm -> number
facts: they drive ``train_sample_count`` / ``compute_scale`` through the real hook
methods into ``NetworkSimulator`` and assert the same delays the pre-refactor
``simulate_client_delay(..., alg_name=...)`` produced. If any golden number shifts,
that is the drift these tests exist to catch -- do not edit the expected values.
"""

import numpy as np

from fedmaq.core.kd_utils import kd_server_sim_time
from fedmaq.core.strategy import NetworkSimulator
from fedmaq.core.strategy_hooks import (
    CFDHook,
    FedKDHook,
    FedMDHook,
    PassthroughHook,
)

# Fixed link/compute so every delay below is a deterministic golden number.
UPLOAD_BW = np.array([10.0])  # Mbps -> 1.25 MB/s
DOWNLOAD_BW = np.array([20.0])  # Mbps -> 2.5 MB/s
COMP_SPEED = np.array([100.0])  # samples/sec


def _sim() -> NetworkSimulator:
    return NetworkSimulator(UPLOAD_BW, DOWNLOAD_BW, COMP_SPEED, num_clients=1)


def _delay(hook, num_samples, epochs, num_public, public_epochs, server_round):
    """Drive a hook's time-model contributions through NetworkSimulator."""
    train_sample_count = hook.local_train_sample_count(
        num_samples=num_samples,
        epochs=epochs,
        num_public=num_public,
        public_epochs=public_epochs,
        server_round=server_round,
    )
    return _sim().simulate_client_delay(
        cid=0,
        model_size_bytes=1_000_000,
        bytes_uploaded=500_000,
        train_sample_count=train_sample_count,
        compute_scale=hook.compute_speed_scale(),
    )


def test_fedkd_compute_penalty_golden():
    """FedKD scales effective compute speed by the 2.5x dual-model penalty."""
    hook = FedKDHook({"algorithm": {}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=1)
    # 200 samples * 5 epochs = 1000; effective speed = 100 / 2.5 = 40 -> 25.0 s.
    assert t_train == 25.0


def test_fedavg_no_compute_penalty_golden():
    """Baseline (PassthroughHook, no penalty) train time, as a contrast."""
    _, t_train, _ = _delay(PassthroughHook(), 200, 5, 200, 5, server_round=1)
    # 1000 samples / 100 samples/sec = 10.0 s.
    assert t_train == 10.0


def test_fedmd_round1_pretrain_golden():
    """FedMD round 1 folds in 10+10 pretraining epochs on public+private data."""
    hook = FedMDHook({"algorithm": {}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=1)
    # (200*10 + 200*10 + 200*5 + 200*5) / 100 = 6000 / 100 = 60.0 s.
    assert t_train == 60.0


def test_fedmd_round2_no_pretrain_golden():
    """FedMD round >1 drops the one-time pretraining cost."""
    hook = FedMDHook({"algorithm": {}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=2)
    # (200*5 + 200*5) / 100 = 2000 / 100 = 20.0 s.
    assert t_train == 20.0


def test_cfd_round1_no_digest_golden():
    """CFD round 1: no downstream labels yet, so no digest-phase compute."""
    hook = CFDHook({"dataset": {"name": "mnist", "num_classes": 4}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=1)
    # 200 samples * 5 epochs / 100 samples/sec = 10.0 s (private-only).
    assert t_train == 10.0


def test_cfd_round2_adds_digest_phase_golden():
    """CFD round >1 folds in the client digest phase (distill_epochs on public set)."""
    hook = CFDHook(
        {"dataset": {"name": "mnist", "num_classes": 4}, "algorithm": {"distill_epochs": 2}}
    )
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=2)
    # (200*5 + 200*2) / 100 = 1400 / 100 = 14.0 s.
    assert t_train == 14.0


def test_server_kd_sim_time_golden():
    """Server-side KD delay: proxy_size * kd_epochs * teachers / server_speed."""
    # 200 public * 1 epoch * 5 teachers / 2000 samples/sec = 0.5 s.
    assert kd_server_sim_time(
        num_public=200, kd_epochs=1, num_teachers=5, server_compute_speed=2000.0
    ) == 0.5


def test_transmission_delays_algorithm_independent_golden():
    """Download/upload delays depend only on bytes and bandwidth, not the algorithm."""
    for hook in (PassthroughHook(), FedKDHook({"algorithm": {}}), FedMDHook({})):
        t_download, _, t_upload = _delay(hook, 200, 5, 200, 5, server_round=2)
        assert abs(t_download - 0.4) < 1e-9
        assert abs(t_upload - 0.4) < 1e-9
