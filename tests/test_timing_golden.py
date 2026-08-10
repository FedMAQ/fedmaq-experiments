"""Golden tests for simulated time and byte accounting."""

import numpy as np

from fedmaq.core.kd_utils import kd_server_sim_time
from fedmaq.core.strategy import PhysicalCostModel
from fedmaq.core.strategy_hooks import (
    CFDHook,
    FedKDHook,
    FedMDHook,
    PassthroughHook,
)

UPLOAD_BW = np.array([10.0])  # Mbps -> 1.25 MB/s
DOWNLOAD_BW = np.array([20.0])  # Mbps -> 2.5 MB/s
COMP_SPEED = np.array([100.0])  # samples/sec


def _sim() -> PhysicalCostModel:
    return PhysicalCostModel(UPLOAD_BW, DOWNLOAD_BW, COMP_SPEED, num_clients=1)


def _delay(hook, num_samples, epochs, num_public, public_epochs, server_round):
    """Drive a hook's time-model contributions through PhysicalCostModel."""
    train_sample_count = hook.local_train_sample_count(
        num_samples=num_samples,
        epochs=epochs,
        num_public=num_public,
        public_epochs=public_epochs,
        server_round=server_round,
    )
    return _sim().client_round_delay(
        cid=0,
        model_size_bytes=1_000_000,
        bytes_uploaded=500_000,
        train_sample_count=train_sample_count,
        compute_scale=hook.compute_speed_scale(),
    )


def test_fedkd_compute_penalty_golden():
    """FedKD scales effective compute speed by the 1.3x dual-model penalty."""
    hook = FedKDHook({"algorithm": {}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=1)
    assert t_train == 13.0


def test_fedavg_no_compute_penalty_golden():
    """Baseline (PassthroughHook, no penalty) train time, as a contrast."""
    _, t_train, _ = _delay(PassthroughHook(), 200, 5, 200, 5, server_round=1)
    assert t_train == 10.0


def test_fedmd_round1_pretrain_golden():
    """FedMD round 1 folds in 10+10 pretraining epochs on public+private data."""
    hook = FedMDHook({"algorithm": {}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=1)
    assert t_train == 60.0


def test_fedmd_round2_no_pretrain_golden():
    """FedMD round >1 drops the one-time pretraining cost."""
    hook = FedMDHook({"algorithm": {}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=2)
    assert t_train == 20.0


def test_cfd_round1_no_digest_golden():
    """CFD round 1: no downstream labels yet, so no digest-phase compute."""
    hook = CFDHook({"dataset": {"name": "mnist", "num_classes": 4}})
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=1)
    assert t_train == 10.0


def test_cfd_round2_adds_digest_phase_golden():
    """CFD round >1 folds in the client digest phase (distill_epochs on public set)."""
    hook = CFDHook(
        {"dataset": {"name": "mnist", "num_classes": 4}, "algorithm": {"distill_epochs": 2}}
    )
    _, t_train, _ = _delay(hook, 200, 5, 200, 5, server_round=2)
    assert t_train == 14.0


def test_server_kd_sim_time_golden():
    """Server-side KD delay: proxy_size * kd_epochs * teachers / server_speed."""
    assert (
        kd_server_sim_time(num_public=200, kd_epochs=1, num_teachers=5, server_compute_speed=2000.0)
        == 0.5
    )


def test_transmission_delays_algorithm_independent_golden():
    """Download/upload delays depend only on bytes and bandwidth, not the algorithm."""
    for hook in (PassthroughHook(), FedKDHook({"algorithm": {}}), FedMDHook({})):
        t_download, _, t_upload = _delay(hook, 200, 5, 200, 5, server_round=2)
        assert abs(t_download - 0.4) < 1e-9
        assert abs(t_upload - 0.4) < 1e-9
