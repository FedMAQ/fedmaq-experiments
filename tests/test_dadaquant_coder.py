"""Tests for DAdaQuant's as-published secondary byte axis (#26).

Round-trip coverage on the coder itself ("this is a coder, not a heuristic"),
plus the hook/fit/telemetry plumbing that logs the secondary total alongside
the primary one without folding the two axes together.
"""

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from torch.utils.data import TensorDataset

from fedmaq.baselines.dadaquant_coder import (
    _BitReader,
    _BitWriter,
    _elias_omega_decode_from,
    _elias_omega_encode_into,
    dadaquant_pack,
    dadaquant_unpack,
)
from fedmaq.baselines.quantization import DAdaQuantCompressionHook, FedPAQCompressionHook
from fedmaq.core.client import GenericClient, LossHook
from fedmaq.core.models import SimpleCNN, get_model_parameters
from fedmaq.core.payload_archive import PayloadArchive
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager

# --- Elias omega primitive: round-trip over positive integers ---


def _elias_roundtrip(n: int) -> int:
    writer = _BitWriter()
    _elias_omega_encode_into(writer, n)
    reader = _BitReader(writer.to_bytes())
    return _elias_omega_decode_from(reader)


def test_elias_omega_roundtrips_small_integers():
    for n in range(1, 2000):
        assert _elias_roundtrip(n) == n


def test_elias_omega_roundtrips_large_and_random_integers():
    rng = np.random.default_rng(0)
    for n in rng.integers(1, 2**40, size=500).tolist():
        assert _elias_roundtrip(int(n)) == n


def test_elias_omega_rejects_nonpositive_input():
    writer = _BitWriter()
    with pytest.raises(ValueError, match="n >= 1"):
        _elias_omega_encode_into(writer, 0)


def test_elias_omega_known_encodings_match_hand_derivation():
    # Hand-verified against the standard Elias omega definition (Wikipedia):
    # 1 -> "0", 2 -> "100", 3 -> "110", 4 -> "101000".
    cases = {1: "0", 2: "100", 3: "110", 4: "101000"}
    for n, expected_bits in cases.items():
        writer = _BitWriter()
        _elias_omega_encode_into(writer, n)
        produced = writer.to_bytes()
        bit_str = "".join(f"{byte:08b}" for byte in produced)[: len(expected_bits)]
        assert bit_str == expected_bits


# --- dadaquant_pack/unpack: round-trip over full code arrays ---


def _codes_roundtrip(codes: np.ndarray) -> np.ndarray:
    packed = dadaquant_pack(codes)
    return dadaquant_unpack(packed, codes.size)


def test_pack_unpack_roundtrips_random_sparse_codes():
    rng = np.random.default_rng(1)
    for _ in range(200):
        size = int(rng.integers(1, 500))
        q = int(rng.integers(1, 32))
        # Skew toward zero, matching stochastic-rounded quantization codes.
        codes = np.where(rng.random(size) < 0.8, 0, rng.integers(-q, q + 1, size=size)).astype(
            np.int64
        )
        assert np.array_equal(_codes_roundtrip(codes), codes)


def test_pack_unpack_roundtrips_dense_codes():
    rng = np.random.default_rng(2)
    codes = rng.integers(-127, 128, size=1000).astype(np.int64)
    assert np.array_equal(_codes_roundtrip(codes), codes)


def test_pack_unpack_roundtrips_all_zero_codes():
    codes = np.zeros(50, dtype=np.int64)
    assert np.array_equal(_codes_roundtrip(codes), codes)


def test_pack_unpack_roundtrips_all_nonzero_codes():
    codes = np.arange(1, 51, dtype=np.int64)
    assert np.array_equal(_codes_roundtrip(codes), codes)


def test_pack_unpack_roundtrips_single_element_arrays():
    for value in (0, 1, -1, 127, -127):
        codes = np.array([value], dtype=np.int64)
        assert np.array_equal(_codes_roundtrip(codes), codes)


def test_pack_unpack_roundtrips_trailing_and_leading_zero_runs():
    codes = np.array([0, 0, 0, 5, 0, 0, -3, 0, 0, 0, 0], dtype=np.int64)
    assert np.array_equal(_codes_roundtrip(codes), codes)


def test_pack_unpack_roundtrips_empty_array():
    codes = np.zeros(0, dtype=np.int64)
    assert dadaquant_pack(codes) == b""
    assert dadaquant_unpack(b"", 0).size == 0


def test_pack_flattens_multidimensional_arrays_consistently():
    rng = np.random.default_rng(3)
    codes_2d = rng.integers(-8, 9, size=(20, 15)).astype(np.int64)
    roundtripped = dadaquant_unpack(dadaquant_pack(codes_2d), codes_2d.size)
    assert np.array_equal(roundtripped, codes_2d.ravel())


def test_sparse_codes_pack_smaller_than_dense_raw_encoding():
    """Sanity check that the coder actually exploits sparsity (the mechanism
    #26 exists to measure), not just that it round-trips."""
    rng = np.random.default_rng(4)
    size = 5000
    codes = np.where(rng.random(size) < 0.95, 0, rng.integers(1, 9, size=size)).astype(np.int64)
    packed_len = len(dadaquant_pack(codes))
    raw_len = codes.astype(np.int64).nbytes
    assert packed_len < raw_len


# --- Hook integration: only DAdaQuant reports a secondary total ---


def test_dadaquant_hook_report_contains_secondary_bytes():
    hook = DAdaQuantCompressionHook(q=4, rng=np.random.default_rng(0))
    _, report = hook.compress([np.ones((100,), dtype=np.float32)])
    assert isinstance(report.secondary_bytes, int)
    assert report.secondary_bytes > 0


def test_dadaquant_hook_secondary_bytes_covers_all_zero_tensor():
    """All-zero tensors skip quantization but still contribute a (tiny, since
    it's one long zero run) secondary payload -- matching the primary axis's
    coverage of the same branch (#25)."""
    hook = DAdaQuantCompressionHook(q=4, rng=np.random.default_rng(0))
    _, report = hook.compress([np.zeros((1000,), dtype=np.float32)])
    assert report.secondary_bytes is not None
    assert report.secondary_bytes < 20


def test_fedpaq_hook_report_has_no_secondary_bytes():
    """FedPAQ's source paper specifies no transport coder (2026-08-26 audit) --
    only DAdaQuant gets a secondary axis."""
    hook = FedPAQCompressionHook(q=8, rng=np.random.default_rng(0))
    _, report = hook.compress([np.array([-2.0, 0.0, 2.0], dtype=np.float32)])
    assert report.secondary_bytes is None


# --- Client fit integration ---


def _mnist_loader():
    data = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    return torch.utils.data.DataLoader(TensorDataset(data, labels), batch_size=4)


def _make_client(alg_cfg: dict, compressor_hook) -> GenericClient:
    model = SimpleCNN(in_channels=1, num_classes=10)
    loader = _mnist_loader()
    cfg = {
        "seed": 42,
        "experiment": {"local_epochs": 1, "learning_rate": 0.01, "weight_decay": 0.0},
        "algorithm": alg_cfg,
        "dataset": {"name": "mnist", "num_classes": 10},
    }
    return GenericClient(
        cid="0",
        trainloader=loader,
        testloader=loader,
        model=model,
        loss_hook=LossHook(),
        compressor_hook=compressor_hook,
        config=cfg,
    )


def test_dadaquant_fit_reports_secondary_bytes_uploaded():
    # "dadaquant" resolves to DAdaQuantFit via the alg-name registry
    # (client_hooks/__init__.py) -- not set explicitly here.
    client = _make_client(
        {"name": "dadaquant", "q_min": 1, "q_max": 8}, DAdaQuantCompressionHook(q=4)
    )
    params = get_model_parameters(client.model)

    _, _, fit_metrics = client.fit(params, {"server_round": 1})

    assert fit_metrics["secondary_bytes_uploaded"] > 0
    assert fit_metrics["secondary_bytes_uploaded"] > 0


def test_fedpaq_fit_omits_secondary_bytes_uploaded():
    client = _make_client({"name": "fedpaq", "q": 8}, FedPAQCompressionHook(q=8))
    params = get_model_parameters(client.model)

    _, _, fit_metrics = client.fit(params, {"server_round": 1})

    assert "secondary_bytes_uploaded" not in fit_metrics


# --- TelemetryManager.record_fit_round: aggregation ---


class _FakeProxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


def _make_strategy(tmp_path, monkeypatch) -> tuple[TelemetryFedAvg, TelemetryManager]:
    monkeypatch.setattr("fedmaq.core.telemetry._HYDRA_AVAILABLE", False)
    cfg_dict = {
        "experiment": {
            "num_clients": 2,
            "client_fraction": 1.0,
            "total_rounds": 1,
            "num_public_samples": 3000,
            "telemetry": {"wandb_enabled": False},
        },
        "algorithm": {"name": "dadaquant"},
    }
    tm = TelemetryManager(cfg_dict)
    tm.log_dir = tmp_path
    tm.jsonl_path = tmp_path / "experiment_log.jsonl"
    tm.csv_path = tmp_path / "experiment_log.csv"
    tm.payload_archive = PayloadArchive(tmp_path)
    strategy = TelemetryFedAvg(
        telemetry_manager=tm,
        config=cfg_dict,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    return strategy, tm


def _fit_res(cid: int, bytes_uploaded: int, secondary_bytes: int | None) -> FitRes:
    metrics: dict = {
        "partition_id": cid,
        "bytes_uploaded": bytes_uploaded,
        "payload_bytes": bytes_uploaded,
    }
    if secondary_bytes is not None:
        metrics["secondary_bytes_uploaded"] = secondary_bytes
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=10,
        metrics=metrics,
    )


def test_record_fit_round_sums_secondary_bytes_across_clients(tmp_path, monkeypatch):
    strategy, tm = _make_strategy(tmp_path, monkeypatch)
    results = [
        (_FakeProxy("0"), _fit_res(0, bytes_uploaded=100, secondary_bytes=30)),
        (_FakeProxy("1"), _fit_res(1, bytes_uploaded=200, secondary_bytes=50)),
    ]

    tm.record_fit_round(strategy, server_round=1, results=results, aggregated_parameters=None)

    snapshot = tm.snapshot_for_round(1)
    assert snapshot.round_secondary_bytes == 80


def test_record_fit_round_secondary_bytes_none_when_unreported(tmp_path, monkeypatch):
    strategy, tm = _make_strategy(tmp_path, monkeypatch)
    results = [
        (_FakeProxy("0"), _fit_res(0, bytes_uploaded=100, secondary_bytes=None)),
        (_FakeProxy("1"), _fit_res(1, bytes_uploaded=200, secondary_bytes=None)),
    ]

    tm.record_fit_round(strategy, server_round=1, results=results, aggregated_parameters=None)

    snapshot = tm.snapshot_for_round(1)
    assert snapshot.round_secondary_bytes is None


def test_record_fit_round_secondary_bytes_not_averaged_into_client_metrics(tmp_path, monkeypatch):
    """secondary_bytes_uploaded must be excluded from the generic numeric-key
    auto-average path (like bytes_uploaded/payload_bytes) -- it's a round
    total, not a per-client value to average."""
    strategy, tm = _make_strategy(tmp_path, monkeypatch)
    results = [
        (_FakeProxy("0"), _fit_res(0, bytes_uploaded=100, secondary_bytes=30)),
    ]

    tm.record_fit_round(strategy, server_round=1, results=results, aggregated_parameters=None)

    snapshot = tm.snapshot_for_round(1)
    assert "client/avg_secondary_bytes_uploaded" not in snapshot.round_client_metrics
