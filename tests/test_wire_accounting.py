"""Integration tests for wire protocol accounting and cross-q lifecycle (#88).

Verifies:
1. FedPAQ and FedMAQ transmission through packed wire payloads.
2. Cross-q transition handling: differential state cold-start on q change, diff-coding on same-q,
   and continuous float32 residual error-feedback preservation.
3. Independent declared budget parity test between DAdaQuant and FedPAQ at nominal equality.
4. Bidirectional headline and upload-only parallel telemetry conservation.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import RecordDict

from fedmaq.baselines.postprocess import FedMAQPostProcessCompressionHook
from fedmaq.baselines.quantization import DAdaQuantCompressionHook, FedPAQCompressionHook
from fedmaq.core.wire_codec import HEADER_SIZE, unpack_quantized_tensor


def test_fedpaq_transmits_through_packed_wire_protocol() -> None:
    hook = FedPAQCompressionHook(q=4, rng=np.random.default_rng(0))
    deltas = [
        np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32),
        np.array([2.0, -2.0], dtype=np.float32),
    ]

    quantized, report = hook.compress(deltas)

    assert len(report.payloads) == 2
    # Tensor 0: 4 elements * 4 bits = 16 bits (2 bytes) + 16B header = 18B
    assert len(report.payloads[0]) == HEADER_SIZE + 2
    # Tensor 1: 2 elements * 4 bits = 8 bits (1 byte) + 16B header = 17B
    assert len(report.payloads[1]) == HEADER_SIZE + 1
    assert report.payload_bytes == 18 + 17

    # Verify each payload unpacks correctly
    unpacked0 = unpack_quantized_tensor(report.payloads[0])
    assert unpacked0.q == 4
    assert unpacked0.bit_width == 4
    assert unpacked0.is_diff is False
    assert len(unpacked0.codes) == 4


def test_cross_q_transition_differential_and_residual_lifecycle() -> None:
    """Test differential cold-start on q change and continuous residual preservation."""
    state = RecordDict()
    delta = np.array([1.0, 0.4], dtype=np.float32)

    # Round 1: q = 8 (fresh state -> cold start, is_diff=False)
    hook_r1 = FedMAQPostProcessCompressionHook(q=8, state=state, rng=np.random.default_rng(10))
    out_r1, report_r1 = hook_r1.compress([delta.copy()])
    unpacked_r1 = unpack_quantized_tensor(report_r1.payloads[0])
    assert unpacked_r1.q == 8
    assert unpacked_r1.bit_width == 8
    assert unpacked_r1.is_diff is False

    # Round 2: same q = 8 (diff coding engages -> is_diff=True, bit_width=9)
    hook_r2 = FedMAQPostProcessCompressionHook(q=8, state=state, rng=np.random.default_rng(10))
    out_r2, report_r2 = hook_r2.compress([delta.copy()])
    unpacked_r2 = unpack_quantized_tensor(report_r2.payloads[0])
    assert unpacked_r2.q == 8
    assert unpacked_r2.bit_width == 9  # q + 1 bits for diffs
    assert unpacked_r2.is_diff is True

    # Round 3: q changes to 4 (q transition -> cold-start diff coding, is_diff=False, bit_width=4)
    hook_r3 = FedMAQPostProcessCompressionHook(q=4, state=state, rng=np.random.default_rng(10))
    out_r3, report_r3 = hook_r3.compress([delta.copy()])
    unpacked_r3 = unpack_quantized_tensor(report_r3.payloads[0])
    assert unpacked_r3.q == 4
    assert unpacked_r3.bit_width == 4
    assert unpacked_r3.is_diff is False

    # Residual error feedback was preserved into round 3
    residual_r2 = state.get("fedmaq_postprocess_residual").to_numpy_ndarrays()[0]
    assert np.any(residual_r2 != 0)

    # Round 4: same q = 4 (diff coding engages again for q=4 -> is_diff=True, bit_width=5)
    hook_r4 = FedMAQPostProcessCompressionHook(q=4, state=state, rng=np.random.default_rng(10))
    out_r4, report_r4 = hook_r4.compress([delta.copy()])
    unpacked_r4 = unpack_quantized_tensor(report_r4.payloads[0])
    assert unpacked_r4.q == 4
    assert unpacked_r4.bit_width == 5  # q + 1 bits
    assert unpacked_r4.is_diff is True


def test_dadaquant_and_fedpaq_declared_budget_parity() -> None:
    """DAdaQuant and FedPAQ declared budget parity where nominal precision is equal.

    FedPAQ with q=8 has 2^(8-1) - 1 = 127 positive levels (255 total levels).
    DAdaQuant with q=127 levels has 127 positive levels (255 total levels, ceil(log2(255)) = 8b).
    Where their nominal precision is equal, pre-transport wire payload_bytes match exactly.
    """
    size = 1024
    rng = np.random.default_rng(42)
    deltas = [rng.normal(size=size).astype(np.float32)]

    fedpaq_hook = FedPAQCompressionHook(q=8, rng=np.random.default_rng(1))
    _, fedpaq_report = fedpaq_hook.compress(deltas)

    dada_hook = DAdaQuantCompressionHook(q=127, rng=np.random.default_rng(1))
    _, dada_report = dada_hook.compress(deltas)

    # Both transmit 1024 elements * 8 bits = 1024 bytes data + 16 bytes header = 1040 bytes
    expected_wire_bytes = HEADER_SIZE + 1024
    assert fedpaq_report.payload_bytes == expected_wire_bytes
    assert dada_report.payload_bytes == expected_wire_bytes
    assert fedpaq_report.payload_bytes == dada_report.payload_bytes


def test_telemetry_upload_and_bidirectional_conservation(tmp_path: pytest.TempPathFactory) -> None:
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

    from fedmaq.core.payload_archive import PayloadArchive
    from fedmaq.core.strategy import TelemetryFedAvg
    from fedmaq.core.telemetry import TelemetryManager

    class _FakeProxy:
        def __init__(self, cid: str) -> None:
            self.cid = cid

    cfg = {
        "experiment": {
            "num_clients": 2,
            "client_fraction": 1.0,
            "total_rounds": 1,
            "num_public_samples": 100,
            "telemetry": {"wandb_enabled": False},
        },
        "algorithm": {"name": "fedpaq", "q": 8},
    }

    tm = TelemetryManager(cfg)
    tm.log_dir = tmp_path
    tm.jsonl_path = tmp_path / "log.jsonl"
    tm.csv_path = tmp_path / "log.csv"
    tm.payload_archive = PayloadArchive(tmp_path)

    strategy = TelemetryFedAvg(
        telemetry_manager=tm,
        config=cfg,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    fit_res_0 = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=50,
        metrics={"partition_id": 0, "bytes_uploaded": 500, "payload_bytes": 1000},
    )
    fit_res_1 = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=50,
        metrics={"partition_id": 1, "bytes_uploaded": 700, "payload_bytes": 1400},
    )

    tm.record_fit_round(
        strategy,
        server_round=1,
        results=[(_FakeProxy("0"), fit_res_0), (_FakeProxy("1"), fit_res_1)],
        aggregated_parameters=None,
    )

    snapshot = tm.snapshot_for_round(1)
    assert snapshot.round_upload_bytes == 1200
    assert snapshot.round_bytes == 1200  # download is 0 when aggregated_parameters is None

    tm.log(
        1,
        {
            "round": 1,
            "communication/round_bytes": snapshot.round_bytes,
            "communication/round_upload_bytes": snapshot.round_upload_bytes,
        },
    )

    assert tm.cumulative_bytes == 1200
    assert tm.cumulative_upload_bytes == 1200
    assert tm.cumulative_upload_bytes <= tm.cumulative_bytes
