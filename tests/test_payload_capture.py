"""Tests for the payload-capture side of #25 AC 2 ("reproducible offline").

A summed ``payload_bytes`` scalar (the pre-existing ``round_payload_bytes`` CSV
column) cannot be re-scored against a different encoder, because encoders like
zlib are content-sensitive, not just size-sensitive. These tests cover the
opt-in capability that closes that gap: every ``compress()`` return carries
its per-call payloads in ``UploadReport.payloads``, client fit strategies attach
them to ``fit_metrics`` when ``experiment.telemetry.log_payloads`` is set, and
``TelemetryManager`` persists them to a side file that can be replayed offline.
"""

import bz2
import pickle

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from torch.utils.data import TensorDataset

from fedmaq.baselines.compression import FedKDCompressionHook
from fedmaq.baselines.postprocess import FedMAQPostProcessCompressionHook
from fedmaq.baselines.quantization import DAdaQuantCompressionHook, FedPAQCompressionHook
from fedmaq.baselines.transport import measure_bytes, pack_payloads, unpack_payloads
from fedmaq.core.client import CompressionHook, GenericClient, LossHook
from fedmaq.core.client_hooks.base import attach_payloads_if_enabled
from fedmaq.core.models import SimpleCNN, get_model_parameters
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.strategy_hooks.passthrough import PassthroughHook
from fedmaq.core.telemetry import TelemetryManager

# --- UploadReport payloads reproduce each hook's returned measured size ---


def _assert_payloads_reproduce_byte_size(hook, deltas):
    _, report = hook.compress(deltas)
    assert sum(len(p) for p in report.payloads) == report.payload_bytes
    assert sum(measure_bytes(p) for p in report.payloads) == report.measured_bytes
    return report.payloads


def test_identity_hook_report_payloads_reproduce_byte_size():
    deltas = [np.array([-2.0, 0.0, 2.0], dtype=np.float32), np.zeros((0,), dtype=np.float32)]
    payloads = _assert_payloads_reproduce_byte_size(CompressionHook(), deltas)
    # The empty tensor contributes no measure_bytes call and no payload.
    assert len(payloads) == 1


def test_fedpaq_report_payloads_reproduce_byte_size():
    deltas = [np.array([-2.0, 0.0, 2.0], dtype=np.float32)]
    _assert_payloads_reproduce_byte_size(
        FedPAQCompressionHook(q=8, rng=np.random.default_rng(0)), deltas
    )


def test_dadaquant_report_payloads_reproduce_byte_size():
    deltas = [np.ones((100,), dtype=np.float32)]
    _assert_payloads_reproduce_byte_size(
        DAdaQuantCompressionHook(q=4, rng=np.random.default_rng(0)), deltas
    )


def test_default_download_payloads_use_float32_and_shared_transport():
    hook = PassthroughHook()
    parameters = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.zeros((0,), dtype=np.float32),
    ]

    report = hook.download_size_bytes(None, parameters)
    expected_payload = parameters[0].astype(np.float32).tobytes()

    assert report.payloads == (expected_payload,)
    assert report.measured_bytes == measure_bytes(expected_payload)


def test_feddistill_download_payloads_reproduce_byte_size_in_both_rounds():
    """FedDistill charges both download legs through the shared seam (ADR-0018).

    Round 1 has no consensus matrix, so only the weight leg exists; the logit
    leg appears from round 2 and is measured as its own payload rather than
    concatenated, because ``measure_bytes`` is not additive over concatenation.
    """
    from fedmaq.core.client_hooks.feddistill import logits_to_bytes
    from fedmaq.core.strategy_hooks.feddistill import FedDistillHook

    hook = FedDistillHook({"dataset": {"num_classes": 3}})
    parameters = [np.arange(6, dtype=np.float32).reshape(2, 3)]

    first = hook.download_size_bytes(None, parameters)
    assert sum(measure_bytes(p) for p in first.payloads) == first.measured_bytes

    hook.global_logits = np.full((3, 3), 0.25, dtype=np.float32)
    second = hook.download_size_bytes(None, parameters)
    logit_payload = logits_to_bytes(hook.global_logits)

    assert second.payloads[-1] == logit_payload
    assert sum(measure_bytes(p) for p in second.payloads) == second.measured_bytes
    assert second.measured_bytes == first.measured_bytes + measure_bytes(logit_payload)


def test_fedmaq_postprocess_report_payloads_reproduce_byte_size():
    deltas = [np.array([-2.0, 0.0, 2.0], dtype=np.float32), np.zeros((3,), dtype=np.float32)]
    _assert_payloads_reproduce_byte_size(
        FedMAQPostProcessCompressionHook(q=8, rng=np.random.default_rng(0)), deltas
    )


def test_fedkd_upload_report_payloads_reproduce_byte_size():
    rng = np.random.default_rng(0)
    deltas = [rng.standard_normal((10, 5)).astype(np.float32)]
    _assert_payloads_reproduce_byte_size(FedKDCompressionHook(energy=0.9), deltas)


def test_all_zero_tensor_still_contributes_one_payload():
    """All-zero tensors skip quantization but still route through measure_bytes
    (#25's own design) -- the payload capture must not silently drop that call."""
    deltas = [np.zeros((5,), dtype=np.float32)]
    payloads = _assert_payloads_reproduce_byte_size(FedPAQCompressionHook(q=8), deltas)
    assert len(payloads) == 1


# --- attach_payloads_if_enabled: default-off, opt-in roundtrip ---


class _FakeClient:
    def __init__(self, log_payloads: bool | None):
        config: dict = {"experiment": {}}
        if log_payloads is not None:
            config["experiment"]["telemetry"] = {"log_payloads": log_payloads}
        self.config = config


def test_attach_payloads_if_enabled_default_off_leaves_metrics_untouched():
    fit_metrics: dict = {"bytes_uploaded": 42}
    attach_payloads_if_enabled(_FakeClient(None), fit_metrics, [b"abc"])
    assert fit_metrics == {"bytes_uploaded": 42}


def test_attach_payloads_if_enabled_explicit_false_leaves_metrics_untouched():
    fit_metrics: dict = {}
    attach_payloads_if_enabled(_FakeClient(False), fit_metrics, [b"abc"])
    assert "payloads_framed" not in fit_metrics


def test_attach_payloads_if_enabled_true_roundtrips_the_payloads():
    payloads = [b"AAAA", b"", b"BBBBBBBBBB"]
    fit_metrics: dict = {}
    attach_payloads_if_enabled(_FakeClient(True), fit_metrics, payloads)
    assert unpack_payloads(fit_metrics["payloads_framed"]) == payloads


# --- Client-hook integration: standard/FedKD/FedDistill fit() attach the gate ---


def _mnist_loader():
    data = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    return torch.utils.data.DataLoader(TensorDataset(data, labels), batch_size=4)


def _make_client(alg_cfg: dict, compressor_hook, log_payloads: bool) -> GenericClient:
    model = SimpleCNN(in_channels=1, num_classes=10)
    loader = _mnist_loader()
    cfg = {
        "seed": 42,
        "experiment": {
            "local_epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "telemetry": {"log_payloads": log_payloads},
        },
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


def test_standard_fit_attaches_payloads_matching_compressor_hook_when_enabled():
    client = _make_client({"name": "fedpaq", "q": 8}, FedPAQCompressionHook(q=8), True)
    params = get_model_parameters(client.model)

    _, _, fit_metrics = client.fit(params, {"server_round": 1})

    assert (
        sum(len(p) for p in unpack_payloads(fit_metrics["payloads_framed"]))
        == fit_metrics["payload_bytes"]
    )


def test_standard_fit_omits_payloads_when_disabled():
    client = _make_client({"name": "fedpaq", "q": 8}, FedPAQCompressionHook(q=8), False)
    params = get_model_parameters(client.model)

    _, _, fit_metrics = client.fit(params, {"server_round": 1})

    assert "payloads_framed" not in fit_metrics


def test_fedkd_fit_attaches_payloads_matching_compressor_hook_when_enabled():
    client = _make_client(
        {"name": "fedkd", "temperature": 2.0}, FedKDCompressionHook(energy=0.9), True
    )
    params = get_model_parameters(client.model)

    _, _, fit_metrics = client.fit(params, {"server_round": 1})

    assert (
        sum(len(p) for p in unpack_payloads(fit_metrics["payloads_framed"]))
        == fit_metrics["payload_bytes"]
    )


def test_feddistill_fit_attaches_combined_weight_and_logit_payloads_when_enabled():
    client = _make_client({"name": "feddistill", "reg_alpha": 1.0}, CompressionHook(), True)
    params = get_model_parameters(client.model)

    _, _, fit_metrics = client.fit(params, {"server_round": 1})

    payloads = unpack_payloads(fit_metrics["payloads_framed"])
    # Weight-leg payloads, then the logit-leg payload last, matching the order
    # of the measure_bytes calls in FedDistillFit.fit.
    assert payloads[-1] == fit_metrics["client_logits"]
    assert sum(measure_bytes(p) for p in payloads) == fit_metrics["bytes_uploaded"]


# --- TelemetryManager.record_fit_round: persistence + offline replay ---


class _FakeProxy:
    """Minimal ClientProxy stand-in -- record_fit_round never calls into it."""

    def __init__(self, cid: str) -> None:
        self.cid = cid


def _make_strategy(
    tmp_path, monkeypatch, log_payloads: bool
) -> tuple[TelemetryFedAvg, TelemetryManager]:
    monkeypatch.setattr("fedmaq.core.telemetry._HYDRA_AVAILABLE", False)
    cfg_dict = {
        "experiment": {
            "num_clients": 1,
            "client_fraction": 1.0,
            "total_rounds": 1,
            "num_public_samples": 3000,
            "telemetry": {"wandb_enabled": False, "log_payloads": log_payloads},
        },
        "algorithm": {"name": "fedavg"},
    }
    tm = TelemetryManager(cfg_dict)
    tm.log_dir = tmp_path
    tm.jsonl_path = tmp_path / "experiment_log.jsonl"
    tm.csv_path = tmp_path / "experiment_log.csv"
    tm.payloads_dir = tmp_path / "payloads"
    strategy = TelemetryFedAvg(
        telemetry_manager=tm,
        config=cfg_dict,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
    )
    return strategy, tm


def _fit_res_with_payloads(payloads: list[bytes]) -> FitRes:
    bytes_uploaded = sum(measure_bytes(p) for p in payloads)
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=10,
        metrics={
            "partition_id": 0,
            "bytes_uploaded": bytes_uploaded,
            "payload_bytes": sum(len(p) for p in payloads),
            "payloads_framed": pack_payloads(payloads),
        },
    )


@pytest.mark.parametrize(
    "partition_metrics",
    [{}, {"partition_id": -1}, {"partition_id": 1}],
)
def test_record_fit_round_matches_sort_failure_for_unresolved_partition(
    tmp_path, monkeypatch, partition_metrics
):
    strategy, tm = _make_strategy(tmp_path, monkeypatch, log_payloads=False)
    proxy = _FakeProxy("node-without-partition")
    metrics = {"bytes_uploaded": 100, "payload_bytes": 100, **partition_metrics}
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=10,
        metrics=metrics,
    )

    with pytest.raises(ValueError) as recorder_error:
        tm.record_fit_round(
            strategy, server_round=1, results=[(proxy, fit_res)], aggregated_parameters=None
        )
    with pytest.raises(ValueError) as sort_error:
        strategy._partition_sort_key(proxy, fit_res)

    assert str(recorder_error.value) == str(sort_error.value)


def test_record_fit_round_persists_payloads_and_replays_the_original_total(tmp_path, monkeypatch):
    strategy, tm = _make_strategy(tmp_path, monkeypatch, log_payloads=True)
    payloads = [b"A" * 40, b"B" * 60]
    fit_res = _fit_res_with_payloads(payloads)

    tm.record_fit_round(
        strategy, server_round=1, results=[(_FakeProxy("0"), fit_res)], aggregated_parameters=None
    )

    payloads_path = tm.payloads_dir / "round_0001.pkl"
    assert payloads_path.exists()
    with open(payloads_path, "rb") as f:
        persisted = pickle.load(f)

    assert persisted == {0: payloads}
    # Round-trip identity: replaying the persisted payloads through the same
    # measure_bytes reproduces the round's logged bytes_uploaded exactly.
    replayed_total = sum(measure_bytes(p) for p in persisted[0])
    assert replayed_total == fit_res.metrics["bytes_uploaded"]


def test_record_fit_round_persists_download_payloads_and_replays_logged_total(
    tmp_path, monkeypatch
):
    strategy, tm = _make_strategy(tmp_path, monkeypatch, log_payloads=True)
    upload_payloads = [b"client update"]
    fit_res = _fit_res_with_payloads(upload_payloads)
    aggregated = ndarrays_to_parameters([np.array([1.0, -2.0, 3.0], dtype=np.float32)])

    _, round_total = tm.record_fit_round(
        strategy,
        server_round=1,
        results=[(_FakeProxy("0"), fit_res)],
        aggregated_parameters=aggregated,
    )

    path = tm.payloads_dir / "download_round_0001.pkl"
    assert path.exists()
    with open(path, "rb") as f:
        persisted = pickle.load(f)

    replayed_download_total = sum(
        measure_bytes(payload)
        for client_payloads in persisted.values()
        for payload in client_payloads
    )
    assert replayed_download_total == round_total - fit_res.metrics["bytes_uploaded"]


def test_record_fit_round_persisted_payloads_rescore_under_an_alternate_encoder(
    tmp_path, monkeypatch
):
    """The actual AC 2 demonstration: a hypothetical alternate encoder can be
    scored against an already-completed round with no training re-run."""
    strategy, tm = _make_strategy(tmp_path, monkeypatch, log_payloads=True)
    rng = np.random.default_rng(1)
    payloads = [rng.integers(0, 256, size=500, dtype=np.uint8).tobytes()]
    fit_res = _fit_res_with_payloads(payloads)

    tm.record_fit_round(
        strategy, server_round=1, results=[(_FakeProxy("0"), fit_res)], aggregated_parameters=None
    )

    with open(tm.payloads_dir / "round_0001.pkl", "rb") as f:
        persisted = pickle.load(f)

    original_total = fit_res.metrics["bytes_uploaded"]
    alternate_total = sum(len(bz2.compress(p)) for p in persisted[0])
    assert alternate_total != original_total


def test_record_fit_round_writes_no_payloads_file_when_flag_disabled(tmp_path, monkeypatch):
    strategy, tm = _make_strategy(tmp_path, monkeypatch, log_payloads=False)
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([]),
        num_examples=10,
        metrics={"partition_id": 0, "bytes_uploaded": 100, "payload_bytes": 100},
    )

    tm.record_fit_round(
        strategy,
        server_round=1,
        results=[(_FakeProxy("0"), fit_res)],
        aggregated_parameters=ndarrays_to_parameters(
            [np.array([1.0, -2.0, 3.0], dtype=np.float32)]
        ),
    )

    assert not tm.payloads_dir.exists()
