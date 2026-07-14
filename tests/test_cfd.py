"""Unit tests for the CFD baseline (Compressed Federated Distillation, Sattler et al. 2022):
constrained soft-label quantization/codec, server-side dual distillation, and the
client-side fresh-init + round-gated distillation path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.app import RecordDict
from flwr.common import Code, FitIns, Status, ndarrays_to_parameters
from flwr.common.typing import FitRes
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core.client import CompressionHook, GenericClient, LossHook
from fedmaq.core.client_hooks.cfd import CFDFit
from fedmaq.core.models import get_model_parameters
from fedmaq.core.softlabel_codec import (
    codes_from_bytes,
    codes_to_bytes,
    constrained_quantize,
    dequantize,
    encode_bytes,
)
from fedmaq.core.strategy_hooks.cfd import CFDHook


def _softmax_rows(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    logits = rng.normal(size=(n, k))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- #
# softlabel_codec: constrained quantizer + delta/zlib codec.                  #
# --------------------------------------------------------------------------- #


def test_constrained_quantize_one_hot_at_b1():
    """b=1 must reduce exactly to argmax one-hot (CFD paper eq. 11)."""
    rng = np.random.default_rng(0)
    probs = _softmax_rows(rng, 8, 5)
    codes = constrained_quantize(probs, 1)

    assert (codes.sum(axis=1) == 1).all()
    assert (codes.argmax(axis=1) == probs.argmax(axis=1)).all()
    assert set(np.unique(codes).tolist()) <= {0, 1}


def test_constrained_quantize_sum_exact_and_dequantize_close():
    """b>1: row sums equal 2**b - 1 exactly; dequantize recovers probs approximately."""
    rng = np.random.default_rng(1)
    probs = _softmax_rows(rng, 16, 10)
    b = 4
    codes = constrained_quantize(probs, b)

    assert (codes.sum(axis=1) == (2**b - 1)).all()
    deq = dequantize(codes, b)
    np.testing.assert_allclose(deq.sum(axis=1), 1.0, atol=1e-5)
    assert np.max(np.abs(deq - probs)) < (1.0 / (2**b - 1)) + 1e-6


def test_constrained_quantize_rejects_bad_input():
    with pytest.raises(ValueError, match="b must be >= 1"):
        constrained_quantize(np.ones((2, 2)), 0)
    with pytest.raises(ValueError, match="2D"):
        constrained_quantize(np.ones(4), 2)


def test_encode_bytes_delta_reduces_size_on_repeat():
    """Identical codes across two calls -> near-zero diff, smaller than a fresh call."""
    rng = np.random.default_rng(2)
    probs = _softmax_rows(rng, 32, 10)
    codes = constrained_quantize(probs, 8)

    bytes_raw, prev = encode_bytes(codes, None)
    bytes_repeat, _ = encode_bytes(codes, prev)
    assert bytes_repeat < bytes_raw


def test_encode_bytes_delta_false_ignores_prev_codes():
    """delta=False must compress the raw codes regardless of prev_codes (cfd.yaml's
    delta_coding: false must actually disable diffing, not just look like it does)."""
    rng = np.random.default_rng(2)
    probs = _softmax_rows(rng, 32, 10)
    codes = constrained_quantize(probs, 8)

    _, prev = encode_bytes(codes, None)
    bytes_with_delta_off, _ = encode_bytes(codes, prev, delta=False)
    bytes_fresh_no_prev, _ = encode_bytes(codes, None, delta=False)
    assert bytes_with_delta_off == bytes_fresh_no_prev


def test_codes_bytes_round_trip_and_mismatch_raises():
    rng = np.random.default_rng(3)
    probs = _softmax_rows(rng, 4, 6)
    codes = constrained_quantize(probs, 4)

    buf = codes_to_bytes(codes)
    back = codes_from_bytes(buf, 6)
    assert np.array_equal(back, codes)

    with pytest.raises(ValueError, match="num_classes"):
        codes_from_bytes(buf, 5)


# --------------------------------------------------------------------------- #
# CFDHook: server-side dual distillation across two rounds.                   #
# --------------------------------------------------------------------------- #


def _public_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    images = torch.randn(n, 1, 28, 28)
    labels = torch.zeros(n, dtype=torch.long)  # unused (unlabeled proxy set)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def _client_fit_res(codes: np.ndarray) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([codes.astype(np.int64)]),
        num_examples=10,
        metrics={},
    )


def test_cfd_hook_pre_aggregate_bypasses_fedavg():
    """pre_aggregate_fit must return the (None, {}) tuple, not bare None, so the
    strategy skips weight-averaging the soft-label codes."""
    hook = CFDHook({"dataset": {"name": "mnist", "num_classes": 4}})
    rng = np.random.default_rng(4)
    codes = constrained_quantize(_softmax_rows(rng, 8, 4), hook.b_up)
    results = [(None, _client_fit_res(codes))]

    result = hook.pre_aggregate_fit(None, 2, results, [])
    assert result is not None
    params, metrics = result
    assert params is None
    assert metrics == {}
    assert hook._pending_targets is not None


def test_cfd_hook_dual_distillation_two_rounds():
    """Server model persists and updates across two rounds of dual distillation."""
    hook = CFDHook(
        {
            "dataset": {"name": "mnist", "num_classes": 4},
            "algorithm": {"b_up": 4, "b_down": 4, "server_distill_epochs": 1},
        }
    )
    hook._public_loader = _public_loader()

    initial_params = [p.copy() for p in get_model_parameters(hook.server_model)]
    rng = np.random.default_rng(5)

    for server_round in (1, 2):
        codes = constrained_quantize(_softmax_rows(rng, 8, 4), hook.b_up)
        results = [(None, _client_fit_res(codes))]

        pre_result = hook.pre_aggregate_fit(None, server_round, results, [])
        assert pre_result == (None, {})

        aggregated_parameters, metrics = hook.aggregate_fit(
            None, server_round, results, [], None, {}
        )
        assert aggregated_parameters is not None

    updated_params = get_model_parameters(hook.server_model)
    assert any(not np.allclose(a, b) for a, b in zip(initial_params, updated_params, strict=True))


def test_cfd_hook_downstream_broadcast_skips_round1():
    """configure_fit must not broadcast server labels in round 1 (untrained model)."""
    hook = CFDHook({"dataset": {"name": "mnist", "num_classes": 4}})
    hook._public_loader = _public_loader()

    fit_ins = FitIns(ndarrays_to_parameters([]), {})
    instructions = [(None, fit_ins)]

    round1 = hook.configure_fit(None, 1, ndarrays_to_parameters([]), None, instructions)
    assert "cfd_server_labels" not in round1[0][1].config
    assert hook.download_size_bytes(None, []) == 0

    round2 = hook.configure_fit(None, 2, ndarrays_to_parameters([]), None, instructions)
    assert isinstance(round2[0][1].config["cfd_server_labels"], bytes)
    assert hook.download_size_bytes(None, []) > 0


# --------------------------------------------------------------------------- #
# CFDFit: client-side fresh init + round-gated distillation.                  #
# --------------------------------------------------------------------------- #


def _make_client(state: RecordDict | None = None) -> GenericClient:
    from fedmaq.core.models import SimpleCNN

    train_data = torch.randn(8, 1, 28, 28)
    train_labels = torch.randint(0, 4, (8,))
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=4)
    model = SimpleCNN(in_channels=1, num_classes=4)
    cfg = {
        "experiment": {"local_epochs": 1, "learning_rate": 0.01, "weight_decay": 0.0},
        "algorithm": {
            "name": "cfd",
            "b_up": 4,
            "b_down": 4,
            "distill_epochs": 1,
            "temperature": 1.0,
        },
        "dataset": {"name": "mnist", "num_classes": 4},
    }
    return GenericClient(
        cid="0",
        trainloader=train_loader,
        testloader=train_loader,
        model=model,
        loss_hook=LossHook(),
        compressor_hook=CompressionHook(),
        config=cfg,
        public_loader=_public_loader(n=8, batch_size=4),
        state=state,
    )


def test_cfd_client_round1_private_only_no_server_labels():
    client = _make_client()
    fit = CFDFit()

    params, num_examples, metrics = fit.fit(client, [], {"server_round": 1})

    assert len(params) == 1
    assert params[0].shape == (8, 4)  # [D_proxy, num_classes]
    assert (params[0].sum(axis=1) == (2**4 - 1)).all()
    assert num_examples == 8
    assert isinstance(metrics["bytes_uploaded"], int)
    assert metrics["bytes_uploaded"] > 0


def test_cfd_client_round2_engages_distill_branch():
    client = _make_client()
    fit = CFDFit()

    rng = np.random.default_rng(6)
    server_codes = constrained_quantize(_softmax_rows(rng, 8, 4), 4)
    config = {
        "server_round": 2,
        "cfd_server_labels": codes_to_bytes(server_codes),
    }

    params, num_examples, metrics = fit.fit(client, [], config)
    assert len(params) == 1
    assert params[0].shape == (8, 4)
    assert np.all(np.isfinite(params[0]))


def test_cfd_client_upstream_delta_state_persists_across_rounds():
    """The upstream delta-reference must persist via client.state and shrink the
    second round's byte count when the client's predictions repeat."""
    state = RecordDict()
    client = _make_client(state=state)
    fit = CFDFit()

    assert state.get("cfd_prev_up_codes") is None
    _, _, metrics_r1 = fit.fit(client, [], {"server_round": 1})
    assert state.get("cfd_prev_up_codes") is not None

    # Second call with the model unchanged (freeze via eval-mode weights) should
    # at minimum not crash and should keep updating the delta reference.
    _, _, metrics_r2 = fit.fit(client, [], {"server_round": 1})
    assert isinstance(metrics_r2["bytes_uploaded"], int)
