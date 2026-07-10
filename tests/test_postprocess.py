"""Unit tests for FedMAQ's post-processing pipeline (error-feedback + diff-coding + zlib)."""

import math
import zlib

import numpy as np
from flwr.app import RecordDict

from fedmaq.baselines import FedMAQPostProcessCompressionHook, get_compressor_hook
from fedmaq.baselines.quantization import FedPAQCompressionHook


def test_round1_matches_plain_fedpaq_output():
    """Fresh state (round 1): error-feedback residual and diff-coding are both
    no-ops, so the returned float arrays must match plain FedPAQ bit-for-bit.

    (Byte counts differ by design — the new hook reports real zlib bytes, plain
    FedPAQ reports the synthetic ceil(bits*size/8)+4 formula — so only the
    dequantized array list is compared, not the tuple's int.)
    """
    rng = np.random.default_rng(0)
    deltas = [rng.normal(size=(4, 4)).astype(np.float32), rng.normal(size=(8,)).astype(np.float32)]

    plain_out, _ = FedPAQCompressionHook(q=8).compress([d.copy() for d in deltas])
    post_out, _ = FedMAQPostProcessCompressionHook(q=8).compress([d.copy() for d in deltas])

    for p, o in zip(plain_out, post_out, strict=True):
        np.testing.assert_allclose(p, o)


def test_state_persists_and_diffing_engages_on_second_call():
    """State is populated after one compress() call; a second call reads it back
    and produces a different (diffed) byte count than an isolated fresh-state call."""
    state = RecordDict()
    hook = FedMAQPostProcessCompressionHook(q=8, state=state)
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(16,)).astype(np.float32)

    assert state.get("fedmaq_postprocess_residual") is None
    hook.compress([delta.copy()])
    assert state.get("fedmaq_postprocess_residual") is not None
    assert state.get("fedmaq_postprocess_prev_codes") is not None

    # Second call with an identical delta: codes should match round 1's codes,
    # so the diff against prev_codes collapses to all-zero -> smaller byte count
    # than a fresh-state hook compressing the same delta from scratch.
    _, bytes_round2 = hook.compress([delta.copy()])
    fresh_hook = FedMAQPostProcessCompressionHook(q=8)
    _, bytes_fresh = fresh_hook.compress([delta.copy()])
    assert bytes_round2 <= bytes_fresh


def test_error_feedback_carries_quantization_error_into_next_round():
    """Hand-computed: with q=2 (levels=1), quantization error on a non-extremal
    element must be folded into the next round's input via the residual."""
    state2 = RecordDict()
    hook2 = FedMAQPostProcessCompressionHook(q=2, state=state2)  # levels = 1
    d = np.array([1.0, 0.4], dtype=np.float32)
    out_r1, _ = hook2.compress([d.copy()])
    # scale = 1.0, levels = 1: codes = round((d/1.0)*1) = [1, 0], dequant = [1.0, 0.0]
    np.testing.assert_allclose(out_r1[0], [1.0, 0.0])
    residual = state2.get("fedmaq_postprocess_residual").to_numpy_ndarrays()[0]
    np.testing.assert_allclose(residual, [0.0, 0.4], atol=1e-6)

    # Round 2: same delta fed again. d_fb = d + residual = [1.0, 0.8].
    out_r2, _ = hook2.compress([d.copy()])
    d_fb2 = np.array([1.0, 0.8])
    scale2 = float(np.max(np.abs(d_fb2)))
    expected_codes = np.round(d_fb2 / scale2 * 1)
    expected = expected_codes / 1 * scale2
    np.testing.assert_allclose(out_r2[0], expected, atol=1e-6)


def test_diff_coding_reflects_codes_minus_prev_codes():
    """Seeding state with known prev_codes must change the zlib-measured byte
    count relative to a fresh-state call on the same input (raw codes vs diffed)."""
    delta = np.linspace(-1, 1, 64).astype(np.float32)

    fresh_state = RecordDict()
    fresh_hook = FedMAQPostProcessCompressionHook(q=8, state=fresh_state)
    _, bytes_raw = fresh_hook.compress([delta.copy()])
    raw_codes = fresh_state.get("fedmaq_postprocess_prev_codes").to_numpy_ndarrays()[0]

    # Seed a second hook's state with those exact codes as "previous round" and
    # residual zero, then compress the identical delta again -> codes - prev_codes
    # must be all zero, which zlib compresses far smaller than the raw payload.
    from flwr.app import ArrayRecord

    seeded_state = RecordDict()
    seeded_state["fedmaq_postprocess_prev_codes"] = ArrayRecord(
        numpy_ndarrays=[raw_codes.copy()]
    )
    seeded_hook = FedMAQPostProcessCompressionHook(q=8, state=seeded_state)
    _, bytes_diffed = seeded_hook.compress([delta.copy()])

    all_zero_payload = np.zeros_like(raw_codes).tobytes()
    assert bytes_diffed == len(zlib.compress(all_zero_payload)) + 4
    assert bytes_diffed < bytes_raw


def test_byte_count_realism():
    """Compressible input -> byte count strictly below the naive ceil(bits*size/8)+4
    formula. Incompressible (random) input -> byte count still finite and bounded."""
    size = 256
    bits = 8

    compressible = np.zeros(size, dtype=np.float32)
    compressible[0] = 1.0  # mostly-zero -> highly compressible codes
    _, bytes_compressible = FedMAQPostProcessCompressionHook(q=bits).compress(
        [compressible]
    )
    naive_formula = int(math.ceil(bits * size / 8.0)) + 4
    assert bytes_compressible < naive_formula

    rng = np.random.default_rng(2)
    incompressible = rng.normal(size=size).astype(np.float32)
    _, bytes_incompressible = FedMAQPostProcessCompressionHook(q=bits).compress(
        [incompressible]
    )
    assert bytes_incompressible > 0
    # zlib overhead on an incompressible int64-code payload shouldn't wildly
    # inflate beyond the raw payload size.
    assert bytes_incompressible < size * 8 * 2 + 64


def test_shape_mismatch_cold_state_fallback_does_not_raise():
    """A stale prev_codes/residual entry with a mismatched shape must not raise;
    the hook falls back to cold-start behavior for that tensor."""
    from flwr.app import ArrayRecord

    state = RecordDict()
    state["fedmaq_postprocess_residual"] = ArrayRecord(
        numpy_ndarrays=[np.zeros(4, dtype=np.float32)]
    )
    state["fedmaq_postprocess_prev_codes"] = ArrayRecord(
        numpy_ndarrays=[np.zeros(4, dtype=np.int64)]
    )
    hook = FedMAQPostProcessCompressionHook(q=8, state=state)

    mismatched_delta = np.ones(10, dtype=np.float32)
    out, nbytes = hook.compress([mismatched_delta])
    assert out[0].shape == (10,)
    assert nbytes > 0


def test_empty_and_all_zero_tensor_pass_through():
    """Mirrors quantization.py's empty-tensor and all-zero-tensor branches."""
    hook = FedMAQPostProcessCompressionHook(q=8)

    empty = np.zeros((0,), dtype=np.float32)
    zero = np.zeros((5,), dtype=np.float32)
    out, nbytes = hook.compress([empty, zero])

    assert out[0].shape == (0,)
    np.testing.assert_allclose(out[1], zero)
    assert nbytes == 4  # only the all-zero tensor contributes (empty is free)


def test_output_contract_matches_input_shape_dtype():
    """Output list length/shape/dtype must satisfy standard.py's `o + cd` reconstruction."""
    rng = np.random.default_rng(3)
    deltas = [
        rng.normal(size=(3, 5)).astype(np.float32),
        rng.normal(size=(7,)).astype(np.float32),
        np.zeros((0,), dtype=np.float32),
    ]
    out, _ = FedMAQPostProcessCompressionHook(q=8).compress(deltas)

    assert len(out) == len(deltas)
    for o, d in zip(out, deltas, strict=True):
        assert o.shape == d.shape
        assert o.dtype == np.float32


def test_get_compressor_hook_dispatch():
    """post_process=True + 'fedmaq' -> new hook; post_process=False -> plain FedPAQ;
    post_process=True on any non-'fedmaq' alg_name is ignored (defensive)."""
    fedmaq_on = get_compressor_hook("fedmaq", {"post_process": True, "q_min": 2})
    assert isinstance(fedmaq_on, FedMAQPostProcessCompressionHook)

    fedmaq_off = get_compressor_hook("fedmaq", {"post_process": False, "q_min": 2})
    assert isinstance(fedmaq_off, FedPAQCompressionHook)
    assert not isinstance(fedmaq_off, FedMAQPostProcessCompressionHook)

    other_alg = get_compressor_hook("fedpaq", {"post_process": True, "q": 8})
    assert isinstance(other_alg, FedPAQCompressionHook)
    assert not isinstance(other_alg, FedMAQPostProcessCompressionHook)
