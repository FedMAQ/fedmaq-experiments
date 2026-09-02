"""Unit tests for FedMAQ's post-processing pipeline (error-feedback + diff-coding + zlib)."""

import numpy as np
import pytest
from flwr.app import RecordDict

from fedmaq.baselines import FedMAQPostProcessCompressionHook, get_compressor_hook
from fedmaq.baselines.quantization import FedPAQCompressionHook
from fedmaq.baselines.transport import measure_bytes


def test_round1_stochastic_rounding_is_unbiased():
    """E[Q(x)] = x at round 1 (fresh state, so error-feedback/diff-coding are
    no-ops), verified statistically (#24 AC), not asserted from a docstring.

    Postprocess keeps l∞ scale (not FedPAQ's l2 switch -- see the comment in
    postprocess.py explaining why error feedback + l2 diverges), so this checks
    the hook's own operator directly rather than comparing against FedPAQ.
    """
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(2048).astype(np.float32)
    n_draws = 200

    accum = np.zeros_like(delta, dtype=np.float64)
    for i in range(n_draws):
        hook = FedMAQPostProcessCompressionHook(q=4, rng=np.random.default_rng(1000 + i))
        out, _ = hook.compress([delta.copy()])
        accum += out[0]
    mean_estimate = accum / n_draws

    rel_err = np.linalg.norm(mean_estimate - delta) / np.linalg.norm(delta)
    assert rel_err < 0.05


def test_round1_same_seed_twice_is_bit_identical():
    """Determinism under a fixed seed (ADR-0006 surrogate)."""
    delta = np.linspace(-1, 1, 64).astype(np.float32)

    out_a, report_a = FedMAQPostProcessCompressionHook(q=4, rng=np.random.default_rng(5)).compress(
        [delta.copy()]
    )
    out_b, report_b = FedMAQPostProcessCompressionHook(q=4, rng=np.random.default_rng(5)).compress(
        [delta.copy()]
    )

    np.testing.assert_array_equal(out_a[0], out_b[0])
    assert report_a.measured_bytes == report_b.measured_bytes


def test_requires_rng_for_stochastic_rounding():
    """No unseeded fallback (#24): a rng-less hook fails loudly at q>1."""
    hook = FedMAQPostProcessCompressionHook(q=8)
    with pytest.raises(ValueError, match="seeded rng"):
        hook.compress([np.array([-2.0, 0.0, 2.0], dtype=np.float32)])


def test_state_persists_and_diffing_engages_on_second_call():
    """State is populated after one compress() call; a second call reads it back
    and produces a different (diffed) byte count than an isolated fresh-state call.

    The fresh-state comparison hook has its rng state cloned from `hook`'s
    post-round-1 state, so both draw the identical stochastic sequence for
    their next call -- isolating carried state as the only difference, not a
    second independent random draw racing the first (which would make the
    `<=` assertion a coin flip under #24's stochastic rounding).
    """
    state = RecordDict()
    hook = FedMAQPostProcessCompressionHook(q=8, state=state, rng=np.random.default_rng(1))
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(256,)).astype(np.float32)

    assert state.get("fedmaq_postprocess_residual") is None
    hook.compress([delta.copy()])
    assert state.get("fedmaq_postprocess_residual") is not None
    assert state.get("fedmaq_postprocess_prev_codes") is not None

    fresh_hook = FedMAQPostProcessCompressionHook(q=8, rng=np.random.default_rng())
    fresh_hook.rng.bit_generator.state = hook.rng.bit_generator.state

    _, report_round2 = hook.compress([delta.copy()])
    _, report_fresh = fresh_hook.compress([delta.copy()])
    assert report_round2.measured_bytes <= report_fresh.measured_bytes


def test_error_feedback_carries_quantization_error_into_next_round():
    """Round-1 quantization error on a non-extremal element is folded into
    round 2's input via the residual, on the ``q>1`` stochastic branch --
    the one #24 changed (a prior version of this test used ``q=1``'s
    deterministic sign branch, which doesn't exercise stochastic rounding at
    all). Pinned to a fixed seed; determinism itself is covered separately by
    ``test_round1_same_seed_twice_is_bit_identical``.
    """
    state = RecordDict()
    hook = FedMAQPostProcessCompressionHook(q=4, state=state, rng=np.random.default_rng(11))
    d = np.array([1.0, 0.4], dtype=np.float32)  # levels=7, scale=max|d|=1.0 in both rounds

    out_r1, _ = hook.compress([d.copy()])
    # element 0: scaled = 1.0*7 = 7 exactly -> deterministic code 7 -> dequant 1.0.
    # element 1: scaled = 0.4*7 = 2.8 -> floor 2, P(round up to 3) = 0.8; seed
    # 11's first draw rounds up -> dequant 3/7.
    np.testing.assert_allclose(out_r1[0], [1.0, 3 / 7], atol=1e-6)
    residual = state.get("fedmaq_postprocess_residual").to_numpy_ndarrays()[0]
    np.testing.assert_allclose(residual, [0.0, 0.4 - 3 / 7], atol=1e-6)

    out_r2, _ = hook.compress([d.copy()])
    d_fb2 = d + residual
    np.testing.assert_allclose(d_fb2, [1.0, 0.8 - 3 / 7], atol=1e-6)
    # seed 11's second draw also rounds up -> same dequant as round 1.
    np.testing.assert_allclose(out_r2[0], [1.0, 3 / 7], atol=1e-6)


def test_diff_coding_reflects_codes_minus_prev_codes():
    """Seeding state with known prev_codes must change the zlib-measured byte
    count relative to a fresh-state call on the same input (raw codes vs diffed)."""
    delta = np.linspace(-1, 1, 64).astype(np.float32)

    fresh_state = RecordDict()
    fresh_hook = FedMAQPostProcessCompressionHook(
        q=8, state=fresh_state, rng=np.random.default_rng(0)
    )
    _, report_raw = fresh_hook.compress([delta.copy()])
    raw_codes = fresh_state.get("fedmaq_postprocess_prev_codes").to_numpy_ndarrays()[0]

    from flwr.app import ArrayRecord

    from fedmaq.core.wire_codec import pack_quantized_tensor

    seeded_state = RecordDict()
    seeded_state["fedmaq_postprocess_prev_codes"] = ArrayRecord(numpy_ndarrays=[raw_codes.copy()])
    seeded_hook = FedMAQPostProcessCompressionHook(
        q=8, state=seeded_state, rng=np.random.default_rng(0)
    )
    _, report_diffed = seeded_hook.compress([delta.copy()])

    scale = float(np.max(np.abs(delta)))
    all_zero_codes = np.zeros_like(raw_codes)
    assert report_diffed.measured_bytes == measure_bytes(
        pack_quantized_tensor(all_zero_codes, scale, q=8, is_diff=True)
    )
    assert report_diffed.measured_bytes < report_raw.measured_bytes


def test_byte_count_realism():
    """Mostly-zero codes (highly compressible) must measure strictly smaller than
    codes from incompressible (random) input of the same size, and both must be
    finite and bounded."""
    size = 256
    bits = 8

    compressible = np.zeros(size, dtype=np.float32)
    compressible[0] = 1.0  # mostly-zero -> highly compressible codes
    _, report_compressible = FedMAQPostProcessCompressionHook(
        q=bits, rng=np.random.default_rng(1)
    ).compress([compressible])

    rng = np.random.default_rng(2)
    incompressible = rng.normal(size=size).astype(np.float32)
    _, report_incompressible = FedMAQPostProcessCompressionHook(
        q=bits, rng=np.random.default_rng(3)
    ).compress([incompressible])

    assert 0 < report_compressible.measured_bytes < report_incompressible.measured_bytes
    assert report_incompressible.measured_bytes < size * 8 * 2 + 64


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
    hook = FedMAQPostProcessCompressionHook(q=8, state=state, rng=np.random.default_rng(0))

    mismatched_delta = np.ones(10, dtype=np.float32)
    out, report = hook.compress([mismatched_delta])
    assert out[0].shape == (10,)
    assert report.measured_bytes > 0


def test_empty_and_all_zero_tensor_pass_through():
    """Mirrors quantization.py's empty-tensor and all-zero-tensor branches.

    Neither branch reaches the stochastic-rounding path, so no rng is needed.
    """
    from fedmaq.core.wire_codec import pack_quantized_tensor

    hook = FedMAQPostProcessCompressionHook(q=8)

    empty = np.zeros((0,), dtype=np.float32)
    zero = np.zeros((5,), dtype=np.float32)
    out, report = hook.compress([empty, zero])

    assert out[0].shape == (0,)
    np.testing.assert_allclose(out[1], zero)
    expected = measure_bytes(
        pack_quantized_tensor(np.zeros(5, dtype=np.int64), 0.0, q=8, is_diff=False)
    )
    assert report.measured_bytes == expected


def test_output_contract_matches_input_shape_dtype():
    """Output list length/shape/dtype must satisfy standard.py's `o + cd` reconstruction."""
    rng = np.random.default_rng(3)
    deltas = [
        rng.normal(size=(3, 5)).astype(np.float32),
        rng.normal(size=(7,)).astype(np.float32),
        np.zeros((0,), dtype=np.float32),
    ]
    out, _ = FedMAQPostProcessCompressionHook(q=8, rng=np.random.default_rng(4)).compress(deltas)

    assert len(out) == len(deltas)
    for o, d in zip(out, deltas, strict=True):
        assert o.shape == d.shape
        assert o.dtype == np.float32


def test_get_compressor_hook_dispatch():
    """post_process=True + 'fedmaq' -> new hook; post_process=False -> plain FedPAQ;
    'fedpaq_pipeline' -> new hook; post_process=True on ordinary 'fedpaq' is ignored (defensive)."""
    fedmaq_on = get_compressor_hook("fedmaq", {"post_process": True, "q_min": 2})
    assert isinstance(fedmaq_on, FedMAQPostProcessCompressionHook)

    fedmaq_off = get_compressor_hook("fedmaq", {"post_process": False, "q_min": 2})
    assert isinstance(fedmaq_off, FedPAQCompressionHook)
    assert not isinstance(fedmaq_off, FedMAQPostProcessCompressionHook)

    # Ordinary fedpaq ignores post_process=True as defensive invariant
    other_alg = get_compressor_hook("fedpaq", {"post_process": True, "q": 8})
    assert isinstance(other_alg, FedPAQCompressionHook)
    assert not isinstance(other_alg, FedMAQPostProcessCompressionHook)

    # Distinct registered fedpaq_pipeline arm activates the post-processing hook
    fedpaq_pipe = get_compressor_hook("fedpaq_pipeline", {"post_process": True, "q": 8})
    assert isinstance(fedpaq_pipe, FedMAQPostProcessCompressionHook)
    assert fedpaq_pipe.q == 8
