"""Regression test for F10: FedKD SVD rank collapse (docs/audits/distillation-direction-audit.md).

Without a rank floor, energy->rank is non-monotonic on concentrated spectra:
as the round-scheduled energy target rises, the retained rank can still
collapse toward 1 well past the low-energy rounds. min_rank_frac fixes this
by flooring the retained rank as a fraction of full rank.
"""

import numpy as np

from fedmaq.baselines.compression import compress_tensor
from fedmaq.core.strategy_hooks.fedkd import FedKDHook


def _concentrated_spectrum_matrix(n: int = 32, m: int = 9, seed: int = 0) -> np.ndarray:
    """Matrix with a dominant leading singular value (mimics depthwise-conv deltas)."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((n, m)))
    v, _ = np.linalg.qr(rng.standard_normal((m, m)))
    sigma = np.array([10.0] + [0.05] * (m - 1))
    return (u * sigma) @ v.T


def test_low_energy_without_floor_collapses_to_rank_one():
    mat = _concentrated_spectrum_matrix()
    u, sigma, v = compress_tensor(mat, energy=0.5)
    assert sigma.size == 1


def test_min_rank_frac_floors_retained_rank():
    mat = _concentrated_spectrum_matrix()
    full_rank = min(mat.shape)
    u, sigma, v = compress_tensor(mat, energy=0.5, min_rank_frac=0.25)
    assert sigma.size >= int(np.ceil(0.25 * full_rank))


def test_min_rank_frac_zero_is_a_noop():
    mat = _concentrated_spectrum_matrix()
    baseline = compress_tensor(mat, energy=0.5)
    floored = compress_tensor(mat, energy=0.5, min_rank_frac=0.0)
    assert baseline[1].size == floored[1].size


def test_min_rank_frac_never_exceeds_full_rank():
    mat = _concentrated_spectrum_matrix()
    full_rank = min(mat.shape)
    u, sigma, v = compress_tensor(mat, energy=0.99, min_rank_frac=1.5)
    assert sigma.size == full_rank


def test_download_payload_bytes_pinned_golden():
    """Golden check that payload_bytes is the raw pre-encoding serialization
    size, not the zlib-measured transmitted size returned alongside it — the
    entire point of #25's payload_bytes companion (see download_size_bytes's
    docstring). Pinned from shapes/dtype directly rather than by re-deriving
    it through svd_payload/measure_bytes, so a regression that leaks the
    measured size into payload_bytes cannot pass by construction.

    At energy=0.5 the concentrated-spectrum matrix (32x9) retains rank 1
    (see test_low_energy_without_floor_collapses_to_rank_one), so the SVD
    factors are U(32x1) + Sigma(1) + V(1x9) = 42 float32 values = 168 bytes.
    """
    hook = FedKDHook({"algorithm": {}})
    hook._current_energy = 0.5

    report = hook.download_size_bytes(None, [_concentrated_spectrum_matrix()])

    assert report.payload_bytes == 168


def test_download_size_bytes_payload_bytes_zero_for_empty_ndarrays():
    hook = FedKDHook({"algorithm": {}})
    report = hook.download_size_bytes(None, [])
    assert report.measured_bytes == 0
    assert report.payload_bytes == 0


def test_get_eval_metrics_surfaces_download_payload_bytes_after_download_size_bytes():
    hook = FedKDHook({"algorithm": {}})
    hook._current_energy = 0.5

    assert "algorithm/fedkd/download_payload_bytes" not in hook.get_eval_metrics(None, 0)

    report = hook.download_size_bytes(None, [_concentrated_spectrum_matrix(seed=2)])

    metrics = hook.get_eval_metrics(None, 1)
    assert metrics["algorithm/fedkd/download_payload_bytes"] == report.payload_bytes
    assert "algorithm/fedkd/download_payload_bytes" in hook.metric_keys()
