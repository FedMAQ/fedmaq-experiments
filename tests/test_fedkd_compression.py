"""Regression test for F10: FedKD SVD rank collapse (docs/audits/distillation-direction-audit.md).

Without a rank floor, energy->rank is non-monotonic on concentrated spectra:
as the round-scheduled energy target rises, the retained rank can still
collapse toward 1 well past the low-energy rounds. min_rank_frac fixes this
by flooring the retained rank as a fraction of full rank.
"""

import numpy as np

from fedmaq.baselines.compression import compress_tensor


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
