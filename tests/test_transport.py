"""Unit tests for the held-constant transport seam (#25)."""

import zlib

import numpy as np

from fedmaq.baselines.transport import measure_bytes


def test_measure_bytes_matches_zlib_compress_length():
    payload = b"\x00" * 256
    assert measure_bytes(payload) == len(zlib.compress(payload))


def test_measure_bytes_is_deterministic():
    payload = np.linspace(-1, 1, 128).astype(np.float32).tobytes()
    assert measure_bytes(payload) == measure_bytes(payload)


def test_measure_bytes_compressible_smaller_than_incompressible():
    compressible = bytes(1024)
    rng = np.random.default_rng(0)
    incompressible = rng.integers(0, 256, size=1024, dtype=np.uint8).tobytes()
    assert measure_bytes(compressible) < measure_bytes(incompressible)


def test_measure_bytes_empty_payload_is_finite():
    assert measure_bytes(b"") > 0
