"""Unit tests for the held-constant transport seam (#25)."""

import bz2
import zlib

import numpy as np

from fedmaq.baselines.transport import UploadReport, measure_bytes, pack_payloads, unpack_payloads


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


def test_pack_unpack_payloads_roundtrip():
    payloads = [b"", b"a", b"\x00" * 300, np.arange(10, dtype=np.int64).tobytes()]
    assert unpack_payloads(pack_payloads(payloads)) == payloads


def test_pack_unpack_payloads_empty_list():
    assert unpack_payloads(pack_payloads([])) == []


def test_upload_report_from_payloads_preserves_measurement_boundaries():
    payloads = (b"A" * 20, b"B" * 30)

    report = UploadReport.from_payloads(payloads, secondary_bytes=17)

    assert report.payloads == payloads
    assert report.payload_bytes == sum(map(len, payloads))
    assert report.measured_bytes == sum(map(measure_bytes, payloads))
    assert report.secondary_bytes == 17


def test_summed_measure_bytes_differs_from_measuring_the_concatenation():
    """AC 2's reproducibility target is the per-call sum every hook actually logs
    (one ``measure_bytes`` call per tensor/leg), not one call over a concatenated
    blob -- compression is not additive over concatenation, so persisting only a
    concatenated payload would silently change what gets reproduced."""
    payloads = [b"AAAAAAAAAAAAAAAAAAAA", b"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"]
    summed = sum(measure_bytes(p) for p in payloads)
    concatenated = measure_bytes(b"".join(payloads))
    assert summed != concatenated


def test_unpacked_payloads_replay_under_an_alternate_encoder():
    """Demonstrates AC 2's actual claim: given only the persisted payloads (no
    training re-run), a hypothetical alternate encoder can be scored against the
    exact original per-call totals."""
    payloads = [
        bytes(200),
        np.random.default_rng(0).integers(0, 256, size=200, dtype=np.uint8).tobytes(),
    ]
    framed = pack_payloads(payloads)

    replayed = unpack_payloads(framed)
    assert replayed == payloads

    original_total = sum(measure_bytes(p) for p in replayed)
    alternate_total = sum(len(bz2.compress(p)) for p in replayed)
    assert original_total != alternate_total
