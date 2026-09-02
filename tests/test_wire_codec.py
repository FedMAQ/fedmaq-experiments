"""Unit tests for the packed wire codec and protocol (#88).

Tests header specification, bit packing/unpacking, symmetric and differential
alphabet mappings, known vectors, randomized round-trips, empty payloads,
malformed input rejections, and byte-conservation invariants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedmaq.core.wire_codec import (
    HEADER_SIZE,
    WIRE_MAGIC,
    WIRE_VERSION,
    pack_bits,
    pack_quantized_tensor,
    required_bit_width,
    symmetric_levels,
    unpack_bits,
    unpack_quantized_tensor,
)


def test_symmetric_levels_matches_specification() -> None:
    assert symmetric_levels(1) == 1
    assert symmetric_levels(2) == 1
    assert symmetric_levels(3) == 3
    assert symmetric_levels(4) == 7
    assert symmetric_levels(8) == 127
    assert symmetric_levels(16) == 32767
    assert symmetric_levels(32) == 2147483647

    with pytest.raises(ValueError, match="q must be >= 1"):
        symmetric_levels(0)


def test_required_bit_width_bounds() -> None:
    assert required_bit_width(0) == 1
    assert required_bit_width(1) == 1
    assert required_bit_width(2) == 2
    assert required_bit_width(3) == 2
    assert required_bit_width(7) == 3
    assert required_bit_width(15) == 4
    assert required_bit_width(254) == 8
    assert required_bit_width(255) == 8
    assert required_bit_width(256) == 9

    with pytest.raises(ValueError, match="non-negative"):
        required_bit_width(-1)


def test_bit_packing_known_vectors() -> None:
    # 2-bit values: [0, 1, 2, 3] -> (0 | (1<<2) | (2<<4) | (3<<6)) = 0b11100100 = 0xE4
    vals_2bit = np.array([0, 1, 2, 3], dtype=np.int64)
    packed_2bit = pack_bits(vals_2bit, 2)
    assert packed_2bit == bytes([0xE4])
    np.testing.assert_array_equal(unpack_bits(packed_2bit, 4, 2), vals_2bit)

    # 3-bit values: [1, 2, 4] ->
    # val0: 0b001 (bits 0..2)
    # val1: 0b010 (bits 3..5)
    # val2: 0b100 (bits 6..8 -> bit 6,7 in byte 0, bit 8 in byte 1)
    # byte 0: 0b001 | (0b010 << 3) | (0b00 << 6) = 1 | 16 | 0 = 17 = 0x11
    # byte 1: 0b001 = 1 = 0x01
    vals_3bit = np.array([1, 2, 4], dtype=np.int64)
    packed_3bit = pack_bits(vals_3bit, 3)
    assert packed_3bit == bytes([0x11, 0x01])
    np.testing.assert_array_equal(unpack_bits(packed_3bit, 3, 3), vals_3bit)


def test_pack_quantized_tensor_header_layout() -> None:
    codes = np.array([0, 1, -1], dtype=np.int64)
    scale = 2.5
    payload = pack_quantized_tensor(codes, scale=scale, q=4)

    assert len(payload) == HEADER_SIZE + 2  # 3 elements * 4 bits = 12 bits -> 2 bytes
    assert payload[:4] == WIRE_MAGIC
    assert payload[4] == WIRE_VERSION
    assert payload[5] == 4  # nominal q
    assert payload[6] == 0  # flags (not diff)
    assert payload[7] == 4  # bit_width


def test_pack_unpack_known_quantized_vectors() -> None:
    # q = 2: L = 1, alphabet [-1, 0, 1], offset codes [0, 1, 2], bit_width = 2
    codes_q2 = np.array([-1, 0, 1, 0], dtype=np.int64)
    payload_q2 = pack_quantized_tensor(codes_q2, scale=1.0, q=2)
    unpacked_q2 = unpack_quantized_tensor(payload_q2)

    assert unpacked_q2.q == 2
    assert unpacked_q2.bit_width == 2
    assert unpacked_q2.scale == 1.0
    assert unpacked_q2.is_diff is False
    np.testing.assert_array_equal(unpacked_q2.codes, codes_q2)

    # q = 8: L = 127, alphabet [-127..127], bit_width = 8
    codes_q8 = np.array([-127, -50, 0, 50, 127], dtype=np.int64)
    payload_q8 = pack_quantized_tensor(codes_q8, scale=0.125, q=8)
    unpacked_q8 = unpack_quantized_tensor(payload_q8)

    assert unpacked_q8.q == 8
    assert unpacked_q8.bit_width == 8
    assert math.isclose(unpacked_q8.scale, 0.125, rel_tol=1e-6)
    np.testing.assert_array_equal(unpacked_q8.codes, codes_q8)


def test_differential_packing_uses_q_plus_1_bits() -> None:
    # q = 4: L = 7, diff alphabet [-14..14], bit_width = 5 (4+1)
    diff_codes = np.array([-14, -7, 0, 7, 14], dtype=np.int64)
    payload = pack_quantized_tensor(diff_codes, scale=0.5, q=4, is_diff=True)

    assert len(payload) == HEADER_SIZE + (5 * 5 + 7) // 8  # 16 + 4 = 20 bytes
    unpacked = unpack_quantized_tensor(payload)
    assert unpacked.is_diff is True
    assert unpacked.bit_width == 5
    assert unpacked.q == 4
    np.testing.assert_array_equal(unpacked.codes, diff_codes)


def test_custom_level_quantizer_for_dadaquant() -> None:
    # DAdaQuant with 127 levels per sign (255 total levels -> 8 bits)
    codes = np.array([-127, -64, 0, 64, 127], dtype=np.int64)
    payload = pack_quantized_tensor(codes, scale=1.5, levels=127)

    assert len(payload) == HEADER_SIZE + 5  # 5 elements * 8 bits = 5 bytes
    unpacked = unpack_quantized_tensor(payload)
    assert unpacked.bit_width == 8
    assert math.isclose(unpacked.scale, 1.5, rel_tol=1e-6)
    np.testing.assert_array_equal(unpacked.codes, codes)


def test_empty_tensor_roundtrip() -> None:
    empty = np.zeros(0, dtype=np.int64)
    payload = pack_quantized_tensor(empty, scale=0.0, q=8)
    assert len(payload) == HEADER_SIZE

    unpacked = unpack_quantized_tensor(payload)
    assert unpacked.codes.size == 0
    assert unpacked.scale == 0.0
    assert unpacked.q == 8
    assert unpacked.bit_width == 8


@pytest.mark.parametrize("q", [2, 3, 4, 5, 6, 7, 8, 16, 32])
def test_randomized_roundtrip_all_supported_q(q: int) -> None:
    rng = np.random.default_rng(q * 100)
    L = symmetric_levels(q)
    size = 127

    codes = rng.integers(-L, L + 1, size=size, dtype=np.int64)
    scale = float(rng.uniform(0.01, 10.0))

    payload = pack_quantized_tensor(codes, scale=scale, q=q, is_diff=False)
    expected_data_len = (size * q + 7) // 8
    assert len(payload) == HEADER_SIZE + expected_data_len

    unpacked = unpack_quantized_tensor(payload)
    assert unpacked.q == q
    assert unpacked.bit_width == q
    assert math.isclose(unpacked.scale, scale, rel_tol=1e-5)
    assert unpacked.is_diff is False
    np.testing.assert_array_equal(unpacked.codes, codes)


@pytest.mark.parametrize("q", [2, 3, 4, 5, 6, 7, 8, 16])
def test_randomized_differential_roundtrip(q: int) -> None:
    rng = np.random.default_rng(q * 200)
    L = symmetric_levels(q)
    size = 100

    diffs = rng.integers(-2 * L, 2 * L + 1, size=size, dtype=np.int64)
    scale = float(rng.uniform(0.01, 5.0))

    payload = pack_quantized_tensor(diffs, scale=scale, q=q, is_diff=True)
    b = q + 1
    expected_data_len = (size * b + 7) // 8
    assert len(payload) == HEADER_SIZE + expected_data_len

    unpacked = unpack_quantized_tensor(payload)
    assert unpacked.is_diff is True
    assert unpacked.bit_width == b
    np.testing.assert_array_equal(unpacked.codes, diffs)


def test_malformed_buffer_too_short_raises() -> None:
    with pytest.raises(ValueError, match="Buffer too short"):
        unpack_quantized_tensor(b"FMQ1\x01\x00")


def test_malformed_magic_mismatch_raises() -> None:
    valid_payload = pack_quantized_tensor(np.array([0, 1], dtype=np.int64), scale=1.0, q=4)
    corrupted = b"BAD!" + valid_payload[4:]
    with pytest.raises(ValueError, match="Invalid wire magic"):
        unpack_quantized_tensor(corrupted)


def test_malformed_version_mismatch_raises() -> None:
    valid_payload = pack_quantized_tensor(np.array([0, 1], dtype=np.int64), scale=1.0, q=4)
    corrupted = valid_payload[:4] + bytes([99]) + valid_payload[5:]
    with pytest.raises(ValueError, match="Unsupported wire version"):
        unpack_quantized_tensor(corrupted)


def test_malformed_non_finite_scale_raises() -> None:
    with pytest.raises(ValueError, match="scale must be finite"):
        pack_quantized_tensor(np.array([0], dtype=np.int64), scale=float("nan"), q=4)
    with pytest.raises(ValueError, match="scale must be finite"):
        pack_quantized_tensor(np.array([0], dtype=np.int64), scale=float("inf"), q=4)


def test_out_of_alphabet_codes_raise_on_pack() -> None:
    # q = 2: L = 1 -> valid codes are [-1, 0, 1]
    with pytest.raises(ValueError, match="exceed alphabet bound"):
        pack_quantized_tensor(np.array([-2, 0, 1], dtype=np.int64), scale=1.0, q=2)

    with pytest.raises(ValueError, match="exceed alphabet bound"):
        pack_quantized_tensor(np.array([-1, 0, 2], dtype=np.int64), scale=1.0, q=2)


def test_trailing_bytes_raise_on_unpack() -> None:
    valid_payload = pack_quantized_tensor(np.array([0, 1], dtype=np.int64), scale=1.0, q=4)
    with pytest.raises(ValueError, match="trailing unconsumed bytes"):
        unpack_quantized_tensor(valid_payload + b"EXTRA")


def test_byte_conservation_exact_formula() -> None:
    """Verifies that serialized byte length matches header size + ceil(N*b/8)."""
    for q in [2, 3, 4, 5, 6, 7, 8, 16, 32]:
        for n in [0, 1, 7, 8, 9, 15, 16, 31, 32, 100]:
            codes = np.zeros(n, dtype=np.int64)
            payload = pack_quantized_tensor(codes, scale=1.0, q=q)
            expected = HEADER_SIZE + (n * q + 7) // 8
            assert len(payload) == expected, f"Failed for q={q}, n={n}"


def test_q1_ternary_quantization_round_trip() -> None:
    """q=1 uses L=1 with 2 bits to pack ternary codes [-1, 0, 1]."""
    codes = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int64)
    payload = pack_quantized_tensor(codes, scale=2.5, q=1)
    unpacked = unpack_quantized_tensor(payload)
    assert unpacked.q == 1
    assert unpacked.bit_width == 2
    assert unpacked.is_diff is False
    np.testing.assert_array_equal(unpacked.codes, codes)
    assert unpacked.scale == pytest.approx(2.5)


def test_unsupported_q_less_than_1_raises_on_pack() -> None:
    """Quantizer requires q >= 1."""
    with pytest.raises(ValueError, match="must be >= 1"):
        pack_quantized_tensor(np.array([0, 1], dtype=np.int64), scale=1.0, q=0)


