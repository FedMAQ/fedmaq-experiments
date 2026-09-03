"""Versioned packed wire protocol and codec for quantized tensors (#88).

Replaces nominal int64 array serialization with a versioned, round-trippable
packed wire representation for FedPAQ and FedMAQ quantization.
"""

from __future__ import annotations

import math
import struct
from typing import NamedTuple

import numpy as np

WIRE_MAGIC: bytes = b"FMQ1"
WIRE_VERSION: int = 1
HEADER_STRUCT: struct.Struct = struct.Struct("<4sBBBBIf")
HEADER_SIZE: int = HEADER_STRUCT.size

FLAG_DIFF: int = 0x01
FLAG_CUSTOM_LEVELS: int = 0x02


class WireTensor(NamedTuple):
    """Deserialized payload from the packed wire protocol."""

    codes: np.ndarray
    scale: float
    q: int
    levels: int
    bit_width: int
    is_diff: bool


def symmetric_levels(q: int) -> int:
    """Return the positive levels for a symmetric ``q``-bit quantizer ($2^{q-1}-1$)."""
    if q < 1:
        raise ValueError(f"Quantization bit-width q must be >= 1, got {q}")
    return max(1, (1 << (q - 1)) - 1)


def required_bit_width(max_unsigned_value: int) -> int:
    """Return the minimum number of bits needed to store ``max_unsigned_value``."""
    if max_unsigned_value < 0:
        raise ValueError(f"max_unsigned_value must be non-negative, got {max_unsigned_value}")
    if max_unsigned_value == 0:
        return 1
    return int(max_unsigned_value).bit_length()


def pack_bits(values: np.ndarray, bit_width: int) -> bytes:
    """Pack unsigned integer values of ``bit_width`` bits into little-endian bytes (LSB-first)."""
    if values.size == 0:
        return b""
    if bit_width < 1 or bit_width > 33:
        # 33 = 32 (max protocol quantization bit-width, control_messages.assigned_q)
        # + 1 extra bit that differential codes need for their doubled alphabet [-2L, 2L].
        raise ValueError(f"Unsupported bit_width {bit_width}; must lie in [1, 33]")

    if bit_width == 8:
        return values.astype(np.uint8).tobytes()
    if bit_width == 16:
        return values.astype("<u2").tobytes()
    if bit_width == 32:
        return values.astype("<u4").tobytes()

    vals = values.ravel().astype(np.uint64)
    out = bytearray()
    bit_buf = 0
    bit_count = 0
    mask = (1 << bit_width) - 1

    for v in vals:
        v_int = int(v) & mask
        bit_buf |= v_int << bit_count
        bit_count += bit_width
        while bit_count >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    if bit_count > 0:
        out.append(bit_buf & 0xFF)

    return bytes(out)


def unpack_bits(buf: bytes, num_elements: int, bit_width: int) -> np.ndarray:
    """Unpack unsigned integer values from little-endian bytes (LSB-first)."""
    if num_elements == 0:
        return np.zeros(0, dtype=np.int64)
    if bit_width < 1 or bit_width > 33:
        # Same bound as pack_bits; see its comment for the 33 derivation.
        raise ValueError(f"Unsupported bit_width {bit_width}; must lie in [1, 33]")

    expected_len = (num_elements * bit_width + 7) // 8
    if len(buf) < expected_len:
        raise ValueError(
            f"Packed buffer too short: expected at least {expected_len} bytes for "
            f"{num_elements} elements of {bit_width} bits, got {len(buf)} bytes."
        )

    if bit_width == 8:
        return np.frombuffer(buf[:num_elements], dtype=np.uint8).astype(np.int64)
    if bit_width == 16:
        return np.frombuffer(buf[: num_elements * 2], dtype="<u2").astype(np.int64)
    if bit_width == 32:
        return np.frombuffer(buf[: num_elements * 4], dtype="<u4").astype(np.int64)

    out = np.empty(num_elements, dtype=np.int64)
    bit_buf = 0
    bit_count = 0
    mask = (1 << bit_width) - 1
    byte_idx = 0

    for i in range(num_elements):
        while bit_count < bit_width:
            bit_buf |= buf[byte_idx] << bit_count
            byte_idx += 1
            bit_count += 8
        out[i] = bit_buf & mask
        bit_buf >>= bit_width
        bit_count -= bit_width

    return out


def pack_quantized_tensor(
    codes: np.ndarray,
    scale: float,
    q: int | None = None,
    levels: int | None = None,
    is_diff: bool = False,
) -> bytes:
    """Pack quantized integer codes plus float32 scale into a versioned wire payload.

    Parameters
    ----------
    codes : np.ndarray
        Integer codes in [-L, L] for direct codes or [-2L, 2L] for diff codes.
    scale : float
        Float32 normalization scale.
    q : int | None
        Quantization bit-width (for power-of-2 FedPAQ quantizers). If provided,
        positive levels L = symmetric_levels(q) and bit_width = q (direct) or q+1 (diff).
    levels : int | None
        Positive quantization levels (for arbitrary-level quantizers like DAdaQuant).
        If provided without q, bit_width is derived as ceil(log2(2L+1)) (direct) or
        ceil(log2(4L+1)) (diff).
    is_diff : bool
        Whether codes represent differential codes against a prior round.
    """
    scale_val = float(scale)
    if not math.isfinite(scale_val):
        raise ValueError(f"Quantization scale must be finite, got {scale_val}")

    if q is not None and levels is not None:
        if q < 1:
            raise ValueError(f"Quantization bit-width must be >= 1, got {q}")
        if symmetric_levels(q) != levels:
            raise ValueError(f"Both q={q} and levels={levels} given but they conflict.")
        L = levels
        nominal_q = q
        is_custom = False
    elif q is not None:
        if q < 1:
            raise ValueError(f"Quantization bit-width must be >= 1, got {q}")
        L = symmetric_levels(q)
        nominal_q = q
        is_custom = False
    elif levels is not None:
        if levels < 1 or levels > 255:
            raise ValueError(f"Custom quantization levels must lie in [1, 255], got {levels}")
        L = levels
        nominal_q = levels
        is_custom = True
    else:
        raise ValueError("Either q or levels must be provided to pack_quantized_tensor.")

    codes_flat = codes.ravel()
    num_elements = codes_flat.size

    flags = 0
    if is_diff:
        flags |= FLAG_DIFF
    if is_custom:
        flags |= FLAG_CUSTOM_LEVELS

    multiplier = 2 if is_diff else 1
    bound = multiplier * L
    max_val = 2 * bound
    if nominal_q >= 2 and not is_custom:
        bit_width = (nominal_q + 1) if is_diff else nominal_q
    else:
        bit_width = required_bit_width(max_val)

    if num_elements > 0:
        min_code = int(np.min(codes_flat))
        max_code = int(np.max(codes_flat))
        if min_code < -bound or max_code > bound:
            prefix = "Differential code" if is_diff else "Code"
            suffix = f" for symmetric quantization (L={L})." if not is_diff else f" (L={L})."
            raise ValueError(
                f"{prefix} values [{min_code}, {max_code}] exceed alphabet bound "
                f"[-{bound}, {bound}]{suffix}"
            )
        unsigned_codes = (codes_flat + bound).astype(np.int64)
    else:
        unsigned_codes = np.zeros(0, dtype=np.int64)

    header = HEADER_STRUCT.pack(
        WIRE_MAGIC,
        WIRE_VERSION,
        nominal_q,
        flags,
        bit_width,
        num_elements,
        scale_val,
    )

    packed_data = pack_bits(unsigned_codes, bit_width) if num_elements > 0 else b""
    return header + packed_data


def unpack_quantized_tensor(buf: bytes) -> WireTensor:
    """Unpack a versioned wire payload back into codes and metadata.

    Raises ValueError on any header mismatch, truncation, non-finite scale,
    out-of-bound codes, or trailing garbage bytes.
    """
    if len(buf) < HEADER_SIZE:
        raise ValueError(
            f"Buffer too short for wire header: expected at least {HEADER_SIZE} bytes, "
            f"got {len(buf)} bytes."
        )

    magic, version, nominal_q, flags, bit_width, num_elements, scale = HEADER_STRUCT.unpack_from(
        buf, 0
    )

    if magic != WIRE_MAGIC:
        raise ValueError(f"Invalid wire magic: expected {WIRE_MAGIC!r}, got {magic!r}")
    if version != WIRE_VERSION:
        raise ValueError(f"Unsupported wire version: expected {WIRE_VERSION}, got {version}")
    if not math.isfinite(scale):
        raise ValueError(f"Wire scale is non-finite: {scale}")
    if bit_width < 1 or bit_width > 33:
        raise ValueError(f"Wire bit_width {bit_width} outside supported range [1, 33]")

    is_diff = bool(flags & FLAG_DIFF)
    is_custom = bool(flags & FLAG_CUSTOM_LEVELS)

    if is_custom:
        L = nominal_q
    elif nominal_q > 0:
        L = symmetric_levels(nominal_q)
    else:
        # nominal_q == 0 is never emitted by pack_quantized_tensor (q and levels are
        # both validated >= 1 there), so this branch is unreachable via any payload
        # this module itself writes. Kept so the decoder stays total on any header
        # satisfying the struct format, deriving L from bit_width alone -- delete only
        # after confirming no other wire producer relies on this fallback.
        if not is_diff:
            L = (1 << (bit_width - 1)) if bit_width > 1 else 1
        else:
            L = (1 << (bit_width - 2)) if bit_width > 2 else 1

    expected_data_len = (num_elements * bit_width + 7) // 8 if num_elements > 0 else 0
    total_expected_len = HEADER_SIZE + expected_data_len
    if len(buf) != total_expected_len:
        if len(buf) < total_expected_len:
            raise ValueError(
                f"Wire payload truncated: expected {total_expected_len} bytes "
                f"({HEADER_SIZE} header + {expected_data_len} data), got {len(buf)} bytes."
            )
        raise ValueError(
            f"Wire buffer contains {len(buf) - total_expected_len} trailing unconsumed bytes."
        )

    if num_elements == 0:
        return WireTensor(
            codes=np.zeros(0, dtype=np.int64),
            scale=scale,
            q=nominal_q,
            levels=L,
            bit_width=bit_width,
            is_diff=is_diff,
        )

    unsigned_codes = unpack_bits(buf[HEADER_SIZE:], num_elements, bit_width)

    multiplier = 2 if is_diff else 1
    bound = multiplier * L
    codes = unsigned_codes - bound
    if nominal_q > 0 and np.any(unsigned_codes > 2 * bound):
        kind = "differential code" if is_diff else "unsigned code"
        raise ValueError(f"Decoded {kind} exceeds alphabet limit {2 * bound} for q={nominal_q}")

    return WireTensor(
        codes=codes,
        scale=scale,
        q=nominal_q,
        levels=L,
        bit_width=bit_width,
        is_diff=is_diff,
    )
