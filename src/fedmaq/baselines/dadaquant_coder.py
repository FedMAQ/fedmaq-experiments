"""DAdaQuant's as-published transport coder: 0-run-length + Elias omega (#26).

Hönig et al. 2022 specifies 0-run-length encoding (runs of zero-valued
quantization codes collapsed to their length) followed by Elias omega coding
(a universal code for positive integers, with no fixed maximum -- it adapts to
whatever bit-width ``q_t`` reaches that round) as DAdaQuant's transport stage.
The primary byte axis (:mod:`fedmaq.baselines.transport`) holds one encoder
(zlib) constant across every arm by design; this is a second, parallel
measurement specific to DAdaQuant, logged alongside rather than substituted
in, so the primary comparison stays isolated to compression *policy* while
this axis answers the "but DAdaQuant's own paper compresses further" question
directly.

Audit note (2026-08-26 baseline-implementation audit, X1 finding): DAdaQuant
is the only baseline in the stack whose source paper mandates a transport
coder beyond the quantization step itself -- FedPAQ, FedKD, FedDistill,
FedAvg and FedProx specify no such stage. "Difference coding" (the paper's
third component alongside 0-RLE and Elias omega) is not implemented here
because it is already structural: every arm quantizes parameter *deltas*
(the difference between incoming and locally-updated weights), not raw
weights, so DAdaQuant's difference-coding requirement is satisfied by the
existing client update path rather than by anything this module adds.

Scale metadata (the per-tensor float32 normalization factor) travels
alongside the coded payload as 4 raw bytes, uncoded -- matching every other
baseline's ``+4`` convention (see ``quantization.py::_serialize_codes``) and
the paper's own coder, which targets quantization codes, not a single
per-tensor float already carried at nearly its Shannon-optimal size.
"""

import numpy as np


class _BitWriter:
    """Accumulates a bitstream MSB-first; byte-aligned only when read out."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._buf.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def write_bits(self, value: int, length: int) -> None:
        """Write the low ``length`` bits of ``value``, most-significant first."""
        for shift in range(length - 1, -1, -1):
            self.write_bit((value >> shift) & 1)

    def to_bytes(self) -> bytes:
        """Return the bitstream so far, zero-padded to a whole number of bytes."""
        if self._nbits == 0:
            return bytes(self._buf)
        return bytes(self._buf) + bytes([self._cur << (8 - self._nbits)])


class _BitReader:
    """Reads a bitstream MSB-first from :meth:`_BitWriter.to_bytes` output."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte_idx, bit_idx = divmod(self._pos, 8)
        bit = (self._data[byte_idx] >> (7 - bit_idx)) & 1
        self._pos += 1
        return bit

    def read_bits(self, length: int) -> int:
        value = 0
        for _ in range(length):
            value = (value << 1) | self.read_bit()
        return value


def _elias_omega_encode_into(writer: _BitWriter, n: int) -> None:
    """Append ``n``'s Elias omega code (``n >= 1``, self-delimiting) to ``writer``."""
    if n < 1:
        raise ValueError(f"Elias omega coding requires n >= 1, got {n}.")
    groups: list[tuple[int, int]] = []
    while n > 1:
        bit_length = n.bit_length()
        groups.append((n, bit_length))
        n = bit_length - 1
    for value, length in reversed(groups):
        writer.write_bits(value, length)
    writer.write_bit(0)


def _elias_omega_decode_from(reader: _BitReader) -> int:
    """Inverse of :func:`_elias_omega_encode_into`."""
    n = 1
    while reader.read_bit() == 1:
        rest = reader.read_bits(n)
        n = (1 << n) | rest
    return n


def _zero_runs_and_values(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Decompose ``flat`` into (zero-run lengths, values, trailing zero-run).

    ``run_lengths[i]`` zeros immediately precede ``values[i]``; the trailing
    count covers any zeros after the last nonzero value (0 if there is none,
    e.g. the array ends on a nonzero value or has no nonzero values at all --
    see the ``nz_idx.size == 0`` branch). Vectorized via ``np.flatnonzero``
    rather than a per-element Python loop, since DAdaQuant's stochastic
    rounding is expected to produce codes that are mostly zero -- the
    sparsity this coder is meant to exploit.
    """
    nz_idx = np.flatnonzero(flat)
    if nz_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), flat.size

    prev = np.concatenate(([-1], nz_idx[:-1]))
    run_lengths = nz_idx - prev - 1
    values = flat[nz_idx]
    trailing = flat.size - (nz_idx[-1] + 1)
    return run_lengths, values, int(trailing)


def dadaquant_pack(codes: np.ndarray) -> bytes:
    """Encode integer quantization codes via 0-RLE + Elias omega.

    ``codes`` is flattened (row-major); the returned bytes do not encode
    shape or element count -- callers already carry that externally (the
    same convention every other coder in this codebase follows), and
    :func:`dadaquant_unpack` needs it back to know when to stop.
    """
    flat = np.asarray(codes).ravel()
    run_lengths, values, trailing = _zero_runs_and_values(flat)

    writer = _BitWriter()
    for run, value in zip(run_lengths.tolist(), values.tolist(), strict=True):
        _elias_omega_encode_into(writer, run + 1)
        writer.write_bit(1 if value < 0 else 0)
        _elias_omega_encode_into(writer, abs(value))
    if trailing > 0:
        _elias_omega_encode_into(writer, trailing + 1)
    return writer.to_bytes()


def dadaquant_unpack(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of :func:`dadaquant_pack`. Returns a flat ``int64`` array."""
    reader = _BitReader(data)
    codes = np.zeros(n_elements, dtype=np.int64)
    pos = 0
    while pos < n_elements:
        run = _elias_omega_decode_from(reader) - 1
        pos += run
        if pos == n_elements:
            break
        sign = reader.read_bit()
        magnitude = _elias_omega_decode_from(reader)
        codes[pos] = -magnitude if sign else magnitude
        pos += 1
    return codes
