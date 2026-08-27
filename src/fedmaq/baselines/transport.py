"""Held-constant transport: the single byte-measurement seam for every arm (#25).

Before this module, "bytes transmitted" had four independent implementations
(FedPAQ/DAdaQuant's analytic ``ceil(size*bits/8)+4`` formula, FedMAQ's
zlib-measured path, the identity hook's raw ``nbytes``, and FedKD's raw SVD
factor count) plus a fifth undocumented one in FedDistill's client fit. Comparing
a model against a measurement was not a fair comparison, and it ran in the
direction that flattered FedMAQ.

``measure_bytes`` is the one function every arm now routes through. Each arm
keeps its own ``serialize``-shaped logic (payload shape is arm-specific: float32
tensors, int codes + scale, SVD factors) but the encoder itself is payload-agnostic
and identical across arms. Metadata (e.g. a quantization scale) is serialized
*into* the payload handed here, not added outside compression — so there is no
separate additive convention left anywhere for a byte total to route around.

Encoder choice: zlib at its default settings, inherited from the FedMAQ
post-processing path this seam generalizes (no stated reason to change it).
"""

import struct
import zlib

_LENGTH_PREFIX = struct.Struct("<Q")


def measure_bytes(payload: bytes) -> int:
    """Return the transmitted size of ``payload`` under the held-constant transport.

    ``payload`` must already include any metadata (e.g. a quantization scale)
    that would travel over the wire alongside the data — this function does not
    add anything on top of the compressed result.
    """
    return len(zlib.compress(payload))


def pack_payloads(payloads: list[bytes]) -> bytes:
    """Frame a list of pre-encoding payloads into one length-prefixed blob.

    Every :func:`measure_bytes` call site measures one payload per tensor/leg
    and sums the results — ``measure_bytes`` is not additive over concatenation,
    so a single concatenated blob cannot be re-scored against a different
    encoder and reproduce the original per-payload sum (#25 AC 2). Framing
    preserves each call's boundary so :func:`unpack_payloads` can hand every
    payload back individually.
    """
    return b"".join(_LENGTH_PREFIX.pack(len(p)) + p for p in payloads)


def unpack_payloads(buf: bytes) -> list[bytes]:
    """Inverse of :func:`pack_payloads`."""
    payloads = []
    offset = 0
    while offset < len(buf):
        (length,) = _LENGTH_PREFIX.unpack_from(buf, offset)
        offset += _LENGTH_PREFIX.size
        payloads.append(buf[offset : offset + length])
        offset += length
    return payloads
