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

import zlib


def measure_bytes(payload: bytes) -> int:
    """Return the transmitted size of ``payload`` under the held-constant transport.

    ``payload`` must already include any metadata (e.g. a quantization scale)
    that would travel over the wire alongside the data — this function does not
    add anything on top of the compressed result.
    """
    return len(zlib.compress(payload))
