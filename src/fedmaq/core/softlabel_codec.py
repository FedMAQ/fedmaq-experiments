"""Constrained soft-label quantization + delta/zlib codec for CFD (Sattler et al. 2022).

Shared by ``core/strategy_hooks/cfd.py`` (server->client downstream labels) and
``core/client_hooks/cfd.py`` (client->server upstream labels). Mirrors the
error-feedback-free half of ``baselines/postprocess.py``'s delta+zlib pattern,
applied to soft-label probability vectors instead of weight deltas.

Lives under ``core/`` rather than ``baselines/`` (unlike ``postprocess.py``)
because ``client_hooks/cfd.py`` needs it, and ``baselines/__init__.py`` imports
``fedmaq.core.client.CompressionHook`` at module scope -- importing this codec
from ``baselines`` would pull that whole package in while ``core.client`` is
still mid-import (via ``core.client`` -> ``core.client_hooks`` -> this module's
importer), a genuine circular import. This module has no ``fedmaq`` dependencies
of its own, so placing it directly under ``core/`` sidesteps the cycle.
"""

from __future__ import annotations

import zlib

import numpy as np


def constrained_quantize(probs: np.ndarray, b: int) -> np.ndarray:
    """b-bit uniform quantization of each row of ``probs`` onto the probability
    simplex, exactly preserving ``codes.sum(axis=1) == 2**b - 1`` (CFD paper eq. 10).

    Uses the largest-remainder (Hamilton apportionment) method: floor each
    scaled probability to an integer code, then distribute the remaining
    quantization budget to the components with the largest fractional
    remainder. Unlike independent per-component rounding, this keeps the row
    sum exact while minimizing L1 rounding error. For ``b == 1`` this reduces
    exactly to argmax one-hot (eq. 11): the only component with nonzero
    "remainder budget" (1 unit) is the one with the largest original
    probability.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``[N, K]``, each row a probability distribution over ``K`` classes.
    b : int
        Bit-width (>= 1). ``s = 2**b - 1`` quantization levels per component.

    Returns
    -------
    np.ndarray
        Integer codes, shape ``[N, K]``, dtype int64, each row summing to ``s``.
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    probs_arr = np.asarray(probs, dtype=np.float64)
    if probs_arr.ndim != 2:
        raise ValueError(f"probs must be 2D [N, K], got shape {probs_arr.shape}")

    s = (1 << b) - 1
    k = probs_arr.shape[1]
    scaled = probs_arr * s
    floor_codes = np.floor(scaled).astype(np.int64)
    remainder = scaled - floor_codes
    budget = np.clip(s - floor_codes.sum(axis=1), 0, k)

    # Rank each row's components by descending fractional remainder (stable
    # tie-break by original index); bump the top `budget[i]` components by one.
    order = np.argsort(-remainder, axis=1, kind="stable")
    rank = np.argsort(order, axis=1)
    bump = rank < budget[:, None]

    return floor_codes + bump.astype(np.int64)


def dequantize(codes: np.ndarray, b: int) -> np.ndarray:
    """Map integer codes back to probability vectors (``codes / (2**b - 1)``)."""
    s = (1 << b) - 1
    return (codes.astype(np.float32)) / float(s)


def encode_bytes(
    codes: np.ndarray, prev_codes: np.ndarray | None, delta: bool = True
) -> tuple[int, np.ndarray]:
    """Delta-code ``codes`` against ``prev_codes`` and zlib-compress for a real
    byte count. Mirrors ``postprocess.py``'s delta+zlib+shape-mismatch-fallback
    pattern (lines 124-146): falls back to raw (undiffed) codes when
    ``prev_codes`` is absent or shape-mismatched.

    Returns
    -------
    tuple[int, np.ndarray]
        ``(compressed_byte_count, codes)`` -- the raw (undiffed) codes are
        returned so the caller can stash them as next round's delta reference.
    """
    if delta and prev_codes is not None and prev_codes.shape == codes.shape:
        diffed = codes - prev_codes
    else:
        diffed = codes
    compressed = zlib.compress(diffed.astype(np.int64).tobytes())
    return len(compressed), codes.astype(np.int64)


def codes_to_bytes(codes: np.ndarray) -> bytes:
    """Serialize integer codes for the Flower ``FitIns.config`` bytes channel."""
    return codes.astype(np.int64).tobytes()


def codes_from_bytes(buf: bytes, num_classes: int) -> np.ndarray:
    """Deserialize a ``[N, num_classes]`` code matrix, failing loud on drift."""
    arr = np.frombuffer(buf, dtype=np.int64)
    if arr.size % num_classes != 0:
        raise ValueError(
            f"CFD code buffer has {arr.size} ints, not divisible by num_classes={num_classes}."
        )
    return arr.reshape(-1, num_classes).copy()
