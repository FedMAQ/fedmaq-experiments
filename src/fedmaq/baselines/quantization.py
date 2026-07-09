"""Quantization-based baseline compression hooks (FedPAQ, DAdaQuant)."""

import math
from collections.abc import Callable

import numpy as np

from fedmaq.core.client import CompressionHook


def _quantize_deltas(
    deltas: list[np.ndarray],
    quantize_elem: Callable[[np.ndarray, float], np.ndarray],
    bits_per_element: int,
) -> tuple[list[np.ndarray], int]:
    """Shared quantize-and-account skeleton for uniform quantization hooks.

    Iterates ``deltas``, skipping empty tensors and all-zero tensors (scale 0)
    as pass-throughs, and otherwise applies ``quantize_elem(d, scale)`` where
    ``scale = max|d|``. Byte size is ``ceil(size * bits_per_element / 8) + 4``
    per non-trivial tensor (the trailing 4 bytes carry the float32 scale).
    """
    quantized_deltas: list[np.ndarray] = []
    total_bytes = 0

    for d in deltas:
        if d.size == 0:
            quantized_deltas.append(d)
            continue

        scale = float(np.max(np.abs(d)))
        if scale > 0.0:
            quantized_deltas.append(quantize_elem(d, scale).astype(np.float32))
            element_bits = d.size * bits_per_element
            total_bytes += int(math.ceil(element_bits / 8.0)) + 4
        else:
            quantized_deltas.append(d)
            total_bytes += 4  # scale = 0.0

    return quantized_deltas, total_bytes


class FedPAQCompressionHook(CompressionHook):
    """Uniform symmetric quantization hook implementing FedPAQ."""

    def __init__(self, q: int = 8) -> None:
        """Initialize the compression hook with quantization bit-width.

        Parameters
        ----------
        q : int
            Number of quantization *bits* (default: 8).
            Each element is represented with ``q`` bits, giving
            ``2^(q-1) - 1`` positive quantization levels.
        """
        self.q = q

    @property
    def levels(self) -> int:
        """Number of positive quantization levels for symmetric bounds (e.g. 127 for 8-bit).

        Floored at 1 so the ``q=1`` case (which would otherwise give 0 positive
        levels and a 0/0 NaN on dequantization) is well-defined; ``compress``
        additionally special-cases ``q<=1`` as pure sign quantization.
        """
        return max(1, (1 << (self.q - 1)) - 1)

    def _quantize_elem(self, d: np.ndarray, scale: float) -> np.ndarray:
        if self.q <= 1:
            # 1-bit sign quantization: each element -> sign(d)*scale, i.e. values
            # in {-scale, +scale} (exact zeros stay 0). Avoids the 0/0 NaN that a
            # 0-positive-level uniform quantizer would give.
            return np.sign(d) * scale
        # Normalize to [-1, 1], map to [-levels, levels], round, map back.
        normalized = d / scale
        quantized = np.round(normalized * self.levels)
        return (quantized / self.levels) * scale

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas using symmetric uniform quantization.

        Parameters
        ----------
        deltas : list[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        tuple[list[np.ndarray], int]
            Quantized deltas and the estimated size in bytes.
        """
        return _quantize_deltas(deltas, self._quantize_elem, self.q)


class DAdaQuantCompressionHook(CompressionHook):
    """Doubly-adaptive quantization hook implementing DAdaQuant's client-side quantizer.

    .. note::
        The attribute ``q`` represents the number of quantization *levels per sign*
        (symmetric around zero), NOT a bit-width. The total number of discrete levels
        is ``2*q + 1`` (integers in [-q, q]).  Byte-size is estimated as
        ``ceil(log2(2*q + 1))`` bits per element, which differs from FedPAQ where
        ``q`` is a true bit-width.

        This attribute is written at runtime by :class:`~fedmaq.core.client.GenericClient`
        via ``compressor_hook.q = int(config["q"])`` when the server sends an updated
        quantization level.
    """

    def __init__(self, q: int = 8, rng: np.random.Generator | None = None) -> None:
        """Initialize the compression hook.

        Parameters
        ----------
        q : int
            Number of quantization *levels per sign* (default: 8).
            The quantizer maps values to integers in [-q, q] (2q+1 total levels).
        rng : np.random.Generator | None
            Seeded NumPy random generator for reproducible stochastic rounding.
            If None, a default (unseeded) generator is used.
        """
        self.q = q
        self.rng = rng if rng is not None else np.random.default_rng()

    def _quantize_elem(self, d: np.ndarray, scale: float) -> np.ndarray:
        # Normalize to [-1, 1], scale to [-q, q], stochastic-round, map back.
        scaled = (d / scale) * self.q
        floor_val = np.floor(scaled)
        prob = scaled - floor_val
        rand_val = self.rng.random(scaled.shape)
        quantized = np.where(rand_val < prob, floor_val + 1, floor_val)
        return (quantized / self.q) * scale

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas using stochastic uniform quantization with ``self.q`` bins per sign.

        Parameters
        ----------
        deltas : list[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        tuple[list[np.ndarray], int]
            Quantized deltas and the estimated size in bytes.
        """
        # Bits needed to represent 2q+1 levels (e.g. q=8 -> log2(17) ~ 4.09 -> 5 bits)
        bits_per_element = math.ceil(math.log2(max(2, 2 * self.q + 1)))
        return _quantize_deltas(deltas, self._quantize_elem, bits_per_element)
