"""Quantization-based baseline compression hooks (FedPAQ, DAdaQuant)."""

import math

import numpy as np

from fedmaq.core.client import CompressionHook


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
        """Number of positive quantization levels for symmetric bounds (e.g. 127 for 8-bit)."""
        return (1 << (self.q - 1)) - 1

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
        quantized_deltas = []
        total_bytes = 0

        for d in deltas:
            if d.size == 0:
                quantized_deltas.append(d)
                continue

            # Absolute maximum for scale
            scale = float(np.max(np.abs(d)))

            if scale > 0.0:
                # Normalize to [-1, 1]
                normalized = d / scale
                # Map to [-levels, levels] and round to nearest integer
                quantized = np.round(normalized * self.levels)
                # Map back to float32 domain
                dequantized = (quantized / self.levels) * scale
                quantized_deltas.append(dequantized.astype(np.float32))

                # Estimate size: q bits per element + 4 bytes (float32) for scale metadata
                element_bits = d.size * self.q
                total_bytes += int(math.ceil(element_bits / 8.0)) + 4
            else:
                quantized_deltas.append(d)
                total_bytes += 4  # scale = 0.0

        return quantized_deltas, total_bytes


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
        quantized_deltas = []
        total_bytes = 0

        # Bits needed to represent 2q+1 levels (e.g. q=8 → log2(17) ≈ 4.09 → 5 bits)
        bits_per_element = math.ceil(math.log2(max(2, 2 * self.q + 1)))

        for d in deltas:
            if d.size == 0:
                quantized_deltas.append(d)
                continue

            scale = float(np.max(np.abs(d)))

            if scale > 0.0:
                # Normalize to [-1, 1]
                normalized = d / scale
                # Scale to [-q, q]
                scaled = normalized * self.q
                # Stochastic rounding
                floor_val = np.floor(scaled)
                prob = scaled - floor_val
                rand_val = self.rng.random(scaled.shape)
                quantized = np.where(rand_val < prob, floor_val + 1, floor_val)
                # Map back to float32 domain
                dequantized = (quantized / self.q) * scale
                quantized_deltas.append(dequantized.astype(np.float32))

                # Estimate size: bits_per_element bits per element + 4 bytes scale metadata
                element_bits = d.size * bits_per_element
                total_bytes += int(math.ceil(element_bits / 8.0)) + 4
            else:
                quantized_deltas.append(d)
                total_bytes += 4  # scale = 0.0

        return quantized_deltas, total_bytes
