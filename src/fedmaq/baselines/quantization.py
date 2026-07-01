"""Quantization-based baseline compression hooks (FedPAQ, etc.)."""

import numpy as np

from fedmaq.core.client import CompressionHook


class FedPAQCompressionHook(CompressionHook):
    """Uniform symmetric quantization hook implementing FedPAQ."""

    def __init__(self, q: int = 8) -> None:
        """Initialize the compression hook with quantization bit-width.

        Parameters
        ----------
        q : int
            Number of quantization bits (default: 8).
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
        deltas : List[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        Tuple[List[np.ndarray], int]
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

                # Estimate size: q bits per element + 4 bytes (Float32) for scale metadata
                element_bits = d.size * self.q
                total_bytes += int(np.ceil(element_bits / 8.0)) + 4
            else:
                quantized_deltas.append(d)
                total_bytes += 4  # scale = 0.0

        return quantized_deltas, total_bytes


class DAdaQuantCompressionHook(CompressionHook):
    """Doubly-adaptive quantization hook implementing DAdaQuant's client-side quantizer."""

    def __init__(self, q: int = 8) -> None:
        """Initialize the compression hook.

        Parameters
        ----------
        q : int
            Quantization level (default: 8).
        """
        self.q = q

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas using stochastic uniform quantization with self.q bins per sign.

        Parameters
        ----------
        deltas : List[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        Tuple[List[np.ndarray], int]
            Quantized deltas and the estimated size in bytes.
        """
        quantized_deltas = []
        total_bytes = 0

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
                rand_val = np.random.rand(*scaled.shape)
                quantized = np.where(rand_val < prob, floor_val + 1, floor_val)
                # Map back to float32 domain
                dequantized = (quantized / self.q) * scale
                quantized_deltas.append(dequantized.astype(np.float32))

                # Estimate size: self.q bits per element + 4 bytes scale metadata
                element_bits = d.size * self.q
                total_bytes += int(np.ceil(element_bits / 8.0)) + 4
            else:
                quantized_deltas.append(d)
                total_bytes += 4  # scale = 0.0

        return quantized_deltas, total_bytes
