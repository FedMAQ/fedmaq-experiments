"""Quantization-based baseline compression hooks (FedPAQ, DAdaQuant)."""

from collections.abc import Callable

import numpy as np

from fedmaq.baselines.transport import measure_bytes
from fedmaq.core.client import CompressionHook


def _normalize_and_round(d: np.ndarray, scale: float, levels: int) -> np.ndarray:
    """Normalize ``d`` by ``scale`` and round to integer-valued codes in [-levels, levels]."""
    normalized = d / scale
    return np.round(normalized * levels)


def _codes_to_float(codes: np.ndarray, scale: float, levels: int) -> np.ndarray:
    """Map integer-valued codes in [-levels, levels] back to float via ``scale``."""
    return (codes / levels) * scale


def _serialize_codes(codes: np.ndarray, scale: float) -> bytes:
    """Pack integer codes plus their float32 scale into one transmittable payload.

    The scale travels *inside* the payload handed to ``measure_bytes`` rather
    than as a separate additive convention (see ``transport.py``).

    Codes are stored at a fixed int64 width regardless of the quantizer's
    nominal bit-width — a deliberate deferral (#25). Narrowing to the nominal
    width (e.g. int8 for q<=8) would itself lower every quantized arm's total,
    confounding *this* seam's accounting change with a wire-format change; the
    fixed width instead means the same "wasted precision" is charged uniformly
    to every quantized arm, so a comparison between them still isolates the
    encoder's real entropy-coding effect.

    Consequence, verified empirically, not merely assumed: at low bit-widths
    (few, highly-skewed code values) zlib recovers well below the int64 padding
    and measured bytes drop sharply below the old analytic formula. At high
    bit-widths approaching the tensor's real information content (FedPAQ's
    configured q=8 among them) zlib cannot fully reclaim the 8x padding, and
    measured bytes can come out *above* the old formula instead. Both are
    legitimate outcomes of the same held-constant transport — the seam's job is
    uniform accounting, not a guaranteed drop — but which one shows up for a
    given arm/config is data-dependent and must be checked per #25 AC 3, not
    assumed from this docstring or from synthetic test data.
    """
    return codes.astype(np.int64).tobytes() + np.float32(scale).tobytes()


def _quantize_deltas(
    deltas: list[np.ndarray],
    quantize_elem: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[list[np.ndarray], int, int]:
    """Shared quantize-and-account skeleton for uniform quantization hooks.

    Iterates ``deltas``, skipping empty tensors (free: nothing is transmitted)
    and otherwise applies ``quantize_elem(d, scale)`` where ``scale = max|d|``,
    which returns ``(dequantized, codes)``. All-zero tensors (scale 0) skip
    quantization but still route their (all-zero) codes through the same
    measured transport, so no per-arm byte arithmetic survives outside it.

    Returns ``(quantized_deltas, measured_bytes, payload_bytes)`` — the last
    is the pre-encoding payload size (codes + scale, before ``measure_bytes``),
    logged so a future encoder change can be re-scored offline without
    re-running training (see ``TelemetryManager.record_fit_round``).
    """
    quantized_deltas: list[np.ndarray] = []
    total_bytes = 0
    total_payload_bytes = 0

    for d in deltas:
        if d.size == 0:
            quantized_deltas.append(d)
            continue

        scale = float(np.max(np.abs(d)))
        if scale > 0.0:
            dequantized, codes = quantize_elem(d, scale)
            quantized_deltas.append(dequantized.astype(np.float32))
        else:
            codes = np.zeros_like(d, dtype=np.int64)
            quantized_deltas.append(d)

        payload = _serialize_codes(codes, scale)
        total_bytes += measure_bytes(payload)
        total_payload_bytes += len(payload)

    return quantized_deltas, total_bytes, total_payload_bytes


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

    def _quantize_elem(self, d: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        if self.q <= 1:
            # 1-bit sign quantization: each element -> sign(d)*scale, i.e. values
            # in {-scale, +scale} (exact zeros stay 0). Avoids the 0/0 NaN that a
            # 0-positive-level uniform quantizer would give.
            codes = np.sign(d).astype(np.int64)
            return codes.astype(np.float32) * scale, codes
        # Normalize to [-1, 1], map to [-levels, levels], round, map back.
        codes = _normalize_and_round(d, scale, self.levels)
        return _codes_to_float(codes, scale, self.levels), codes.astype(np.int64)

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas using symmetric uniform quantization.

        Parameters
        ----------
        deltas : list[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        tuple[list[np.ndarray], int]
            Quantized deltas and the measured size in bytes. The pre-encoding
            payload size is stashed on ``self.last_payload_bytes``.
        """
        quantized_deltas, total_bytes, payload_bytes = _quantize_deltas(
            deltas, self._quantize_elem
        )
        self.last_payload_bytes = payload_bytes
        return quantized_deltas, total_bytes


class DAdaQuantCompressionHook(CompressionHook):
    """Doubly-adaptive quantization hook implementing DAdaQuant's client-side quantizer.

    .. note::
        The attribute ``q`` represents the number of quantization *levels per sign*
        (symmetric around zero), NOT a bit-width. The total number of discrete levels
        is ``2*q + 1`` (integers in [-q, q]), which differs from FedPAQ where ``q``
        is a true bit-width.

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

    def _quantize_elem(self, d: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        # Normalize to [-1, 1], scale to [-q, q], stochastic-round, map back.
        scaled = (d / scale) * self.q
        floor_val = np.floor(scaled)
        prob = scaled - floor_val
        rand_val = self.rng.random(scaled.shape)
        codes = np.where(rand_val < prob, floor_val + 1, floor_val)
        return (codes / self.q) * scale, codes.astype(np.int64)

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas using stochastic uniform quantization with ``self.q`` bins per sign.

        Parameters
        ----------
        deltas : list[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        tuple[list[np.ndarray], int]
            Quantized deltas and the measured size in bytes. The pre-encoding
            payload size is stashed on ``self.last_payload_bytes``.
        """
        quantized_deltas, total_bytes, payload_bytes = _quantize_deltas(
            deltas, self._quantize_elem
        )
        self.last_payload_bytes = payload_bytes
        return quantized_deltas, total_bytes
