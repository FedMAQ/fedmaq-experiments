"""Quantization-based baseline compression hooks (FedPAQ, DAdaQuant)."""

from collections.abc import Callable

import numpy as np

from fedmaq.baselines.dadaquant_coder import dadaquant_pack
from fedmaq.baselines.transport import UploadReport
from fedmaq.core.client import CompressionHook


def _stochastic_round(scaled: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unbiased stochastic (dithered) rounding: floor(x) w.p. 1-frac(x), else ceil(x).

    ``E[result] = scaled`` for any input, unlike ``np.round`` which is
    deterministic and biased (#24). Lifted from DAdaQuant's original inline
    implementation into shared code so FedPAQ and FedMAQ's quantizer share it.
    """
    floor_val = np.floor(scaled)
    prob = scaled - floor_val
    rand_val = rng.random(scaled.shape)
    return np.where(rand_val < prob, floor_val + 1, floor_val)


def _require_rng(rng: np.random.Generator | None, hook_name: str) -> np.random.Generator:
    """Raise loudly instead of silently defaulting to an unseeded generator (#24)."""
    if rng is None:
        raise ValueError(f"{hook_name} requires a seeded rng for stochastic rounding (got None).")
    return rng


def _codes_to_float(codes: np.ndarray, scale: float, levels: int) -> np.ndarray:
    """Map integer-valued codes in [-levels, levels] back to float via ``scale``."""
    return (codes / levels) * scale


def symmetric_levels(q: int) -> int:
    """Return the positive levels for a symmetric ``q``-bit quantizer."""
    return max(1, (1 << (q - 1)) - 1)


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
    scale_fn: Callable[[np.ndarray], float],
    quantize_elem: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    on_codes: Callable[[np.ndarray, float], None] | None = None,
) -> tuple[list[np.ndarray], list[bytes]]:
    """Shared quantize-and-account skeleton for uniform quantization hooks.

    Iterates ``deltas``, skipping empty tensors (free: nothing is transmitted)
    and otherwise computing ``scale = scale_fn(d)`` before applying
    ``quantize_elem(d, scale)``, which returns ``(dequantized, codes)``.
    ``scale_fn`` is a per-call-site choice (l2 vs l∞ normalization, #24) rather
    than hardcoded here. All-zero tensors (scale 0) skip quantization but still
    produce an all-zero payload for the shared upload report.

    ``on_codes``, if given, is called with each non-empty tensor's
    ``(codes, scale)`` right after they're determined -- both branches above,
    matching the primary axis's coverage exactly. This is the extension point
    for a hook-specific *secondary* byte axis (#26: DAdaQuant's as-published
    coder) without folding that axis into this shared, arm-agnostic skeleton
    or duplicating its loop.

    Returns ``(quantized_deltas, payloads)``. The payloads are the actual
    per-tensor byte strings, one per measurement boundary, so a future encoder
    change can be re-scored against the exact original payloads offline.
    """
    quantized_deltas: list[np.ndarray] = []
    payloads: list[bytes] = []

    for d in deltas:
        if d.size == 0:
            quantized_deltas.append(d)
            continue

        scale = scale_fn(d)
        if scale > 0.0:
            dequantized, codes = quantize_elem(d, scale)
            quantized_deltas.append(dequantized.astype(np.float32))
        else:
            codes = np.zeros_like(d, dtype=np.int64)
            quantized_deltas.append(d)

        if on_codes is not None:
            on_codes(codes, scale)

        payload = _serialize_codes(codes, scale)
        payloads.append(payload)

    return quantized_deltas, payloads


class FedPAQCompressionHook(CompressionHook):
    """Uniform symmetric quantization hook implementing FedPAQ.

    Unbiased stochastic rounding, l2-normalized (#24, matching
    ``chapter_3.tex``'s ``Q_s(v_j) = ‖v‖₂ · sgn(v_j) · ξ_j(v,s)``) for the
    ``q>1`` path. The ``q<=1`` sign-quantization path has no rounding step to
    make unbiased and keeps l∞ (``max|d|``) unconditionally: l2 would inflate
    every coordinate by ``~√d`` there, not damp it (see ``_scale``).
    """

    def __init__(self, q: int = 8, rng: np.random.Generator | None = None) -> None:
        """Initialize the compression hook with quantization bit-width.

        Parameters
        ----------
        q : int
            Number of quantization *bits* (default: 8).
            Each element is represented with ``q`` bits, giving
            ``2^(q-1) - 1`` positive quantization levels.
        rng : np.random.Generator | None
            Seeded NumPy generator for the ``q>1`` stochastic-rounding path.
            Required whenever that path is reached; ``q<=1`` never consumes it.
        """
        self.q = q
        self.rng = rng

    @property
    def levels(self) -> int:
        """Number of positive quantization levels for symmetric bounds (e.g. 127 for 8-bit).

        Floored at 1 so the ``q=1`` case (which would otherwise give 0 positive
        levels and a 0/0 NaN on dequantization) is well-defined; ``compress``
        additionally special-cases ``q<=1`` as pure sign quantization.
        """
        return symmetric_levels(self.q)

    def _scale(self, d: np.ndarray) -> float:
        if self.q <= 1:
            return float(np.max(np.abs(d)))
        return float(np.linalg.norm(d))

    def _quantize_elem(self, d: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        if self.q <= 1:
            # 1-bit sign quantization: each element -> sign(d)*scale, i.e. values
            # in {-scale, +scale} (exact zeros stay 0). Avoids the 0/0 NaN that a
            # 0-positive-level uniform quantizer would give.
            codes = np.sign(d).astype(np.int64)
            return codes.astype(np.float32) * scale, codes
        rng = _require_rng(self.rng, "FedPAQCompressionHook")
        # Normalize to [-1, 1], map to [-levels, levels], stochastic-round, map back.
        scaled = (d / scale) * self.levels
        codes = _stochastic_round(scaled, rng)
        return _codes_to_float(codes, scale, self.levels), codes.astype(np.int64)

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], UploadReport]:
        """Quantize ``deltas`` and return the complete upload-byte report."""
        quantized_deltas, payloads = _quantize_deltas(deltas, self._scale, self._quantize_elem)
        return quantized_deltas, UploadReport.from_payloads(payloads)


class DAdaQuantCompressionHook(CompressionHook):
    """Doubly-adaptive quantization hook implementing DAdaQuant's client-side quantizer.

    Each update tensor is normalized by its Euclidean norm, matching the
    source method's quantizer. This differs from FedMAQ's error-feedback path,
    which deliberately retains l-infinity normalization.

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
            Seeded NumPy generator for reproducible stochastic rounding.
            Required at compress()-time; no unseeded fallback (#24) -- an arm
            that omits it fails loudly instead of silently defaulting.
        """
        self.q = q
        self.rng = rng

    def _scale(self, d: np.ndarray) -> float:
        # Kept explicit at this call site rather than inherited from FedPAQ's
        # q-dependent scale switch: DAdaQuant specifies l2 normalization for
        # its own adaptive quantizer at every q.
        return float(np.linalg.norm(d))

    def _quantize_elem(self, d: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        rng = _require_rng(self.rng, "DAdaQuantCompressionHook")
        # Normalize to [-1, 1], scale to [-q, q], stochastic-round, map back.
        scaled = (d / scale) * self.q
        codes = _stochastic_round(scaled, rng)
        return (codes / self.q) * scale, codes.astype(np.int64)

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], UploadReport]:
        """Quantize ``deltas`` and include DAdaQuant's as-published byte axis."""
        secondary_total = 0

        def _accumulate_secondary(codes: np.ndarray, scale: float) -> None:
            nonlocal secondary_total
            # +4: the float32 scale travels alongside the coded payload as raw
            # bytes, same convention as every other baseline's analytic
            # formula (see ``_serialize_codes``) -- DAdaQuant's paper codes
            # quantization levels, not a single per-tensor normalization float.
            secondary_total += len(dadaquant_pack(codes)) + 4

        quantized_deltas, payloads = _quantize_deltas(
            deltas, self._scale, self._quantize_elem, on_codes=_accumulate_secondary
        )
        return quantized_deltas, UploadReport.from_payloads(
            payloads, secondary_bytes=secondary_total
        )
