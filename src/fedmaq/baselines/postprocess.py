"""Post-processing pipeline for FedMAQ's winning formulation (manuscript §4.3).

Chains error-feedback (residual carried across rounds), diff-coding against the
prior round's quantized codes, and lossless zlib encoding on top of the existing
FedPAQ-style symmetric quantizer. Applies only to the primary CIFAR-10/100 +
FEMNIST benchmarking grid, not the Ablation Study — gated per-algorithm-yaml via
``post_process`` and dispatched in :func:`fedmaq.baselines.get_compressor_hook`.
"""

from __future__ import annotations

import logging

import numpy as np
from flwr.app import ArrayRecord, RecordDict

from fedmaq.baselines.quantization import (
    _codes_to_float,
    _require_rng,
    _serialize_codes,
    _stochastic_round,
    symmetric_levels,
)
from fedmaq.baselines.transport import UploadReport
from fedmaq.core.client import CompressionHook

logger = logging.getLogger(__name__)

_RESIDUAL_KEY = "fedmaq_postprocess_residual"
_PREV_CODES_KEY = "fedmaq_postprocess_prev_codes"


class FedMAQPostProcessCompressionHook(CompressionHook):
    """FedPAQ-style quantization plus error-feedback, diff-coding, and zlib encoding.

    State (residual and previous-round codes, one ``ArrayRecord`` each, ordered
    the same as the ``deltas`` list) is read/written via ``self._state``, a
    :class:`flwr.app.RecordDict` scoped to a single client across simulated
    rounds (``Context.state``).
    """

    def __init__(
        self,
        q: int = 8,
        state: RecordDict | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the hook.

        Parameters
        ----------
        q : int
            Number of quantization bits (same semantics as :class:`FedPAQCompressionHook`).
        state : flwr.app.RecordDict | None
            Per-client persistent state. If None, a fresh (per-call-scoped,
            non-persisted) RecordDict is used.
        rng : np.random.Generator | None
            Seeded NumPy generator for the ``q>1`` stochastic-rounding path.
            Required whenever that path is reached; ``q<=1`` never consumes it.
        """
        self.q = q
        self._state = state if state is not None else RecordDict()
        self._logged_shape_mismatch = False
        self.rng = rng

    @property
    def levels(self) -> int:
        return symmetric_levels(self.q)

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], UploadReport]:
        """Apply error feedback and diff coding, then return the upload report."""
        residual_record = self._state.get(_RESIDUAL_KEY)
        residuals = residual_record.to_numpy_ndarrays() if residual_record is not None else None
        prev_codes_record = self._state.get(_PREV_CODES_KEY)
        prev_codes_list = (
            prev_codes_record.to_numpy_ndarrays() if prev_codes_record is not None else None
        )

        out_deltas: list[np.ndarray] = []
        new_residuals: list[np.ndarray] = []
        new_codes: list[np.ndarray] = []
        payloads: list[bytes] = []

        for i, d in enumerate(deltas):
            if d.size == 0:
                out_deltas.append(d)
                new_residuals.append(d.astype(np.float32))
                new_codes.append(d.astype(np.int64))
                continue

            residual = (
                residuals[i]
                if residuals is not None and i < len(residuals) and residuals[i].shape == d.shape
                else np.zeros_like(d, dtype=np.float32)
            )
            d_fb = d + residual

            # Deliberately l∞ (max|d_fb|), NOT the l2 switch FedPAQ/FedMAQ's
            # memoryless quantizer takes (#24): under error feedback, l2 scale
            # grows with tensor dimension while the per-coordinate quantization
            # step (‖d_fb‖₂/levels) stays fixed at a small `levels`, breaking the
            # contraction property error feedback needs -- verified empirically
            # (residual/delta ratio diverges >1e6x over 40 rounds on a
            # 2M-parameter tensor at FedMAQ's actual q range). l∞ keeps this
            # path's quantization step bounded by the tensor's own peak
            # magnitude regardless of dimension, so it stays stable. Extending
            # l2 here needs its own ticket with a redesigned feedback scheme.
            scale = float(np.max(np.abs(d_fb)))
            if scale == 0.0:
                zero_codes = np.zeros_like(d_fb, dtype=np.int64)
                out_deltas.append(d_fb.astype(np.float32))
                new_residuals.append(np.zeros_like(d_fb, dtype=np.float32))
                new_codes.append(zero_codes)
                zero_payload = _serialize_codes(zero_codes, scale)
                payloads.append(zero_payload)
                continue

            if self.q <= 1:
                codes = np.sign(d_fb).astype(np.int64)
                dequantized = (codes.astype(np.float32)) * scale
            else:
                rng = _require_rng(self.rng, "FedMAQPostProcessCompressionHook")
                scaled = (d_fb / scale) * self.levels
                codes_f = _stochastic_round(scaled, rng)
                codes = codes_f.astype(np.int64)
                dequantized = _codes_to_float(codes_f, scale, self.levels).astype(np.float32)

            out_deltas.append(dequantized)
            new_residuals.append((d_fb - dequantized).astype(np.float32))
            new_codes.append(codes)

            prev_codes = (
                prev_codes_list[i]
                if prev_codes_list is not None and i < len(prev_codes_list)
                else None
            )
            if prev_codes is not None and prev_codes.shape == codes.shape:
                diffed = codes - prev_codes
            else:
                if prev_codes is not None and not self._logged_shape_mismatch:
                    logger.warning(
                        "FedMAQPostProcessCompressionHook: prev_codes shape %s != "
                        "codes shape %s for tensor %d; falling back to raw codes "
                        "for this round.",
                        prev_codes.shape,
                        codes.shape,
                        i,
                    )
                    self._logged_shape_mismatch = True
                diffed = codes

            payload = _serialize_codes(diffed, scale)
            payloads.append(payload)

        self._state[_RESIDUAL_KEY] = ArrayRecord(numpy_ndarrays=new_residuals)
        self._state[_PREV_CODES_KEY] = ArrayRecord(numpy_ndarrays=new_codes)

        return out_deltas, UploadReport.from_payloads(payloads)
