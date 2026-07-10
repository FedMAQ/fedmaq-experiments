"""Post-processing pipeline for FedMAQ's winning formulation (manuscript §4.3).

Chains error-feedback (residual carried across rounds), diff-coding against the
prior round's quantized codes, and lossless zlib encoding on top of the existing
FedPAQ-style symmetric quantizer. Applies only to the primary CIFAR-10/100 +
FEMNIST benchmarking grid, not the Ablation Study — gated per-algorithm-yaml via
``post_process`` and dispatched in :func:`fedmaq.baselines.get_compressor_hook`.
"""

from __future__ import annotations

import logging
import zlib

import numpy as np
from flwr.app import ArrayRecord, RecordDict

from fedmaq.baselines.quantization import _codes_to_float, _normalize_and_round
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

    def __init__(self, q: int = 8, state: RecordDict | None = None) -> None:
        """Initialize the hook.

        Parameters
        ----------
        q : int
            Number of quantization bits (same semantics as :class:`FedPAQCompressionHook`).
        state : flwr.app.RecordDict | None
            Per-client persistent state. If None, a fresh (per-call-scoped,
            non-persisted) RecordDict is used.
        """
        self.q = q
        self._state = state if state is not None else RecordDict()
        self._logged_shape_mismatch = False

    @property
    def levels(self) -> int:
        """Number of positive quantization levels for symmetric bounds (see FedPAQ)."""
        return max(1, (1 << (self.q - 1)) - 1)

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas with error-feedback + diff-coding + zlib.

        Parameters
        ----------
        deltas : list[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        tuple[list[np.ndarray], int]
            Dequantized deltas (same contract as other :class:`CompressionHook`
            implementations) and the real, zlib-measured byte size.
        """
        residual_record = self._state.get(_RESIDUAL_KEY)
        residuals = (
            residual_record.to_numpy_ndarrays() if residual_record is not None else None
        )
        prev_codes_record = self._state.get(_PREV_CODES_KEY)
        prev_codes_list = (
            prev_codes_record.to_numpy_ndarrays()
            if prev_codes_record is not None
            else None
        )

        out_deltas: list[np.ndarray] = []
        new_residuals: list[np.ndarray] = []
        new_codes: list[np.ndarray] = []
        total_bytes = 0

        for i, d in enumerate(deltas):
            if d.size == 0:
                out_deltas.append(d)
                new_residuals.append(d.astype(np.float32))
                new_codes.append(d.astype(np.int64))
                continue

            residual = (
                residuals[i]
                if residuals is not None
                and i < len(residuals)
                and residuals[i].shape == d.shape
                else np.zeros_like(d, dtype=np.float32)
            )
            d_fb = d + residual

            scale = float(np.max(np.abs(d_fb)))
            if scale == 0.0:
                out_deltas.append(d_fb.astype(np.float32))
                new_residuals.append(np.zeros_like(d_fb, dtype=np.float32))
                new_codes.append(np.zeros_like(d_fb, dtype=np.int64))
                total_bytes += 4  # scale = 0.0, matches _quantize_deltas convention
                continue

            if self.q <= 1:
                codes = np.sign(d_fb).astype(np.int64)
                dequantized = (codes.astype(np.float32)) * scale
            else:
                codes_f = _normalize_and_round(d_fb, scale, self.levels)
                codes = codes_f.astype(np.int64)
                dequantized = _codes_to_float(codes_f, scale, self.levels).astype(
                    np.float32
                )

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

            payload = diffed.tobytes()
            compressed = zlib.compress(payload)
            total_bytes += len(compressed) + 4  # +4: float32 scale (see _quantize_deltas)

        self._state[_RESIDUAL_KEY] = ArrayRecord(numpy_ndarrays=new_residuals)
        self._state[_PREV_CODES_KEY] = ArrayRecord(numpy_ndarrays=new_codes)

        return out_deltas, total_bytes
