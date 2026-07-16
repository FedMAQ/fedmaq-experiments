"""SVD-based dynamic compression hook for FedKD."""

import numpy as np

from fedmaq.core.client import CompressionHook

# Explicit Union type for compress_tensor return value.
# Either a raw 1-tuple (uncompressed pass-through) or a 3-tuple of SVD factors.
CompressedTensor = tuple[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]


def compress_tensor(
    tensor_np: np.ndarray, energy: float, min_rank_frac: float = 0.0
) -> CompressedTensor:
    """Compress a tensor using SVD if its dimension is >= 2.

    ``min_rank_frac`` floors the retained rank at ``min_rank_frac * full_rank``
    (rounded up), regardless of the energy-derived threshold. Without a floor,
    the energy->rank mapping is non-monotonic on concentrated spectra (e.g.
    depthwise-separable conv deltas): as the round-scheduled energy target
    rises through the spectrum's dominant singular values, the threshold can
    still land near rank 1 (see F10, docs/audits/distillation-direction-audit.md).

    Returns either ``(U, Sigma, V)`` (compressed) or ``(tensor_np,)``
    (pass-through for 1-D tensors or SVD failures).
    """
    orig_shape = tensor_np.shape
    if len(orig_shape) < 2:
        return (tensor_np,)

    # Reshape to 2D
    if len(orig_shape) > 2:
        mat = tensor_np.reshape(orig_shape[0], -1)
    else:
        mat = tensor_np

    # SVD
    try:
        u, sigma, v = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback if SVD fails to converge
        return (tensor_np,)

    # Determine threshold based on energy
    sigma_sq = np.square(sigma)
    total_energy = np.sum(sigma_sq)
    if total_energy == 0.0:
        return (tensor_np,)

    cumsum = np.cumsum(sigma_sq)
    threshold = np.searchsorted(cumsum, energy * total_energy) + 1
    if min_rank_frac > 0.0:
        threshold = max(threshold, int(np.ceil(min_rank_frac * len(sigma))))
    threshold = min(max(1, threshold), len(sigma))

    u_trunc = u[:, :threshold]
    sigma_trunc = sigma[:threshold]
    v_trunc = v[:threshold, :]

    return u_trunc, sigma_trunc, v_trunc


def decompress_tensor(compressed: CompressedTensor, orig_shape: tuple[int, ...]) -> np.ndarray:
    """Decompress a compressed tensor representation back to its original shape."""
    if len(compressed) == 1:
        return compressed[0]

    u, sigma, v = compressed
    reconstructed_2d = (u * sigma) @ v
    return reconstructed_2d.reshape(orig_shape)


def svd_compressed_nbytes(compressed: CompressedTensor, fallback_nbytes: int) -> int:
    """Transmitted byte size of an SVD-compressed tensor.

    A 3-tuple ``(U, Sigma, V)`` costs ``(U.size + Sigma.size + V.size) * 4`` bytes
    (float32 factors); a pass-through 1-tuple costs ``fallback_nbytes`` (the raw
    tensor). Shared by the FedKD upload accounting (:class:`FedKDCompressionHook`)
    and the server-side download-size telemetry.
    """
    if len(compressed) == 3:
        u, sigma, v = compressed
        return (u.size + sigma.size + v.size) * 4
    return fallback_nbytes


class FedKDCompressionHook(CompressionHook):
    """SVD-based dynamic compression hook implementing FedKD."""

    def __init__(self, energy: float = 0.5, min_rank_frac: float = 0.0) -> None:
        """Initialize the compression hook with the energy threshold fraction.

        Parameters
        ----------
        energy : float
            Energy ratio to retain in SVD singular values (0.0 to 1.0).
        min_rank_frac : float
            Floor on retained rank as a fraction of full rank (see
            :func:`compress_tensor`). 0.0 disables the floor.
        """
        self.energy = energy
        self.min_rank_frac = min_rank_frac

    def compress(self, deltas: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Compress deltas using SVD.

        Parameters
        ----------
        deltas : list[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        tuple[list[np.ndarray], int]
            Reconstructed deltas and the estimated size in bytes.
        """
        reconstructed_deltas = []
        total_bytes = 0

        for d in deltas:
            if d.size == 0:
                reconstructed_deltas.append(d)
                continue

            orig_shape = d.shape
            compressed = compress_tensor(d, self.energy, self.min_rank_frac)
            total_bytes += svd_compressed_nbytes(compressed, d.nbytes)

            if len(compressed) == 3:
                # Reconstruct/decompress locally to return in reconstructed_params
                decompressed = decompress_tensor(compressed, orig_shape)
                reconstructed_deltas.append(decompressed.astype(np.float32))
            else:
                # Uncompressed pass-through
                reconstructed_deltas.append(d)

        return reconstructed_deltas, total_bytes
