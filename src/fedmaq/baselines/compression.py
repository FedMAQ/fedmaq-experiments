"""SVD-based dynamic compression hook for FedKD."""

from typing import List, Tuple
import numpy as np

from fedmaq.core.client import CompressionHook


def compress_tensor(tensor_np: np.ndarray, energy: float) -> Tuple[np.ndarray, ...]:
    """Compresses a tensor using SVD if its dimension is >= 2.

    Returns either (U, Sigma, V) or (tensor_np,).
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
    threshold = min(max(1, threshold), len(sigma))

    u_trunc = u[:, :threshold]
    sigma_trunc = sigma[:threshold]
    v_trunc = v[:threshold, :]

    return u_trunc, sigma_trunc, v_trunc


def decompress_tensor(
    compressed: Tuple[np.ndarray, ...], orig_shape: Tuple[int, ...]
) -> np.ndarray:
    """Decompress a compressed tensor representation back to its original shape."""
    if len(compressed) == 1:
        return compressed[0]

    u, sigma, v = compressed
    reconstructed_2d = (u * sigma) @ v
    return reconstructed_2d.reshape(orig_shape)


class FedKDCompressionHook(CompressionHook):
    """SVD-based dynamic compression hook implementing FedKD."""

    def __init__(self, energy: float = 0.5) -> None:
        """Initialize the compression hook with the energy threshold fraction.

        Parameters
        ----------
        energy : float
            Energy ratio to retain in SVD singular values (0.0 to 1.0).
        """
        self.energy = energy

    def compress(self, deltas: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
        """Compress deltas using SVD.

        Parameters
        ----------
        deltas : List[np.ndarray]
            List of model weight updates (deltas).

        Returns
        -------
        Tuple[List[np.ndarray], int]
            Reconstructed deltas and the estimated size in bytes.
        """
        reconstructed_deltas = []
        total_bytes = 0

        for d in deltas:
            if d.size == 0:
                reconstructed_deltas.append(d)
                continue

            orig_shape = d.shape
            compressed = compress_tensor(d, self.energy)

            if len(compressed) == 3:
                # Compressed: U, Sigma, V
                u, sigma, v = compressed
                # Reconstruct/decompress locally to return in reconstructed_params
                decompressed = decompress_tensor(compressed, orig_shape)
                reconstructed_deltas.append(decompressed.astype(np.float32))

                # Size calculation: (u.size + sigma.size + v.size) * 4 bytes
                element_bytes = (u.size + sigma.size + v.size) * 4
                total_bytes += element_bytes
            else:
                # Uncompressed
                reconstructed_deltas.append(d)
                total_bytes += d.nbytes

        return reconstructed_deltas, total_bytes
