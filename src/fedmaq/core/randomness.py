"""Stable, purpose-separated random streams for one registered run."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(
    stream: str,
    run_seed: int,
    partition_id: int,
    server_round: int,
) -> int:
    """Derive a reproducible integer seed from a run identity and round.

    The stream name is part of the identity so training and compression cannot
    accidentally share a random sequence. Hashing the tuple avoids collisions
    caused by additive seed composition (for example ``seed + client_id``).
    """
    if not stream:
        raise ValueError("stream must not be empty")
    if server_round < 1:
        raise ValueError("server_round must be at least 1")
    material = f"{stream}\0{int(run_seed)}\0{int(partition_id)}\0{int(server_round)}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def derive_numpy_rng(
    stream: str,
    run_seed: int,
    partition_id: int,
    server_round: int,
) -> np.random.Generator:
    """Return the NumPy generator for one purpose-specific round stream."""
    return np.random.default_rng(derive_seed(stream, run_seed, partition_id, server_round))
