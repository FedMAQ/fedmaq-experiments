"""Versioned control-message schemas and accounting (#88).

Models and accounts for serialized control messages (client gradient-preflight
reports, server quantization assignments) with explicit directionality and multiplicity.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import StrEnum


class ControlDirection(StrEnum):
    """Network transmission direction of a control message."""

    UPLOAD = "upload"
    DOWNLOAD = "download"


CONTROL_SCHEMA_VERSION: int = 1
PREFLIGHT_MAGIC: bytes = b"FMPF"
PREFLIGHT_STRUCT: struct.Struct = struct.Struct("<4sBIfIf")
PREFLIGHT_BYTE_LENGTH: int = PREFLIGHT_STRUCT.size  # 21 bytes

ASSIGNMENT_MAGIC: bytes = b"FMQA"
ASSIGNMENT_STRUCT: struct.Struct = struct.Struct("<4sBIB")
ASSIGNMENT_BYTE_LENGTH: int = ASSIGNMENT_STRUCT.size  # 10 bytes


@dataclass(frozen=True)
class ClientPreflightMessage:
    """Client-to-server gradient-norm, dataset size, and capacity preflight report.

    Direction: UPLOAD (Client -> Server)
    Multiplicity: 1 per sampled client per adaptive round.
    """

    partition_id: int
    grad_norm: float
    dataset_size: int
    capacity_mb: float
    version: int = CONTROL_SCHEMA_VERSION
    direction: ControlDirection = ControlDirection.UPLOAD

    def __post_init__(self) -> None:
        if not math.isfinite(self.grad_norm) or self.grad_norm < 0:
            raise ValueError(f"grad_norm must be non-negative and finite, got {self.grad_norm}")
        if self.dataset_size < 0:
            raise ValueError(f"dataset_size must be non-negative, got {self.dataset_size}")
        if not math.isfinite(self.capacity_mb) or self.capacity_mb < 0:
            raise ValueError(f"capacity_mb must be non-negative and finite, got {self.capacity_mb}")

    def serialize(self) -> bytes:
        return PREFLIGHT_STRUCT.pack(
            PREFLIGHT_MAGIC,
            self.version,
            self.partition_id,
            float(self.grad_norm),
            self.dataset_size,
            float(self.capacity_mb),
        )

    @classmethod
    def deserialize(cls, buf: bytes) -> ClientPreflightMessage:
        if len(buf) != PREFLIGHT_BYTE_LENGTH:
            raise ValueError(
                f"Invalid preflight buffer length: expected {PREFLIGHT_BYTE_LENGTH} bytes, "
                f"got {len(buf)} bytes."
            )
        magic, version, partition_id, grad_norm, dataset_size, capacity_mb = (
            PREFLIGHT_STRUCT.unpack(buf)
        )
        if magic != PREFLIGHT_MAGIC:
            raise ValueError(
                f"Invalid preflight magic: expected {PREFLIGHT_MAGIC!r}, got {magic!r}"
            )
        if version != CONTROL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported control version: expected {CONTROL_SCHEMA_VERSION}, got {version}"
            )
        return cls(
            partition_id=partition_id,
            grad_norm=grad_norm,
            dataset_size=dataset_size,
            capacity_mb=capacity_mb,
            version=version,
        )


@dataclass(frozen=True)
class ServerQuantAssignmentMessage:
    """Server-to-client assigned bit-width instruction.

    Direction: DOWNLOAD (Server -> Client)
    Multiplicity: 1 per sampled client per adaptive round.
    """

    partition_id: int
    assigned_q: int
    version: int = CONTROL_SCHEMA_VERSION
    direction: ControlDirection = ControlDirection.DOWNLOAD

    def __post_init__(self) -> None:
        if self.assigned_q < 1 or self.assigned_q > 32:
            raise ValueError(f"assigned_q must lie in [1, 32], got {self.assigned_q}")

    def serialize(self) -> bytes:
        return ASSIGNMENT_STRUCT.pack(
            ASSIGNMENT_MAGIC,
            self.version,
            self.partition_id,
            self.assigned_q,
        )

    @classmethod
    def deserialize(cls, buf: bytes) -> ServerQuantAssignmentMessage:
        if len(buf) != ASSIGNMENT_BYTE_LENGTH:
            raise ValueError(
                f"Invalid quant assignment buffer length: expected {ASSIGNMENT_BYTE_LENGTH} bytes, "
                f"got {len(buf)} bytes."
            )
        magic, version, partition_id, assigned_q = ASSIGNMENT_STRUCT.unpack(buf)
        if magic != ASSIGNMENT_MAGIC:
            raise ValueError(
                f"Invalid assignment magic: expected {ASSIGNMENT_MAGIC!r}, got {magic!r}"
            )
        if version != CONTROL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported control version: expected {CONTROL_SCHEMA_VERSION}, got {version}"
            )
        return cls(
            partition_id=partition_id,
            assigned_q=assigned_q,
            version=version,
        )
