"""Unit tests for versioned control message schemas and accounting (#88)."""

from __future__ import annotations

import math

import pytest

from fedmaq.core.control_messages import (
    ASSIGNMENT_BYTE_LENGTH,
    ASSIGNMENT_MAGIC,
    CONTROL_SCHEMA_VERSION,
    PREFLIGHT_BYTE_LENGTH,
    PREFLIGHT_MAGIC,
    ClientPreflightMessage,
    ControlDirection,
    ServerQuantAssignmentMessage,
)


def test_client_preflight_message_serialization_roundtrip() -> None:
    msg = ClientPreflightMessage(
        partition_id=7,
        grad_norm=0.4521,
        dataset_size=500,
        capacity_mb=4096.0,
    )

    assert msg.direction == ControlDirection.UPLOAD
    assert msg.version == CONTROL_SCHEMA_VERSION

    buf = msg.serialize()
    assert len(buf) == PREFLIGHT_BYTE_LENGTH
    assert buf[:4] == PREFLIGHT_MAGIC

    deserialized = ClientPreflightMessage.deserialize(buf)
    assert deserialized.partition_id == 7
    assert math.isclose(deserialized.grad_norm, 0.4521, rel_tol=1e-5)
    assert deserialized.dataset_size == 500
    assert math.isclose(deserialized.capacity_mb, 4096.0, rel_tol=1e-5)
    assert deserialized.version == CONTROL_SCHEMA_VERSION
    assert deserialized.direction == ControlDirection.UPLOAD


def test_server_quant_assignment_message_serialization_roundtrip() -> None:
    msg = ServerQuantAssignmentMessage(
        partition_id=3,
        assigned_q=8,
    )

    assert msg.direction == ControlDirection.DOWNLOAD
    assert msg.version == CONTROL_SCHEMA_VERSION

    buf = msg.serialize()
    assert len(buf) == ASSIGNMENT_BYTE_LENGTH
    assert buf[:4] == ASSIGNMENT_MAGIC

    deserialized = ServerQuantAssignmentMessage.deserialize(buf)
    assert deserialized.partition_id == 3
    assert deserialized.assigned_q == 8
    assert deserialized.version == CONTROL_SCHEMA_VERSION
    assert deserialized.direction == ControlDirection.DOWNLOAD


def test_control_messages_malformed_buffer_lengths() -> None:
    with pytest.raises(ValueError, match="buffer length"):
        ClientPreflightMessage.deserialize(b"short")

    with pytest.raises(ValueError, match="buffer length"):
        ServerQuantAssignmentMessage.deserialize(b"short")


def test_control_messages_malformed_magic_mismatch() -> None:
    msg = ClientPreflightMessage(partition_id=0, grad_norm=1.0, dataset_size=10, capacity_mb=2048.0)
    corrupted = b"XXXX" + msg.serialize()[4:]
    with pytest.raises(ValueError, match="magic"):
        ClientPreflightMessage.deserialize(corrupted)

    msg_quant = ServerQuantAssignmentMessage(partition_id=0, assigned_q=4)
    corrupted_quant = b"XXXX" + msg_quant.serialize()[4:]
    with pytest.raises(ValueError, match="magic"):
        ServerQuantAssignmentMessage.deserialize(corrupted_quant)


def test_control_messages_non_finite_fields_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        ClientPreflightMessage(
            partition_id=0, grad_norm=float("nan"), dataset_size=10, capacity_mb=2048.0
        )

    with pytest.raises(ValueError, match="finite"):
        ClientPreflightMessage(
            partition_id=0, grad_norm=1.0, dataset_size=10, capacity_mb=float("inf")
        )

    with pytest.raises(ValueError, match="non-negative"):
        ClientPreflightMessage(
            partition_id=0, grad_norm=-1.0, dataset_size=10, capacity_mb=2048.0
        )

    with pytest.raises(ValueError, match="assigned_q"):
        ServerQuantAssignmentMessage(partition_id=0, assigned_q=0)

