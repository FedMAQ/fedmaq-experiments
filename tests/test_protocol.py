from __future__ import annotations

import pytest


def test_replacement_registration_has_stage_preregistration_and_envelope() -> None:
    from fedmaq.core.protocol import register_protocol

    registration = register_protocol(
        {"protocol": "replacement-v1", "protocol_stage": "stage_1a"},
        {"commit": "a" * 40, "dirty": False},
    )

    assert registration.promotable is True
    assert registration.historical is False
    assert registration.stage == "stage_1a"
    assert len(registration.preregistration_sha256) == 64
    assert registration.assurance_envelope["schema_version"] == 1
    assert registration.assurance_envelope["sha256"]


def test_dirty_or_uncommitted_replacement_runs_cannot_be_promoted() -> None:
    from fedmaq.core.protocol import register_protocol

    registration = register_protocol(
        {"protocol": "replacement-v1", "protocol_stage": "assurance"},
        {"commit": "a" * 40, "dirty": True},
    )

    assert registration.promotable is False
    assert registration.historical is False


def test_historical_registration_is_explicitly_non_promotable() -> None:
    from fedmaq.core.protocol import register_protocol

    registration = register_protocol({}, {"commit": "a" * 40, "dirty": False})

    assert registration.name == "historical-v1"
    assert registration.historical is True
    assert registration.promotable is False


def test_unknown_protocol_stage_fails_closed() -> None:
    from fedmaq.core.protocol import register_protocol

    with pytest.raises(ValueError, match="protocol stage"):
        register_protocol(
            {"protocol": "replacement-v1", "protocol_stage": "after_results"},
            {"commit": "a" * 40, "dirty": False},
        )


def test_historical_manifest_is_not_promotable() -> None:
    from fedmaq.core.protocol import is_promotable_manifest

    assert not is_promotable_manifest({"protocol": {"name": "historical-v1"}})


def test_every_replacement_stage_has_a_distinct_contract() -> None:
    from fedmaq.core.protocol import PROTOCOL_STAGES, preregistration_contract

    contracts = [preregistration_contract(stage) for stage in sorted(PROTOCOL_STAGES)]

    assert {contract["stage"] for contract in contracts} == set(PROTOCOL_STAGES)
    assert {contract["selection_data"] for contract in contracts} == {
        "none",
        "validation_only",
        "frozen_validation_verdicts",
    }
    assert preregistration_contract("assurance")["selection_data"] == "none"
    assert preregistration_contract("assurance")["reserved_test"] is False


def test_replacement_manifest_requires_a_self_consistent_envelope() -> None:
    from fedmaq.core.protocol import is_promotable_manifest, register_protocol

    config = {"protocol": "replacement-v1", "protocol_stage": "downstream"}
    git = {"commit": "a" * 40, "dirty": False}
    registration = register_protocol(config, git)
    manifest = {
        "config_sha256": registration.assurance_envelope["config_sha256"],
        "git": git,
        "protocol": registration.as_dict(),
    }

    assert is_promotable_manifest(manifest)
    manifest["protocol"]["assurance_envelope"]["sha256"] = "tampered"
    assert not is_promotable_manifest(manifest)
