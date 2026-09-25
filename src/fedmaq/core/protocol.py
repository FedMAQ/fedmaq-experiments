"""Versioned replacement-pipeline registration and promotion boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

REPLACEMENT_PROTOCOL = "replacement-v1"
HISTORICAL_PROTOCOL = "historical-v1"
PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_STAGES = frozenset(
    {"matched_tuning", "stage_1a", "stage_1b", "downstream", "assurance", "v2_confirm"}
)
_PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "conf" / "protocol" / "replacement-v1.yaml"


def _sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_protocol_document() -> dict[str, Any]:
    """Load the versioned protocol file that is the source of stage semantics."""
    try:
        document = OmegaConf.to_container(OmegaConf.load(_PROTOCOL_PATH), resolve=True)
    except Exception as exc:
        raise ValueError(f"cannot load replacement protocol {_PROTOCOL_PATH}") from exc
    if not isinstance(document, dict):
        raise ValueError("replacement protocol must be a mapping")
    if document.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("replacement protocol schema version is unsupported")
    if document.get("name") != REPLACEMENT_PROTOCOL:
        raise ValueError("replacement protocol name is inconsistent")
    stages = document.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("replacement protocol must declare stages")
    return cast(dict[str, Any], document)


def preregistration_contract(stage: str) -> dict[str, Any]:
    """Return the immutable contract identifying one replacement stage."""
    document = _load_protocol_document()
    stages = document["stages"]
    if stage not in stages:
        raise ValueError(f"unknown protocol stage {stage!r}")
    stage_config = stages[stage]
    if not isinstance(stage_config, dict):
        raise ValueError(f"protocol stage {stage!r} must be a mapping")
    selection_data = stage_config.get("selection_data")
    reserved_test = stage_config.get("reserved_test")
    split = stage_config.get("split", "val" if selection_data == "validation_only" else "test")
    wire_protocol = document.get("wire_protocol", "packed_wire_v1")
    if not isinstance(selection_data, str) or not isinstance(reserved_test, bool):
        raise ValueError(f"protocol stage {stage!r} is malformed")
    return {
        "schema_version": int(document["schema_version"]),
        "protocol": str(document["name"]),
        "stage": stage,
        "selection_data": selection_data,
        "reserved_test": reserved_test,
        "split": split,
        "wire_protocol": wire_protocol,
        "historical_artifacts_promotable": document.get("historical_artifacts_promotable", False),
        "selection_domains": document.get("selection_domains", {}),
    }


def validate_matrix_against_protocol(
    matrix_name: str,
    matrix: dict[str, Any],
    expanded_count: int,
) -> None:
    """Reject semantic matrix drift before a registered scientific dispatch."""
    document = _load_protocol_document()
    contracts = document.get("matrix_contracts", {})
    contract = contracts.get(matrix_name) if isinstance(contracts, dict) else None
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError(f"protocol matrix contract {matrix_name!r} is malformed")
    rounds_value = contract.get("rounds")
    count_value = contract.get("cell_count")
    if not isinstance(rounds_value, (int, float, str)) or not isinstance(
        count_value, (int, float, str)
    ):
        raise ValueError(f"protocol matrix contract {matrix_name!r} has invalid counts")
    expected = {
        "stage": contract.get("stage"),
        "split": contract.get("split"),
        "ledger": contract.get("ledger"),
        "rounds": int(rounds_value),
        "cell_count": int(count_value),
        "sha256": contract.get("sha256"),
    }
    actual = {
        "stage": matrix.get("stage", matrix.get("protocol_stage")),
        "split": matrix.get("split", "val"),
        "ledger": matrix.get("ledger", "unregistered"),
        "rounds": int(matrix.get("total_rounds", 50)),
        "cell_count": expanded_count,
        "sha256": _sha256(matrix),
    }
    if actual != expected:
        raise ValueError(
            f"matrix {matrix_name!r} diverges from the replacement protocol: "
            f"expected={expected}, actual={actual}"
        )


@dataclass(frozen=True)
class ProtocolRegistration:
    """The protocol and candidate envelope recorded in a run manifest."""

    name: str
    stage: str
    preregistration_sha256: str
    assurance_envelope: dict[str, Any]
    promotable: bool
    historical: bool
    split: str = "val"
    wire_protocol: str = "packed_wire_v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "name": self.name,
            "stage": self.stage,
            "split": self.split,
            "wire_protocol": self.wire_protocol,
            "preregistration_sha256": self.preregistration_sha256,
            "assurance_envelope": self.assurance_envelope,
            "promotable": self.promotable,
            "historical": self.historical,
        }


def register_protocol(config: dict[str, Any], git: dict[str, Any]) -> ProtocolRegistration:
    """Register a run as replacement-pipeline or preserved historical work."""
    name = str(config.get("protocol", HISTORICAL_PROTOCOL))
    if name == HISTORICAL_PROTOCOL:
        return ProtocolRegistration(
            name=name,
            stage="historical",
            preregistration_sha256="",
            assurance_envelope={},
            promotable=False,
            historical=True,
            split="historical",
            wire_protocol="legacy",
        )
    if name != REPLACEMENT_PROTOCOL:
        raise ValueError(f"unsupported protocol {name!r}")

    stage = str(config.get("protocol_stage", "downstream"))
    contract = preregistration_contract(stage)
    split = str(config.get("split", contract["split"]))
    wire_protocol = str(config.get("wire_protocol", contract["wire_protocol"]))
    preregistration_sha256 = _sha256(contract)
    envelope_body = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": name,
        "stage": stage,
        "preregistration_sha256": preregistration_sha256,
        "config_sha256": _sha256(config),
        "git_commit": git.get("commit"),
    }
    envelope = {**envelope_body, "sha256": _sha256(envelope_body)}
    promotable = bool(git.get("commit")) and git.get("dirty") is False
    return ProtocolRegistration(
        name=name,
        stage=stage,
        preregistration_sha256=preregistration_sha256,
        assurance_envelope=envelope,
        promotable=promotable,
        historical=False,
        split=split,
        wire_protocol=wire_protocol,
    )


def is_promotable_manifest(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest may enter replacement evidence analysis."""
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        return False
    envelope = protocol.get("assurance_envelope")
    if not (
        protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION
        and protocol.get("name") == REPLACEMENT_PROTOCOL
        and protocol.get("historical") is False
        and protocol.get("promotable") is True
        and isinstance(envelope, dict)
    ):
        return False

    stage = protocol.get("stage")
    preregistration_sha256 = protocol.get("preregistration_sha256")
    if not isinstance(stage, str) or not isinstance(preregistration_sha256, str):
        return False
    try:
        contract = preregistration_contract(stage)
    except ValueError:
        return False
    if contract["historical_artifacts_promotable"] is not False:
        return False
    expected_preregistration_sha256 = _sha256(contract)
    if expected_preregistration_sha256 != preregistration_sha256:
        return False

    contract_split = contract.get("split")
    if contract_split in ("val", "test"):
        if "run" in manifest:
            run_info = manifest.get("run")
            if not isinstance(run_info, dict) or run_info.get("loader_used") != contract_split:
                return False
        elif "loader_used" in manifest and manifest.get("loader_used") != contract_split:
            return False

    envelope_body = {key: value for key, value in envelope.items() if key != "sha256"}
    git = manifest.get("git")
    return bool(
        envelope.get("sha256") == _sha256(envelope_body)
        and envelope.get("schema_version") == PROTOCOL_SCHEMA_VERSION
        and envelope.get("protocol") == REPLACEMENT_PROTOCOL
        and envelope.get("stage") == stage
        and envelope.get("preregistration_sha256") == preregistration_sha256
        and envelope.get("config_sha256") == manifest.get("config_sha256")
        and isinstance(git, dict)
        and git.get("dirty") is False
        and envelope.get("git_commit") == git.get("commit")
    )
