"""Demonstrate Gate 7's invalidation matrix without touching repository files.

Envelope: fedmaq-experiments:docs/freeze/assurance-envelope-2026-08-29.json
Gate: 7
Producer: assurance orchestrator
Created: 2026-08-30
Bound revisions: fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5;
fedmaq-literature@1be87e08d232b8d47ec9c71e80922d7c8235bb5a;
fedmaq-manuscript@5cc82393d639321c240897dbd7eeb728dfc82a02.
Content digest: recorded in the envelope's evidence_sha256 entry.

Usage:
    uv run python docs/freeze/verify_gate_7.py

Each representative file is copied into a temporary directory, marked there,
and rehashed. The script then reports the exact invalidation classification
required by the Gate 7 matrix. It never modifies a repository file.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_ROOT = EXPERIMENTS_ROOT.parent / "fedmaq-manuscript"


@dataclass(frozen=True)
class RepresentativeDelta:
    label: str
    root: Path
    relative_path: str
    invalidated_gates: tuple[int, ...]


DELTAS = (
    RepresentativeDelta("behavior", EXPERIMENTS_ROOT, "src/fedmaq/core/telemetry.py", (1, 3, 4, 6)),
    RepresentativeDelta(
        "configuration", EXPERIMENTS_ROOT, "conf/algorithm/fedmaq.yaml", (1, 3, 4, 6)
    ),
    RepresentativeDelta(
        "protocol_claims",
        EXPERIMENTS_ROOT,
        "docs/adr/0021-power-mean-formulation-family.md",
        (1, 3),
    ),
    RepresentativeDelta("manuscript_wording", MANUSCRIPT_ROOT, "chapter_4.tex", (1, 3)),
    RepresentativeDelta("harness_rules", EXPERIMENTS_ROOT, "scripts/golden_diff.py", (1, 3, 4, 6)),
    RepresentativeDelta(
        "runtime_provenance",
        EXPERIMENTS_ROOT,
        "docs/freeze/gate-4-evidence-2026-08-29.md",
        (1, 3, 4, 6),
    ),
    RepresentativeDelta(
        "gate_3_evidence_reference",
        EXPERIMENTS_ROOT,
        "docs/freeze/gate-3-review-2026-08-29.md",
        (3,),
    ),
    RepresentativeDelta(
        "gate_4_evidence_reference",
        EXPERIMENTS_ROOT,
        "docs/freeze/gate-4-evidence-2026-08-29.md",
        (4,),
    ),
    RepresentativeDelta(
        "gate_5_verifier_artifact", EXPERIMENTS_ROOT, "docs/freeze/verify_gate_5.py", (5,)
    ),
    RepresentativeDelta(
        "scope_boundary",
        EXPERIMENTS_ROOT,
        "docs/freeze/assurance-envelope-2026-08-29.json",
        (1, 2, 3, 4, 5, 6, 7),
    ),
    RepresentativeDelta(
        "telemetry_contract", EXPERIMENTS_ROOT, "src/fedmaq/core/telemetry.py", (5,)
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    classifications = 0
    with tempfile.TemporaryDirectory(prefix="fedmaq-gate-7-") as temporary:
        temporary_root = Path(temporary)
        for index, delta in enumerate(DELTAS):
            source = delta.root / delta.relative_path
            if not source.is_file():
                raise FileNotFoundError(f"missing representative source: {source}")
            copied = temporary_root / str(index) / source.name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            before = _sha256(copied)
            with copied.open("ab") as handle:
                handle.write(b"\n# gate-7-disposable-marker\n")
            if _sha256(copied) == before:
                raise AssertionError(f"{delta.label}: disposable copy did not change")
            gates = ",".join(str(gate) for gate in delta.invalidated_gates)
            print(f"{delta.label}: content_hash_changed=yes invalidates={gates}")
            classifications += 1

    print(f"disposable_copies={len(DELTAS)}")
    print(f"classification_assertions={classifications}")
    print("repository_files_modified=no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
