"""Declare a dated assurance envelope pinning the current candidate commit.

Run from fedmaq-experiments on a machine where fedmaq-literature and
fedmaq-manuscript are sibling checkouts and GitHub is reachable. Every repository
must be clean and at its published origin/main; the freeze certificate must be
current. The envelope never overwrites an earlier one: a second declaration on the
same day gets a numeric suffix.

The purpose, issues, and pin semantics are per-candidate arguments, so each
envelope states what it certifies instead of inheriting an earlier campaign's text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from fedmaq.core.run_identity import config_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SPECS = {
    "fedmaq-experiments": (
        REPO_ROOT,
        "remediated pipeline implementation, configuration, dispatch, tests",
    ),
    "fedmaq-literature": (
        REPO_ROOT.parent / "fedmaq-literature",
        "curated method and protocol knowledge",
    ),
    "fedmaq-manuscript": (
        REPO_ROOT.parent / "fedmaq-manuscript",
        "thesis method, protocol, and reporting claims",
    ),
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def revision_entry(name: str, repo: Path, role: str) -> dict[str, object]:
    identity = f"https://github.com/FedMAQ/{name}"
    git(repo, "rev-parse", "--show-toplevel")
    if git(repo, "status", "--porcelain"):
        raise SystemExit(f"{name} working tree is dirty; refusing to assert clean_tree")
    revision = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "remote", "get-url", "origin").removesuffix(".git")
    if remote != identity:
        raise SystemExit(f"{name} origin is {remote}, expected {identity}")
    live_main = git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
    if revision != live_main:
        raise SystemExit(f"{name} HEAD {revision} is not published origin/main {live_main}")
    return {"canonical_identity": identity, "revision": revision, "clean_tree": True, "role": role}


def next_path(declared: str) -> Path:
    out = REPO_ROOT / "docs" / "freeze" / f"assurance-envelope-{declared}.json"
    suffix = 2
    while out.exists():
        out = out.with_name(f"assurance-envelope-{declared}-{suffix}.json")
        suffix += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purpose", required=True, help="what this envelope certifies")
    parser.add_argument(
        "--specification-issue", required=True, help="e.g. FedMAQ/fedmaq-experiments#120"
    )
    parser.add_argument(
        "--execution-issue", required=True, help="e.g. FedMAQ/fedmaq-experiments#103"
    )
    parser.add_argument("--pin-semantics", required=True, help="why this commit is the candidate")
    parser.add_argument("--created-by", required=True, help="who declared the envelope")
    args = parser.parse_args()

    revision_vector = {
        name: revision_entry(name, repo, role) for name, (repo, role) in REPO_SPECS.items()
    }
    freeze = subprocess.run(
        [sys.executable, "scripts/check_freeze.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if freeze.returncode != 0:
        raise SystemExit(
            "freeze certificate is not current; state_at_candidate would be false:\n"
            + freeze.stdout
            + freeze.stderr
        )

    declared = date.today().isoformat()
    out = next_path(declared)
    commit = revision_vector["fedmaq-experiments"]["revision"]
    envelope = {
        "schema_version": 1,
        "envelope_id": f"fedmaq-pipeline-assurance-{out.stem.removeprefix('assurance-envelope-')}",
        "created_at": declared,
        "created_by": args.created_by,
        "canonical_location": f"fedmaq-experiments:{out.relative_to(REPO_ROOT).as_posix()}",
        "purpose": args.purpose,
        "specification": {
            "specification_issue": args.specification_issue,
            "execution_issue": args.execution_issue,
        },
        "content_hash": {"algorithm": "sha256", "value": None},
        "revision_vector": revision_vector,
        "candidate": {
            "repository": "fedmaq-experiments",
            "revision": commit,
            "pinned_on": declared,
            "pin_semantics": args.pin_semantics,
        },
        "freeze_certificate": {
            "path": "docs/freeze/source_manifest.json",
            "schema_version": 1,
            "state_at_candidate": "current",
        },
    }
    digest = config_sha256(envelope)
    envelope["content_hash"]["value"] = digest
    out.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT).as_posix()} with experiments candidate {commit}")
    print(f"content_hash {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
