"""Create and verify the machine-readable architecture freeze certificate.

The certificate records the exact source files that can affect a campaign and
their SHA-256 digests.  Discovery is driven by the checked-in scope, so a new
file under a campaign source root is reported as undeclared instead of being
silently omitted from the freeze.

Usage::

    uv run python scripts/check_freeze.py --write
    uv run python scripts/check_freeze.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "freeze" / "source_manifest.json"

DEFAULT_SCOPE: dict[str, list[str]] = {
    "include": [
        ".github/workflows/**/*.yaml",
        ".github/workflows/**/*.yml",
        ".python-version",
        "conf/*.yaml",
        "conf/**/*.yaml",
        "justfile",
        "pyproject.toml",
        "scripts/**/*.py",
        "src/**/*.py",
        "tests/**/*.py",
        "uv.lock",
    ],
    "exclude": [],
}

SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""
    # Git may check the same text out with different line endings on Windows and
    # Linux. The certificate hashes repository content, not the host checkout.
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _scope(scope: dict[str, list[str]] | None) -> dict[str, list[str]]:
    selected = scope or DEFAULT_SCOPE
    return {
        "include": sorted(str(pattern) for pattern in selected.get("include", [])),
        "exclude": sorted(str(pattern) for pattern in selected.get("exclude", [])),
    }


def discover_source_files(
    repo_root: Path = REPO_ROOT,
    *,
    scope: dict[str, list[str]] | None = None,
) -> tuple[str, ...]:
    """Discover source files covered by a freeze scope in stable path order."""
    root = repo_root.resolve()
    selected = _scope(scope)
    excluded = set(selected["exclude"])
    paths: set[str] = set()
    for pattern in selected["include"]:
        candidate = root / pattern
        matches = [candidate] if candidate.is_file() else root.glob(pattern)
        for path in matches:
            if path.is_file():
                relative = _relative_path(path, root)
                if relative not in excluded:
                    paths.add(relative)
    return tuple(sorted(paths))


def build_manifest(
    repo_root: Path = REPO_ROOT,
    *,
    scope: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build a certificate for the source files currently present in ``repo_root``."""
    selected = _scope(scope)
    files = {
        relative: file_sha256(repo_root / relative)
        for relative in discover_source_files(repo_root, scope=selected)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Architecture freeze certificate for campaign-relevant configuration, "
            "runtime, dispatch, and test sources."
        ),
        "scope": selected,
        "files": files,
    }


def _manifest_files(manifest: dict[str, Any]) -> tuple[dict[str, str] | None, list[str]]:
    problems: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"unsupported freeze manifest schema: {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    scope = manifest.get("scope")
    if not isinstance(scope, dict) or not all(
        isinstance(scope.get(key), list) and all(isinstance(item, str) for item in scope[key])
        for key in ("include", "exclude")
    ):
        problems.append("invalid freeze manifest scope: expected string lists for include/exclude")
    files = manifest.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(path, str) and isinstance(digest, str) for path, digest in files.items()
    ):
        problems.append("invalid freeze manifest files: expected a path-to-SHA-256 object")
        return None, problems
    return files, problems


def check_manifest(manifest: dict[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    """Return actionable differences between a certificate and the current source tree."""
    declared, problems = _manifest_files(manifest)
    if declared is None:
        return problems

    scope = manifest.get("scope")
    expected_scope = _scope(DEFAULT_SCOPE)
    if scope != expected_scope:
        problems.append(
            "freeze scope differs from the built-in campaign scope; "
            "review scripts/check_freeze.py before regenerating the certificate"
        )
    current = set(discover_source_files(repo_root, scope=expected_scope))
    expected = set(declared)

    for relative in sorted(expected - current):
        problems.append(f"missing frozen source: {relative}")
    for relative in sorted(current - expected):
        problems.append(f"undeclared source: {relative}")
    for relative in sorted(current & expected):
        actual = file_sha256(repo_root / relative)
        expected_digest = declared[relative]
        if actual != expected_digest:
            problems.append(
                f"changed source: {relative} (expected {expected_digest}, found {actual})"
            )
    return problems


def _render(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="verify the committed certificate (the default action)",
    )
    action.add_argument(
        "--write",
        action="store_true",
        help="write a new certificate for the current source tree",
    )
    args = parser.parse_args()

    if args.write:
        manifest = build_manifest()
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(_render(manifest), encoding="utf-8")
        print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(manifest['files'])} files)")
        return 0

    if not MANIFEST_PATH.is_file():
        print(
            f"missing freeze certificate: {MANIFEST_PATH.relative_to(REPO_ROOT)}; "
            "run `uv run python scripts/check_freeze.py --write`",
            file=sys.stderr,
        )
        return 1
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read freeze certificate {MANIFEST_PATH}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print("invalid freeze manifest: expected a JSON object", file=sys.stderr)
        return 1

    problems = check_manifest(manifest)
    if problems:
        print("freeze certificate FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "The source set or its contents changed; review the campaign boundary, "
            "then regenerate with `uv run python scripts/check_freeze.py --write`.",
            file=sys.stderr,
        )
        return 1
    print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
