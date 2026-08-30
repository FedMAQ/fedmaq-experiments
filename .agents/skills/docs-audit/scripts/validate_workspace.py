"""Read-only structural validator for the six-repository agent context.

The validator intentionally reports semantic authority lookalikes instead of
guessing which document owns a concept.  It has no write path: ``--self-test``
uses a temporary directory and removes it when the process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

REPOSITORIES = (
    "fedmaq-experiments",
    "fedmaq-literature",
    "fedmaq-analyses",
    "fedmaq-manuscript",
    "fedmaq-journal-article",
    "fedmaq-presentations",
)
ENTRYPOINTS = {"AGENTS.md", "CLAUDE.md", "CONTEXT.md"}
TRACKED_PATTERNS = (
    re.compile(r"^AGENTS\.md$"),
    re.compile(r"^CLAUDE\.md$"),
    re.compile(r"^CONTEXT\.md$"),
    re.compile(r"^\.agents/rules/.+\.md$"),
    re.compile(r"^\.agents/skills/.+/SKILL\.md$"),
    re.compile(r"^\.claude/rules/.+\.md$"),
    re.compile(r"^\.claude/skills/.+/SKILL\.md$"),
    re.compile(r"^docs/adr/.+\.md$"),
    re.compile(r"^docs/agents/.+\.md$"),
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
IMPORT = re.compile(r"^\s*@([^\s]+)\s*$", re.MULTILINE)
AUTHORITY_WORDS = re.compile(
    r"\b(?:canonical|authoritative|source of truth|sole live|owns)\b", re.I
)


def _tracked(repo: Path) -> set[str]:
    git = repo / ".git"
    if not git.exists():
        return set()
    # A fixture may be deliberately non-git; the structural checks still run.
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {line.replace("\\", "/") for line in result.stdout.splitlines()}


def _scope_path(root: Path, repo_name: str, relative: str) -> Path:
    return root / repo_name / Path(relative)


def _records(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in inventory.get("records", []):
        for path in record.get("paths", [record.get("path")]):
            if not path:
                continue
            item = dict(record)
            item.pop("paths", None)
            item["path"] = path
            result[(item["repo"], path)] = item
    return result


def _exceptions(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in inventory.get("intentional_exceptions", [])}


def _is_exception(
    exceptions: dict[str, dict[str, Any]], exception_id: str, repo: str, path: str
) -> bool:
    item = exceptions.get(exception_id)
    if not item:
        return False
    return any(
        entry.get("repo") == repo and entry.get("path") == path for entry in item.get("paths", [])
    )


def _discover(repo: Path) -> set[str]:
    candidates: set[str] = set()
    for path in repo.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(repo).as_posix()
        if any(pattern.match(relative) for pattern in TRACKED_PATTERNS):
            candidates.add(relative)
    return candidates


def _links(root: Path, repo: str, path: str) -> tuple[list[str], list[str]]:
    source = _scope_path(root, repo, path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [], []
    valid: list[str] = []
    broken: list[str] = []
    for candidate in MARKDOWN_LINK.findall(text):
        target = candidate.strip().split("#", 1)[0].split("?", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if target.startswith("/"):
            # Literature OKF links are root-absolute within fedmaq-wiki.
            target_path = (root / repo / "fedmaq-wiki" / target.lstrip("/")).resolve()
        else:
            target_path = (source.parent / target).resolve()
        try:
            relative = target_path.relative_to(root).as_posix()
        except ValueError:
            broken.append(target)
            continue
        if target_path.exists():
            valid.append(relative)
        else:
            broken.append(target)
    return valid, broken


def validate(root: Path, inventory_path: Path, *, enforce_git: bool = True) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = _records(inventory)
    exceptions = _exceptions(inventory)
    failures: list[str] = []
    warnings: list[str] = []
    semantic: list[str] = []
    all_documents: dict[str, list[tuple[str, str]]] = {}
    inbound_sets: dict[str, list[str]] = {}

    if tuple(inventory.get("repositories", [])) != REPOSITORIES:
        failures.append("inventory repositories are not the declared six-repository boundary")

    for repo_name in REPOSITORIES:
        repo = root / repo_name
        if not repo.is_dir():
            failures.append(f"missing repository: {repo_name}")
            continue
        discovered = _discover(repo)
        tracked = _tracked(repo) if enforce_git else discovered
        expected = {path for (owner, path) in records if owner == repo_name}
        missing_records = sorted(discovered - expected)
        extra_records = sorted(expected - discovered)
        for path in missing_records:
            failures.append(f"unrecorded in-scope path: {repo_name}/{path}")
        for path in extra_records:
            failures.append(f"inventory path is absent: {repo_name}/{path}")
        if enforce_git:
            for path in sorted(discovered - tracked):
                failures.append(f"in-scope path is not tracked: {repo_name}/{path}")
        for required in ENTRYPOINTS:
            if not (repo / required).is_file():
                failures.append(f"missing entrypoint: {repo_name}/{required}")
        claude_text = (repo / "CLAUDE.md").read_text(encoding="utf-8", errors="replace")
        if "@AGENTS.md" not in claude_text:
            failures.append(f"CLAUDE.md is not an AGENTS wrapper: {repo_name}/CLAUDE.md")
        for path in _discover(repo):
            if "archive" in Path(path).parts:
                failures.append(f"archived agent material: {repo_name}/{path}")

        for path in sorted(discovered):
            item = records.get((repo_name, path))
            if not item:
                continue
            if not item.get("owner") or not item.get("disposition"):
                failures.append(f"incomplete disposition: {repo_name}/{path}")
            if item.get("unresolved") is not False:
                failures.append(f"unresolved disposition: {repo_name}/{path}")
            actual, broken = _links(root, repo_name, path)
            if broken:
                failures.append(
                    f"broken current link in {repo_name}/{path}: " + ", ".join(sorted(broken))
                )
            actual = sorted(actual)
            for target in actual:
                inbound_sets.setdefault(target, []).append(f"{repo_name}/{path}")

            if path == "AGENTS.md":
                imports = IMPORT.findall((repo / path).read_text(encoding="utf-8"))
                if imports and not _is_exception(
                    exceptions, "baseline-nested-agents-imports", repo_name, path
                ):
                    failures.append(f"nested AGENTS import remains: {repo_name}/{path}")
            if AUTHORITY_WORDS.search((repo / path).read_text(encoding="utf-8", errors="replace")):
                semantic.append(f"authority candidate: {repo_name}/{path}")

        # Exact duplicate content is mechanical; meaning-level overlap is not.
        digests: dict[str, list[str]] = {}
        for path in sorted(discovered):
            payload = (repo / path).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            digests.setdefault(digest, []).append(path)
            all_documents.setdefault(digest, []).append((repo_name, path))
        for paths in digests.values():
            if len(paths) > 1:
                failures.append(f"exact duplicate live documents: {repo_name}/" + ", ".join(paths))

    for (repo_name, path), item in records.items():
        key = f"{repo_name}/{path}"
        actual = sorted(set(inbound_sets.get(key, [])))
        inbound_sets[key] = actual
        declared = sorted(item.get("inbound_refs", []))
        if "<validator-derived>" not in declared and actual != declared:
            failures.append(f"inbound reference set mismatch for {key}: " + ", ".join(actual))

    for paths in all_documents.values():
        if len(paths) > 1:
            covered = all(
                _is_exception(exceptions, "exact-duplicate-shared-procedure", repo, path)
                for repo, path in paths
            )
            if not covered:
                rendered = ", ".join(f"{repo}/{path}" for repo, path in paths)
                failures.append(f"exact duplicate live documents across repositories: {rendered}")

    # Intentional exceptions must be visible, scoped, and tied to a successor.
    for exception_id, exception in exceptions.items():
        if not exception.get("reason") or not exception.get("owner"):
            failures.append(f"exception lacks owner/reason: {exception_id}")
        if exception.get("status") not in {"baseline-only", "intentional"}:
            failures.append(f"exception has invalid status: {exception_id}")
        if exception.get("status") == "baseline-only" and not exception.get("expires_after"):
            failures.append(f"baseline exception lacks expiry: {exception_id}")

    external = inventory.get("exclusions", [])
    if not external:
        failures.append("inventory has no explicit out-of-scope exclusions")
    else:
        warnings.extend("excluded: " + str(item) for item in external)

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "semantic_candidates": sorted(set(semantic)),
        "inbound_reference_sets": dict(sorted(inbound_sets.items())),
        "counts": {
            "repositories": len(REPOSITORIES),
            "records": len(records),
            "failures": len(failures),
            "semantic_candidates": len(set(semantic)),
        },
    }
    return result


def _self_test(script: Path) -> int:
    fixture_dir = script.parent.parent / "fixtures" / "negative"
    with tempfile.TemporaryDirectory(prefix="fedmaq-docs-audit-") as temporary:
        root = Path(temporary)
        inventory = {
            "repositories": list(REPOSITORIES),
            "records": [
                {
                    "repo": repo,
                    "paths": ["AGENTS.md", "CLAUDE.md", "CONTEXT.md"],
                    "owner": repo,
                    "disposition": "retain",
                    "successor": None,
                    "inbound_refs": ["<validator-derived>"],
                    "unresolved": False,
                }
                for repo in REPOSITORIES
            ],
            "exclusions": ["installed/shared skills outside these repositories"],
        }
        inventory_path = root / "inventory.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        for repo in REPOSITORIES:
            directory = root / repo
            directory.mkdir()
            (directory / "AGENTS.md").write_text("# entry\n", encoding="utf-8")
            (directory / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
            (directory / "CONTEXT.md").write_text("# context\n", encoding="utf-8")
        shutil.copyfile(fixture_dir / "broken-link.md", root / REPOSITORIES[0] / "AGENTS.md")
        result = validate(root, inventory_path, enforce_git=False)
        if result["status"] != "FAIL" or not any(
            "broken current link" in item for item in result["failures"]
        ):
            print("FAIL: broken-link negative fixture did not fail as expected")
            return 1
        shutil.copyfile(
            fixture_dir / "broken-agents-import.md", root / REPOSITORIES[0] / "AGENTS.md"
        )
        result = validate(root, inventory_path, enforce_git=False)
        if result["status"] != "FAIL" or not any(
            "nested AGENTS import" in item for item in result["failures"]
        ):
            print("FAIL: broken-agents-import negative fixture did not fail as expected")
            return 1
    print("PASS: negative fixtures detected; temporary workspace removed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inventory", type=Path, default=Path("docs/agents/context-modernization-inventory.json")
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    if args.self_test:
        return _self_test(Path(__file__).resolve())
    result = validate(args.workspace.resolve(), (args.workspace / args.inventory).resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["status"])
        for key in ("failures", "warnings", "semantic_candidates"):
            for value in result[key]:
                print(f"{key}: {value}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
