"""Tests for the executable architecture freeze certificate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import check_freeze


def _source_manifest(tmp_path: Path, *relative_paths: str) -> dict:
    for relative_path in relative_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source: {relative_path}\n", encoding="utf-8")
    return check_freeze.build_manifest(tmp_path, scope=check_freeze.DEFAULT_SCOPE)


def test_current_freeze_manifest_is_current():
    manifest = json.loads(check_freeze.MANIFEST_PATH.read_text(encoding="utf-8"))

    assert check_freeze.check_manifest(manifest, check_freeze.REPO_ROOT) == []


def test_freeze_certificate_reports_changed_source(tmp_path):
    manifest = _source_manifest(tmp_path, "src/fedmaq.py")
    (tmp_path / "src/fedmaq.py").write_text("source: changed\n", encoding="utf-8")

    problems = check_freeze.check_manifest(manifest, tmp_path)

    assert problems == [
        "changed source: src/fedmaq.py "
        f"(expected {manifest['files']['src/fedmaq.py']}, "
        f"found {check_freeze.file_sha256(tmp_path / 'src/fedmaq.py')})"
    ]


def test_freeze_certificate_reports_new_source_outside_declared_snapshot(tmp_path):
    manifest = _source_manifest(tmp_path, "src/fedmaq.py")
    (tmp_path / "src/new_campaign_hook.py").write_text("source: new\n", encoding="utf-8")

    problems = check_freeze.check_manifest(manifest, tmp_path)

    assert problems == ["undeclared source: src/new_campaign_hook.py"]


def test_freeze_certificate_reports_missing_source(tmp_path):
    manifest = _source_manifest(tmp_path, "src/fedmaq.py")
    (tmp_path / "src/fedmaq.py").unlink()

    problems = check_freeze.check_manifest(manifest, tmp_path)

    assert problems == ["missing frozen source: src/fedmaq.py"]


def test_freeze_certificate_rejects_a_narrowed_declared_scope():
    manifest = json.loads(check_freeze.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["scope"]["include"] = ["tests/test_freeze.py"]

    problems = check_freeze.check_manifest(manifest, check_freeze.REPO_ROOT)

    assert any("freeze scope differs" in problem for problem in problems)


def test_file_hash_is_stable_across_text_line_endings(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\r\n")
    windows_digest = check_freeze.file_sha256(path)
    path.write_bytes(b"value = 1\n")

    assert check_freeze.file_sha256(path) == windows_digest
