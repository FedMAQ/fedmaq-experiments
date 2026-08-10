"""Regression tests for the agent-context layout."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _skill_names(directory: Path) -> set[str]:
    return {skill.parent.name for skill in directory.glob("*/SKILL.md")}


def test_shared_skills_are_not_copied_into_claude() -> None:
    shared = _skill_names(REPO_ROOT / ".agents" / "skills")
    claude_native = _skill_names(REPO_ROOT / ".claude" / "skills")

    assert not shared & claude_native, "Shared skills belong only in .agents/skills."
