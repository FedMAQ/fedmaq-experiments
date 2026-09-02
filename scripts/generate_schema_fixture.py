"""Generate and verify the canonical report-schema fixture used by CI."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_schema import summary, write_report

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "recut" / "report_schema_fixture.json"


def _write(path: Path) -> None:
    write_report(
        path,
        "ci_schema_fixture",
        {
            "accuracy_at_budget": summary([0.5, 0.6, 0.7]),
            "cumulative_mb": summary([10.0, 20.0, 30.0]),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="fedmaq-schema-") as root:
            candidate = Path(root) / OUTPUT_PATH.name
            _write(candidate)
            expected = candidate.read_text(encoding="utf-8")
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != expected:
            print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is stale or missing", file=sys.stderr)
            return 1
        print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is current")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
