"""Run deterministic analysis and real FedMAQ transition assurance fixtures."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import interpolate_accuracy_at_budget
from scripts.report_schema import load_report, summary, write_report
from tests.run_fixtures import run_fedmaq_q_transition_fixture


def _stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_analysis_fixture(output_dir: Path) -> dict[str, object]:
    """Run a production analysis function and validate its report envelope."""
    frame = pd.DataFrame(
        {
            "communication/cumulative_mb": [10.0, 20.0, 30.0],
            "val/accuracy": [0.4, 0.6, 0.8],
        }
    )
    value = interpolate_accuracy_at_budget(frame, 15.0)
    if value != 0.5:
        raise AssertionError(f"unexpected deterministic analysis value: {value}")
    path = output_dir / "analysis_fixture.json"
    write_report(path, "deterministic_analysis_fixture", {"accuracy_at_budget": summary([value])})
    return load_report(path, expected_type="deterministic_analysis_fixture")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fedmaq-assurance-") as first_root:
        with tempfile.TemporaryDirectory(prefix="fedmaq-assurance-") as second_root:
            first = run_fedmaq_q_transition_fixture(Path(first_root))
            second = run_fedmaq_q_transition_fixture(Path(second_root))
            first_analysis = run_analysis_fixture(Path(first_root))
            second_analysis = run_analysis_fixture(Path(second_root))
    if first != second:
        raise SystemExit("assurance fixture is not deterministic")
    if first_analysis != second_analysis:
        raise SystemExit("analysis fixture is not deterministic")
    print(f"deterministic fixture digest: {_stable_digest(first)}")
    print(f"deterministic analysis digest: {_stable_digest(first_analysis)}")
    print("FedMAQ q-transition assurance fixture passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
