"""Verify that the downstream FedMAQ config is bound to both Stage 1 verdicts.

This is a Gate 2 handoff check, not a matrix runner. It reads the committed
selection-resolution reports and the Hydra-resolved ``algorithm=fedmaq`` config,
then emits the verified binding suitable for the author to preserve with the
downstream candidate evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import verify_gate_2_preconditions

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
DEFAULT_STAGE1A_RESOLUTION = Path("scripts/analysis_output/power_mean_degree_resolution.json")
DEFAULT_STAGE1B_RESOLUTION = Path("scripts/analysis_output/power_mean_omega_resolution.json")


def _read_resolution(path: Path, expected_report_type: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"missing {expected_report_type} at {path}; copy the closed selection artifact "
            "into scripts/analysis_output before declaring Gate 2"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if document.get("report_type") != expected_report_type:
        raise ValueError(
            f"{path} has report_type={document.get('report_type')!r}; "
            f"expected {expected_report_type!r}"
        )
    data = document.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at data")
    return data


def verify_gate_2_selection(
    stage_1a_resolution: Path = DEFAULT_STAGE1A_RESOLUTION,
    stage_1b_resolution: Path = DEFAULT_STAGE1B_RESOLUTION,
) -> dict[str, object]:
    """Return the validated selected pair bound to the shipped FedMAQ config."""
    stage_1a = _read_resolution(stage_1a_resolution, "power_mean_degree_resolution")
    stage_1b = _read_resolution(stage_1b_resolution, "power_mean_omega_resolution")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["algorithm=fedmaq"])
    fedmaq = OmegaConf.to_container(cfg.algorithm, resolve=True)
    return verify_gate_2_preconditions(fedmaq, stage_1a, stage_1b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1a-resolution", type=Path, default=DEFAULT_STAGE1A_RESOLUTION)
    parser.add_argument("--stage1b-resolution", type=Path, default=DEFAULT_STAGE1B_RESOLUTION)
    args = parser.parse_args()
    try:
        verified = verify_gate_2_selection(args.stage1a_resolution, args.stage1b_resolution)
    except ValueError as exc:
        print(f"Gate 2 FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(verified, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
