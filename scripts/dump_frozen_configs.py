"""Snapshot the resolved FedMAQ configs into ``docs/freeze/``.

Chapter 6 §6.2 promises the frozen configuration is recoverable from a tagged
commit. The §4.3.7 ablation arms used to satisfy that by restating every knob as
a literal, which made each arm self-describing at the cost of making the freeze a
five-file hand edit at the highest-stakes moment in the runbook. The arms now
inherit ``conf/algorithm/fedmaq.yaml`` through their defaults list and carry only
their own removal, so the parity requirement is structural -- but reading an arm
no longer tells you what it runs.

This script buys the self-describing property back in generated form. It composes
each ``fedmaq*`` algorithm config exactly as ``scripts/run.py`` would and writes
the resolved result, so the snapshot cannot drift from the configs the way five
hand-maintained copies could. Regenerating it is part of the freeze commit.

    uv run python scripts/dump_frozen_configs.py          # write docs/freeze/
    uv run python scripts/dump_frozen_configs.py --check  # verify it is current

``--check`` exits non-zero when the snapshot is stale, so a freeze commit that
edits ``fedmaq.yaml`` without regenerating is caught rather than shipped.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
SNAPSHOT_PATH = REPO_ROOT / "docs" / "freeze" / "resolved_configs.yaml"

# Chapter 6 §6.2's "recoverable frozen configuration" covers the algorithm layer:
# the mechanism set and the combination logic. The dataset/experiment groups are
# swept by the matrices rather than frozen, so composing at their defaults would
# snapshot a value no run necessarily uses.
_ALGORITHM_GLOB = "fedmaq*.yaml"


def _resolved_algorithm_cfg(name: str) -> dict:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"algorithm={name}"])
    return OmegaConf.to_container(cfg.algorithm, resolve=True)


def _render() -> str:
    """Build the snapshot text: every arm's resolved config plus its hash."""
    # Imported here so the script still runs from a checkout without the package
    # installed, for the --check path in CI.
    from fedmaq.core.manifest import config_sha256

    names = sorted(p.stem for p in (CONF_DIR / "algorithm").glob(_ALGORITHM_GLOB))
    if not names:
        raise SystemExit(f"no algorithm configs matched {_ALGORITHM_GLOB}")

    blocks = [
        "# GENERATED FILE -- do not edit by hand.",
        "# Regenerate with: uv run python scripts/dump_frozen_configs.py",
        "#",
        "# Hydra-resolved form of every conf/algorithm/fedmaq*.yaml, committed so that",
        "# the tag chapter 6 §6.2 points at records what each §4.3.7 ablation arm",
        "# actually ran, not just the one-line removal its config file states.",
        "# `config_sha256` uses the same digest function as run_manifest.json but is",
        "# taken over the algorithm block ALONE, so it is stable across the dataset,",
        "# skew, and seed a matrix sweeps. It will not equal any run manifest's",
        "# config_sha256, which covers the whole resolved run config; it identifies the",
        "# arm, not the run.",
        "",
    ]
    for name in names:
        cfg = _resolved_algorithm_cfg(name)
        blocks.append(f"{name}:")
        blocks.append(f"  config_sha256: {config_sha256(cfg)}")
        body = OmegaConf.to_yaml(OmegaConf.create(cfg), sort_keys=True).rstrip("\n")
        blocks.extend(f"  {line}" if line else "" for line in body.split("\n"))
        blocks.append("")
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed snapshot differs from the configs",
    )
    args = parser.parse_args()

    rendered = _render()

    if args.check:
        if not SNAPSHOT_PATH.is_file():
            print(f"missing snapshot: {SNAPSHOT_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        if SNAPSHOT_PATH.read_text(encoding="utf-8") != rendered:
            print(
                f"{SNAPSHOT_PATH.relative_to(REPO_ROOT)} is stale; regenerate with "
                "`uv run python scripts/dump_frozen_configs.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{SNAPSHOT_PATH.relative_to(REPO_ROOT)} is current")
        return 0

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)} ({date.today().isoformat()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
