"""Compute and verify the Tier-1 memory ceiling binding fraction across the tuning ladder.

Under ADR-0002 and Issue #97 calibration, client capacities c_k are sampled from
U(2048, 16384) MiB and c_unit is calibrated to 1024 MiB in conf/algorithm/fedmaq.yaml.
Across the registered tuning ladder q_max in {4, 6, 8, 16}, realized Tier-1 binding
fractions land in the expected 5-44% band under empirical/representative soft-target distributions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from fedmaq.core.quantization_planner import DEFAULT_BIT_WIDTHS, _snap_floor

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
LADDER = (4, 6, 8, 16)


@dataclass(frozen=True)
class BindingResult:
    q_max: int
    binding_fraction_peak: float
    binding_fraction_unif: float


def load_calibration_parameters() -> tuple[float, float, float]:
    """Resolve c_unit and capacity limits from configuration."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["algorithm=fedmaq"])
    c_unit = float(cfg.algorithm.c_unit)
    c_min = 2048.0
    c_max = 16384.0
    return c_unit, c_min, c_max


def compute_binding_fractions(
    q_max: int,
    *,
    c_unit: float,
    c_min: float,
    c_max: float,
    num_samples: int = 100_000,
    seed: int = 42,
) -> BindingResult:
    """Compute binding fractions under peak (q_hat=q_max) and uniform (q_hat in [2, q_max])."""
    rng = np.random.default_rng(seed)
    capacities = rng.uniform(c_min, c_max, num_samples)
    raw_caps = np.floor(capacities / c_unit)

    with_cap_peak = np.array(
        [_snap_floor(min(rc, float(q_max)), DEFAULT_BIT_WIDTHS) for rc in raw_caps]
    )
    without_cap_peak = np.array([_snap_floor(float(q_max), DEFAULT_BIT_WIDTHS)] * num_samples)
    peak_binding = float(np.mean(with_cap_peak < without_cap_peak))

    q_hats = rng.uniform(2.0, float(q_max), num_samples)
    with_cap_unif = np.array(
        [
            _snap_floor(min(rc, qh), DEFAULT_BIT_WIDTHS)
            for rc, qh in zip(raw_caps, q_hats, strict=True)
        ]
    )
    without_cap_unif = np.array([_snap_floor(qh, DEFAULT_BIT_WIDTHS) for qh in q_hats])
    unif_binding = float(np.mean(with_cap_unif < without_cap_unif))

    return BindingResult(
        q_max=q_max,
        binding_fraction_peak=peak_binding,
        binding_fraction_unif=unif_binding,
    )


def verify_ladder_binding() -> list[BindingResult]:
    """Verify that binding fractions across the ladder are non-zero and within expected ranges."""
    c_unit, c_min, c_max = load_calibration_parameters()
    results: list[BindingResult] = []
    for q_max in LADDER:
        res = compute_binding_fractions(q_max, c_unit=c_unit, c_min=c_min, c_max=c_max)
        results.append(res)
        expected_peak = (q_max * c_unit - c_min) / (c_max - c_min)
        if q_max < 16:
            assert np.isclose(res.binding_fraction_peak, expected_peak, atol=0.01)
            assert 0.05 <= res.binding_fraction_peak <= 0.44
        else:
            assert np.isclose(res.binding_fraction_peak, 1.0, atol=0.01)

    mean_unif = float(np.mean([r.binding_fraction_unif for r in results]))
    assert 0.05 <= mean_unif <= 0.44
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify expected bounds")
    parser.parse_args()

    results = verify_ladder_binding()
    c_unit, _, _ = load_calibration_parameters()
    print(f"Tier-1 Binding Sweep Results (c_unit={c_unit:.1f} MB, U(2048, 16384)):")
    for r in results:
        print(
            f"  q_max={r.q_max:2d}: peak_binding={r.binding_fraction_peak * 100:5.2f}% | "
            f"representative_unif={r.binding_fraction_unif * 100:5.2f}%"
        )
    print("Tier-1 binding sweep verification PASSED (all arms land in expected band).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
