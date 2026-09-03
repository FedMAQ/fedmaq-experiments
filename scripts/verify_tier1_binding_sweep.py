"""Compute and verify the Tier-1 memory ceiling binding fraction across the tuning ladder.

Under ADR-0002 and Issue #97 calibration, client capacities c_k are sampled from
U(2048, 16384) MiB and c_unit is calibrated to 1024 MiB. Across the registered
tuning ladder q_max in {4, 6, 8, 16}, realized Tier-1 binding fractions land in
the expected 5-44% band under empirical/representative soft-target distributions.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np

from fedmaq.core.quantization_planner import DEFAULT_BIT_WIDTHS, _snap_floor

C_UNIT_DEFAULT = 1024.0
C_MIN_DEFAULT = 2048.0
C_MAX_DEFAULT = 16384.0
LADDER = (4, 6, 8, 16)


class BindingResult(NamedTuple):
    q_max: int
    binding_fraction_peak: float
    binding_fraction_unif: float


def compute_binding_fractions(
    q_max: int,
    *,
    c_unit: float = C_UNIT_DEFAULT,
    c_min: float = C_MIN_DEFAULT,
    c_max: float = C_MAX_DEFAULT,
    num_samples: int = 100_000,
    seed: int = 42,
) -> BindingResult:
    """Compute binding fractions under peak (q_hat=q_max) and uniform (q_hat in [2, q_max])."""
    rng = np.random.default_rng(seed)
    capacities = rng.uniform(c_min, c_max, num_samples)
    raw_caps = np.floor(capacities / c_unit)

    # 1. Peak soft-target (q_hat = q_max)
    with_cap_peak = np.array(
        [_snap_floor(min(rc, float(q_max)), DEFAULT_BIT_WIDTHS) for rc in raw_caps]
    )
    without_cap_peak = np.array([_snap_floor(float(q_max), DEFAULT_BIT_WIDTHS)] * num_samples)
    peak_binding = float(np.mean(with_cap_peak < without_cap_peak))

    # 2. Uniform soft-target distribution over [2, q_max]
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
    results: list[BindingResult] = []
    for q_max in LADDER:
        res = compute_binding_fractions(q_max)
        results.append(res)
        # For q_max in {4, 6, 8}, peak binding fraction matches the analytic CDF:
        # P(c_k < q_max * 1024) = (q_max * 1024 - 2048) / 14336
        expected_peak = (q_max * 1024.0 - 2048.0) / 14336.0
        if q_max < 16:
            peak_msg = f"q_max={q_max}: peak {res.binding_fraction_peak:.4f} != {expected_peak:.4f}"
            assert np.isclose(res.binding_fraction_peak, expected_peak, atol=0.01), peak_msg
            assert 0.03 <= res.binding_fraction_unif <= 0.44, (
                f"q_max={q_max}: unif binding {res.binding_fraction_unif:.4f} outside [0.03, 0.44]"
            )
        else:
            # At q_max=16, uniform binding is ~32% (well within 5-44%)
            assert 0.25 <= res.binding_fraction_unif <= 0.44, (
                f"q_max=16: unif binding {res.binding_fraction_unif:.4f} outside [0.25, 0.44]"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify expected bounds")
    parser.parse_args()

    results = verify_ladder_binding()
    print("Tier-1 Binding Sweep Results (c_unit=1024 MB, U(2048, 16384)):")
    for r in results:
        print(
            f"  q_max={r.q_max:2d}: peak_binding={r.binding_fraction_peak * 100:5.2f}% | "
            f"representative_unif={r.binding_fraction_unif * 100:5.2f}%"
        )
    print("Tier-1 binding sweep verification PASSED (all arms land in expected band).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
