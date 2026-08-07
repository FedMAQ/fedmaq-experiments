# ADR-0007 — Architecture deepening: where each concern now lives

**Status**: Accepted · 2026-07-16 through 2026-07-24
**Related**: ADR-0003 (the `run_epochs` seam design), ADR-0006 (the gate that verified every step)
**Supersedes**: `docs/DECISIONS.md` Decisions 20, 42–44, 46–48, 50, 51 (file deleted; see ADR-0014)

## Context

The strategy layer had accumulated god-methods, cross-hook fallbacks, and
attribute bags: one baseline's state leaked into the shared strategy surface,
quantization policy lived inside a hook, and physical-cost simulation was
~50 lines inline in a constructor. A sequence of narrow deepening steps was run
against the bit-exact golden-diff gate (ADR-0006), each verified before the next.

This ADR records **where each concern now lives**, so a future change lands in the
right module instead of migrating back to the call site.

## Decision

### Hook decoupling

Server model-factory dispatch is centralized; cross-hook fallback defaults live in
`config_defaults.py`; the `configure_fit` god-method is split into named helpers
with an extracted `_QuantParams`; DAdaQuant's backward-compatibility property
proxies are removed from `strategy.py` and tests hit `strategy.hook.*`. The
rationale is single: **one baseline's state must not leak into the shared strategy
surface.**

### The seams, and what owns what

| Concern | Owner | Do not move it back to |
| --- | --- | --- |
| Single-model batch loop | `run_epochs` (ADR-0003) | each hook's own loop |
| Delta→compress→reconstruct tail | `compress_and_reconstruct` | `standard.py` / `fedkd.py` |
| FedMAQ quantization policy | `quantization_planner.py` | `strategy_hooks/fedmaq.py` |
| Config quintuple resolution | `resolve_run_context` in `config_defaults.py` | inline `config.get(...)` pulls |
| Physical time/bandwidth/compute/memory | `PhysicalCostModel` (`strategy.py`) | `TelemetryFedAvg.__init__`, `telemetry.py` |
| Per-round telemetry fields | `RoundSnapshot` | 7 separate `last_*` attributes |
| Server-side KD per-batch update | `kd_distill_step` | `run_server_side_kd`'s loop |

**`QuantizationPlanner`.** Owns the policy previously embedded in `FedMAQHook`:
`plan_round(...)`, the `QuantPlan` frozen dataclass (`client_q` + `grad_norms`),
`_QuantParams`, and `compute_fedmaq_q_k_t`. `inject_client_q` is a shared
FitIns-rewrite helper used by both `fedmaq.py` and `dadaquant.py`, so the two hooks
no longer duplicate it. `FedMAQHook` is now orchestration only, reading a single
`self._current_plan` field in place of a three-field spread.

> A circular import surfaced here and was fixed by making
> `strategy_hooks._partition`'s import lazy inside its only caller. That is an
> import-order fix, not a design change.

**`RunContext`.** A frozen dataclass
(`dataset_name`/`num_classes`/`batch_size`/`device`/`alg_cfg`) resolved once via
`resolve_run_context(config)`, adopted across `fedmaq.py`, `dadaquant.py`,
`fedavg_kd.py` and `cfd.py`. CFD-specific fields (`b_up`, `b_down`, `temperature`,
…) stay outside the quintuple — widening it there would be scope creep, not DRY.

**`PhysicalCostModel`.** The old `NetworkSimulator` had interface ≈ implementation
(4 ndarray fields, 3 divisions, a bare tuple return). It now owns array
construction via `from_config(config, num_clients)`, and `client_round_delay`
returns a `ClientDelay` NamedTuple that still unpacks as a 3-tuple for existing
call sites. **Per-hook time-model contributions are untouched** — this deepens the
physics-simulation seam, not the per-algorithm dispatch seam ADR-0002 settled.

**`RoundSnapshot`.** Replaces `TelemetryManager`'s seven `last_*` attributes with
one frozen dataclass, and centralizes round-0 zeroing in `snapshot_for_round`
instead of repeating it as five ternaries at the `strategy.py` call site. The
original behavioural asymmetries are preserved deliberately: on an empty-`results`
early return only `round_client_metrics` resets while the other six retain their
prior round's values, and at round 0 `round_client_metrics` is read unguarded while
the rest are zeroed.

### Verification

Every step above was gated bit-exactly against pre-refactor golden output across
the full `GOLDEN_SET`, with the full test suite green. Behaviour preservation for
`RoundSnapshot` was additionally verified by repo-wide grep that no other `src/` or
`tests/` code referenced the removed attribute names.

## Consequences

- **Future changes go to the owner in the table above**, not back to the call site.
  New telemetry fields belong in `RoundSnapshot` with round-0 behaviour decided
  once in `snapshot_for_round`. Per-algorithm timing stays in `StrategyHook`
  subclasses — do not fold that dispatch into `PhysicalCostModel`.
- `strategy.py`'s re-export of `compute_fedmaq_q_k_t` points at
  `quantization_planner`; keep that redirect if `fedmaq.py` is touched again.
- CFD retains its own separate, unrefactored KD loop. It is a dropped baseline
  (ADR-0005); do not unify it with `kd_distill_step` without a fresh ask.
- Pre-existing lint findings surfaced during these steps were deliberately left
  alone unless the change itself touched the same block. Confirm with `git stash`
  before "fixing" something a deepening step appears to have introduced.
