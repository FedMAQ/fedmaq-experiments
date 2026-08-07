# Experiment Registry

**Scope: exploratory sweeps that carry hand-written analysis.** Each lives in its own
directory with `results.md` (tabular data, Hydra config paths) and `comments.md`
(empirical analysis and narrative alignment).

**Formal, matrix-dispatched stages are not registered here.** They are defined by
their `conf/matrix/*.yaml` header, which is authoritative, and their execution state
lives in the pinned dispatch-state Issue. Registering them here too would be a second
tracker for the same fact, which is what this registry's rule exists to prevent — see
`CONTEXT.md` § Working conventions.

| Experiment | Directory | Description |
| :-- | :-- | :-- |
| MobileNetV2GN smoke (50R) | [mobilenetv2-smoke-50r/](mobilenetv2-smoke-50r/) | 50-round sweeps of FedAvg, FedProx, FedMAQ, DAdaQuant, FedPAQ and FedKD across α ∈ {0.1, 1.0}. |

> [!IMPORTANT]
> **Everything here is exploratory.** These are short-round, single-seed sweeps run to
> validate direction, not thesis results. Their decision rules were superseded by the
> √2σ protocol in [ADR-0008](../adr/0008-exploration-protocol-and-the-empty-refinement-layer.md).
> Read them as history.

ResNet18GN-era smoke tests (July 13–15, 2026) were deleted with the archive directory
in the 2026-08-07 context migration; they were deprecated by the MobileNetV2GN switch
([ADR-0004](../adr/0004-confirmatory-grid-design.md)). Recoverable at
**`f7a095d^:docs/experiments/archive/`**.

The soft-voting explore sweep (Pass 1, `entropy_weight` × `precision_weight`, explore-α
= 0.3, 50R single-seed) was deleted in this pass for the same reason: the √2σ-protocol
factorial in [ADR-0008](../adr/0008-exploration-protocol-and-the-empty-refinement-layer.md)
discarded every refinement mechanism it was exploring, including `soft_voting`. Recoverable
via `git log -- docs/experiments/soft-voting-explore-mobilenetv2/`.

FedKD's near-chance smoke result was a rank-starvation bug, since fixed and
re-confirmed; CFD's collapse was structural and dropped it from the stack. Both are
[ADR-0005](../adr/0005-baseline-stack-membership.md).

## Running a sweep

- **Declarative matrices only**: `uv run python scripts/run_matrix.py --matrix <name>`.
  Never Hydra `--multirun`.
- **Process-isolated**: the runner calls `kill_ray_processes()` between runs to
  eliminate CUDA VRAM leaks and Ray worker accumulation.
- **Canonical output path**:
  `outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<heterogeneity>/seed_<seed>/`
  (phases: `ci` 2R, `smoke` 50R, `explore` 50R, `formal` 100R).

Full dispatch order and operational controls: [docs/agents/execution-model.md](../agents/execution-model.md).
