# FedMAQ Project Status

Single source of truth for current project state. Updated after each experiment batch.

**Last updated**: 2026-07-30 (pre-dispatch sync: primary grid made fully dispatchable)

---

## Important Context

> [!IMPORTANT]
> **All experiments conducted so far are exploratory smoke tests** — short-round sweeps (40–50R) on single seeds to validate the algorithm direction and identify which hyperparameters matter. They are **not** the formal thesis results. **Nothing in the 183-run confirmatory grid has been executed, and neither has the three-stage exploration phase that gates it.** Manuscript Chapters 5 and 6 are ~90% `{[PLACEHOLDER]}` for that reason; nothing there may be written as though results exist.

> [!IMPORTANT]
> **The refinement layer is not frozen.** `soft_voting`, `ema_student`, and `grad_norm_ema` all ship `true` in `conf/algorithm/fedmaq.yaml`, but that is a default, not an exploration result. The freeze happens at the end of Stage 1 in [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md)'s dispatch order. If exploration drops a mechanism, manuscript §3.5 and Chapter 5's $T = 1.0$ justification both need revisiting.

> [!WARNING]
> **Model architecture switched to MobileNetV2GN.** As of 2026-07-15, the default CIFAR model has been changed from ResNet18GN (~11.17M params) to MobileNetV2GN (~2.24M params) for **edge realism** (deployable ~2.24M model on Pi/Jetson tiers). Note: this does **not** improve the compression _ratio_ — at iso-architecture the ratio (~1.7×) is set by bit-width allocation, not param count (see Decision 1). All prior ResNet18GN smoke test results are **deprecated** and must be re-run with MobileNetV2GN. ResNet18GN remains available via `model_name="resnet18gn"` config override. A full hyperparameter sweep on MobileNetV2GN is required before formal experiments.

---

## Algorithm Variants

FedMAQ has been formally partitioned into two variants:

| Variant                         | Client Model  | Params | Status                                          | Primary Use Case                                    |
| :------------------------------ | :------------ | :----: | :---------------------------------------------- | :-------------------------------------------------- |
| **FedMAQ** (`fedmaq`)           | MobileNetV2GN | ~2.24M | Active development — needs MobileNetV2GN tuning | Iso-architecture baseline comparison (edge model)   |
| **FedMAQ-Lite** (`fedmaq_lite`) | SimpleCNN     | ~2.16M | Smoke tests complete — tuned                    | Demonstrates even small models beat large baselines |

---

## Best-Known Accuracy Standings (ResNet18GN era — deprecated)

> [!WARNING]
> All standings and configs from the ResNet18GN/SimpleCNN era have moved to [docs/experiments/archive/RESNET18GN-SUMMARY.md](experiments/archive/RESNET18GN-SUMMARY.md) — retained for historical reference only, since the default CIFAR model is now MobileNetV2GN (see Decision 1). No MobileNetV2GN standings exist yet; this section will be repopulated once formal runs land.

---

## Critical Decisions — RESOLVED (2026-07-16)

All framing/methodology decisions were resolved in a grilling session on 2026-07-16. Full list + rationale: **[docs/DECISIONS.md](DECISIONS.md)**. Grid design detail: [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md).

---

## Key Novel Findings (Smoke Tests)

Full list with rationale: [docs/experiments/archive/RESNET18GN-SUMMARY.md](experiments/archive/RESNET18GN-SUMMARY.md) §"Key Novel Findings". Flagged for MobileNetV2GN re-validation per [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md) ("Exploration Phase Mechanisms").

---

## The Confirmatory Grid — 183 runs, all pending

Every one is dispatched through a matrix file. See [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md) ("Dispatch Order") for the sequence, which is load-bearing.

| Stage | Matrix | Runs | Manuscript |
| :---- | :----- | ---: | :--------- |
| Primary grid, CIFAR-10 | `benchmark_grid` | 42 | §4.5 |
| Primary grid, CIFAR-100 | `benchmark_grid_cifar100` | 42 | §4.5 |
| Primary grid, FEMNIST | `benchmark_grid_femnist` | 21 | §4.5 |
| Formulation study | `formulation_study` | 30 | §4.3.6 |
| Ablation (leave-one-out) | `ablation` | 42 | §4.3.7 |
| Uniform-memory control | `uniform_memory_control` | 6 | §4.1, §4.3 |
| **Total** | | **183** | |

The exploration phase (`pass2_explore` 4, `pass2_factorial` 24, `pass3_freeze_confirm` 4)
runs at the held-out α = 0.3 and is **not** counted among the 183.

The three `benchmark_grid*` files share one `experiment_group`, so `scripts/analysis.py`
reads them as a single 105-run grid. Before 2026-07-30 only the CIFAR-10 file existed and
the other 63 runs lived as a commented `--multirun` in `conf/config.yaml` that also omitted
`algorithm.post_process=true`; `test_primary_grid_files_dispatch_all_105_runs` and the
parametrized `test_primary_grid_turns_the_post_process_pipeline_on` now guard both.

---

## Reference Links

| Document                                                                                                                                                    | Purpose                                                     |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)                                                         | Resolved decisions log (single source of truth)             |
| [docs/adr/0002-hardware-telemetry-grounding.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/adr/0002-hardware-telemetry-grounding.md) | Late-2023 Hardware Grounding & Telemetry specification      |
| [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/HANDOFF.md)                                                                       | Next-agent instructions and immediate action items          |
| [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)                                       | Chronological experiment registry with per-experiment links |
| [docs/audits/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/README.md)                                                 | Codebase and algorithm audit registry & archive             |
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                                       | Canonical glossary (resolves naming drift between repos)    |
