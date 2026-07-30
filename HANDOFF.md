# Handoff Context: FedMAQ Experiments

**Purpose**: Operational orientation and immediate action items for the next agent. Historical details, audit findings, and resolved methodology decisions are maintained in `docs/`.

**Last updated**: 2026-07-30

---

## Quick Pointers & Primary Resources

- **Current State & Standings**: [docs/STATUS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/STATUS.md)
- **Resolved Methodology & Framing Decisions**: [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)
- **Experiment Registry & Historical Runs**: [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)
- **Terminology & Glossary**: [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)

---

## Current Project State Summary

FedMAQ trains **MobileNetV2GN** (~2.24M params) on CIFAR-10 as its primary thesis model.

- **Experiment Script & Output Refactor (Decision 39)**: Standardized output paths into a single canonical hierarchy (`outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<heterogeneity>/seed_<seed>/`) and replaced 17 obsolete ad-hoc Python scripts with declarative YAML matrix manifests (`conf/matrix/*.yaml`) executed via `uv run python scripts/run_matrix.py --matrix <name>`.
- **Exploration Phase**: **Not yet run.** Decisions 33-35's Pass 1 pick (`ew=2.0, pw=0.5, soft_voting=true`) is superseded — the three-stage exploration in `conf/matrix/pass2_*.yaml` and `pass3_freeze_confirm.yaml` decides the refinement layer now, and its own headers are authoritative over this file. All three refinements currently ship `true`, which is a default and not a result.
- **Telemetry Grounding**: Finalized Late-2023 Hardware Ecosystem (Decisions 36–38, [ADR-0002](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/adr/0002-hardware-telemetry-grounding.md)): Raspberry Pi 5 (Cortex-A76 @ 2.4GHz) edge clients (**20.0 s/s** sustained MobileNetV2GN, 600.0 s/s SimpleCNN), 10 Mbps 802.11ac Wi-Fi, and 24-Core Intel Xeon 5th Gen + NVIDIA L40S 48GB + 64GB DDR5 RAM FL Server (**5,000 s/s** CIFAR / **10,000 s/s** FEMNIST, per-dataset via `resolve_server_compute_speed()`). The Pi 5 grounds the *throughput* constants only; the $\mathcal{U}(2048, 16384)$ MB memory range is DynFed's and is continuous, with no tiers, bands, or hardware variants. Reachable Tier-1 caps are $\{4,5,6,7,8,16\}$ — 32-bit is structurally unreachable.
- **Manuscript sync**: §4.1 and §4.3 were reconciled against this repo on 2026-07-30 (manuscript commit `14975d0`). The telemetry note that used to sit here is discharged.
- **Baseline Comparators**: 6 formal baselines (FedAvg, FedProx, FedPAQ, DAdaQuant, FedDistill, FedKD). FedMD & CFD are dropped (Decisions 25/26).
- **The confirmatory grid is 183 runs and none have been executed.** Manuscript Chapters 5 and 6 are ~90% `{[PLACEHOLDER]}` as a direct consequence.

---

## Dispatch Order

**The order below is load-bearing, not a convenience.** Manuscript §4.5 states it, and
running out of order costs runs rather than just time. Each matrix file's own header
carries the reason it sits where it does; read it before dispatching that stage.

### Stage 1 — Exploration (gates everything downstream)

Runs at the held-out $\alpha = 0.3$, absent from the confirmatory grid by design, so
no mechanism is ever selected at a skew it is later reported on. Not counted among
the 183.

1. `--matrix pass2_explore` — screening, R=50, one seed, 4 runs.
2. `--matrix pass2_factorial` — keep-or-drop, fully crossed $2^3$, three seeds, 24 runs.
   Then `scripts/analysis.py:exploration_noise_margin` writes
   `scripts/analysis_output/exploration_margin.json`.
3. **Edit `pass3_freeze_confirm.yaml`** — replace the placeholder overrides in the
   `fedmaq-surviving-set` arm with `surviving_refinement_set` from that JSON. The
   values shipped there are today's defaults, not a prediction.
4. `--matrix pass3_freeze_confirm` — R=100, three seeds, 4 runs.
5. **Freeze.** Write the surviving set into `conf/algorithm/fedmaq.yaml` *and* every
   §4.3.7 ablation arm (`test_ablation_arms_share_one_refinement_layer` enforces they
   stay identical), then git-tag. Manuscript §6.2 promises that tag.

### Stage 2 — Formulation study (30 runs)

6. `--matrix formulation_study`. Must carry Stage 1's surviving layer: its
   Formulation 1 cell is Ablation Configuration 4's parity anchor, and an anchor only
   anchors if it carries the same refinement layer as the arm.

### Stage 3 — Ablation (42 runs)

7. **Before dispatch**, if Formulation 1 or 2 won Stage 2, revisit all four FedMAQ
   arm configs and `ABLATION_ARM_DIFFS` — `fedmaq_no_data`'s removal becomes
   `gamma2=0` and `fedmaq_no_state` stops being the fallback arm. The arms are
   currently pinned to Formulation 3.
8. `--matrix ablation`.

### Stage 4 — Primary grid (105 runs) and control arm (6 runs)

The six baselines here are independent of Stages 1–3 and may run at any time;
FedMAQ's own rows need the freeze.

9. `--matrix benchmark_grid` (CIFAR-10, 42), `--matrix benchmark_grid_cifar100` (42),
   `--matrix benchmark_grid_femnist` (21). All three share one `experiment_group`, so
   `analysis.py` reads them as the single 105-run grid the manuscript describes.
10. `--matrix uniform_memory_control` (6).

**42 + 42 + 21 + 30 + 42 + 6 = 183.** `test_primary_grid_files_dispatch_all_105_runs`
asserts the primary-grid share of that arithmetic.

---

## Key Operational Controls

- **Declarative Matrix Runner Mandate**: Hydra `--multirun` causes CUDA VRAM leaks, and
  it lands runs in a date-keyed `multirun/` tree with no `experiment_group`. Always
  launch sweeps with `uv run python scripts/run_matrix.py --matrix <name>`. Every
  confirmatory run has a matrix file; if you find yourself hand-typing a `--multirun`
  for one, the file is missing and should be written instead. The CIFAR-100 and FEMNIST
  halves of the primary grid were exactly that omission until 2026-07-30 — 63 of 183
  runs specified only as a comment in `conf/config.yaml`, and that comment also dropped
  `algorithm.post_process=true`.
- **`post_process` follows the comparison partner, not the algorithm.** ON for the three
  `benchmark_grid*` files and `uniform_memory_control`; OFF for `formulation_study` and
  every `ablation` arm. Both directions are enforced in `tests/test_simulation.py`.
- **RAM Headroom & Crash Recovery**: Check system RAM headroom before Flower simulations.
  Resume crashed matrix sweeps using `--start_at N` (1-indexed, and the index is into
  that matrix's own task list — re-read the dry run before resuming).
- **Dry-run first**: `--dry_run` prints every composed command without executing. On a
  shared host this is the cheapest way to catch a wrong `experiment=` or output path.
