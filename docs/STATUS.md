# FedMAQ Project Status

Single source of truth for current project state. Updated after each experiment batch.

**Last updated**: 2026-08-01 (audit pass 4: Stage 1c pulls the grid's FedAvg reference rows forward, analysis scopes candidates by `experiment_group`, and the calendar moves every other confirmatory run behind the tag — Decisions 71–72)

---

## Important Context

> [!IMPORTANT]
> **All experiments conducted so far are exploratory smoke tests** — short-round sweeps (40–50R) on single seeds to validate the algorithm direction and identify which hyperparameters matter. They are **not** the formal thesis results. **Nothing in the 183-run confirmatory grid has been executed.** Of the three-stage exploration phase that gates it, only Stage 1 (`pass2_explore`, screening) has run, on 2026-07-31; Stages 2 and 3 are pending, so the refinement layer is unselected. Manuscript Chapters 5 and 6 are ~90% `{[PLACEHOLDER]}` for that reason; nothing there may be written as though results exist.

> [!IMPORTANT]
> **The refinement layer is not frozen.** `soft_voting`, `ema_student`, and `grad_norm_ema` all ship `true` in `conf/algorithm/fedmaq.yaml`, but that is a default, not an exploration result. The freeze happens at the end of Stage 1 in [docs/RUNBOOK.md](RUNBOOK.md)'s dispatch order. If exploration drops a mechanism, manuscript §3.5 and Chapter 5's $T = 1.0$ justification both need revisiting.

> [!WARNING]
> **Model architecture switched to MobileNetV2GN.** As of 2026-07-15, the default CIFAR model has been changed from ResNet18GN (~11.17M params) to MobileNetV2GN (~2.24M params) for **edge realism** (deployable ~2.24M model on Pi/Jetson tiers). Note: this does **not** improve the compression _ratio_ — at iso-architecture the ratio (~1.7×) is set by bit-width allocation, not param count (see Decision 1). All prior ResNet18GN smoke test results are **deprecated** and must be re-run with MobileNetV2GN. ResNet18GN remains available via `model_name="resnet18gn"` config override.

> [!IMPORTANT]
> **Baselines have never been tuned on MobileNetV2GN, and until 2026-08-01 nothing was scheduled to do it.** Decision 31 pre-registered a matched-tuning budget and Decision 29 sequenced it, but no matrix file existed, no `RUNBOOK.md` stage dispatched it, and `chapter_4.tex` had no prose describing it — while FedMAQ received a 38-run exploration phase and a 30-run formulation study on exactly this configuration. The baseline constants are *provenanced* (§4.3.2 sources each one) but their **transfer** to MobileNetV2GN at α ∈ {0.1, 1.0} is untested, every source having published for a different architecture and skew. `conf/matrix/baseline_tuning.yaml` (55 runs, Stage 1b, Decision 67) closes that gap.

---

## Manuscript Sync

Sibling repo `../fedmaq-manuscript`. Reconciliation points, most recent first:

- **2026-08-01** (pass 4) — §4.3.6 discloses that the benchmark grid's six CIFAR-10
  FedAvg rows are dispatched ahead of the rest of the grid and why that position is
  admissible (Decision 71); §4.5 stops scheduling the baselines' 90 confirmatory runs
  before the tag that locks their hyperparameter table, restating August--October as
  baseline *reproduction* and giving the 147-run confirmatory block late January plus
  February (Decision 72). The Gantt gains a `Baseline Matched-Tuning` row, absent
  since §4.3.2 was written.
- **2026-08-01** — §4.3.2 gains the baseline matched-tuning stage (Decision 67); §4.3.6
  gains the split-skew freeze rule, the total-disqualification branch, per-seed
  disqualification, the corrected tie-break, and the reserved recheck (Decisions 64–66,
  68); §4.4 restates the uncounted exploration arithmetic (Decisions 64–70).
- **2026-07-31** — §4.3.1 rewritten to state the noise-margin *rule* rather than seed
  counts that contradicted §4.4, and to describe exploration as three stages; the
  empty-freeze branch pre-registered in §4.3.1 and §4.3.7 body text; §4.3.7's
  parity-anchor claim corrected (Decisions 60–61).
- **2026-07-31** (manuscript `bebdd67`) — §4.4 and §5.7 extended with the statistical
  reasoning behind the deepened reference cell, the factorial's family-wise rate, and
  four limitations that were real, known and unwritten.
- **2026-07-30** (manuscript `14975d0`) — §4.1 and §4.3 reconciled against this repo.

§4.3.4 is canonical for hardware and software specifications; do not restate it in an
ADR or a second registry.

---

## Algorithm Variants

One variant. FedMAQ-Lite was dropped from the thesis (Decision 4) and its code
removed once nothing depended on it; the size-contrast story it carried died
when the main model became MobileNetV2GN at ~2.24M, within 4% of Lite's ~2.16M
SimpleCNN. Its archived smoke results stay in
`docs/experiments/archive/temperature-ablation/` as a historical record.

| Variant               | Client Model  | Params | Status                                          | Primary Use Case                                  |
| :-------------------- | :------------ | :----: | :---------------------------------------------- | :------------------------------------------------ |
| **FedMAQ** (`fedmaq`) | MobileNetV2GN | ~2.24M | Active development — needs MobileNetV2GN tuning | Iso-architecture baseline comparison (edge model) |

---

## Best-Known Accuracy Standings (ResNet18GN era — deprecated)

> [!WARNING]
> All standings and configs from the ResNet18GN/SimpleCNN era have moved to [docs/experiments/archive/RESNET18GN-SUMMARY.md](experiments/archive/RESNET18GN-SUMMARY.md) — retained for historical reference only, since the default CIFAR model is now MobileNetV2GN (see Decision 1). No MobileNetV2GN standings exist yet; this section will be repopulated once formal runs land.

---

## Critical Decisions — RESOLVED (2026-07-16)

All framing/methodology decisions were resolved in a grilling session on 2026-07-16. Full list + rationale: **[docs/DECISIONS.md](DECISIONS.md)**. Grid design detail: [docs/RUNBOOK.md](RUNBOOK.md) ("Dispatch Order") and each `conf/matrix/*.yaml` header, which is authoritative over any prose describing it.

---

## Key Novel Findings (Smoke Tests)

Full list with rationale: [docs/experiments/archive/RESNET18GN-SUMMARY.md](experiments/archive/RESNET18GN-SUMMARY.md) §"Key Novel Findings". Flagged for MobileNetV2GN re-validation by the three-stage exploration phase in [docs/RUNBOOK.md](RUNBOOK.md) ("Dispatch Order"), which supersedes the retired `formal-experiment-plan.md`'s Pass 1/2/3 decomposition (Decision 69).

---

## The Confirmatory Grid — 183 runs, all pending

Every one is dispatched through a matrix file. See [docs/RUNBOOK.md](RUNBOOK.md) ("Dispatch Order") for the sequence, which is load-bearing, and ("Execution Model") for where runs actually happen.

| Stage | Matrix | Runs | Manuscript |
| :---- | :----- | ---: | :--------- |
| Primary grid, CIFAR-10 | `benchmark_grid` | 42 | §4.5 |
| Primary grid, CIFAR-100 | `benchmark_grid_cifar100` | 42 | §4.5 |
| Primary grid, FEMNIST | `benchmark_grid_femnist` | 21 | §4.5 |
| Formulation study | `formulation_study` | 30 | §4.3.6 |
| Ablation (leave-one-out) | `ablation` | 42 | §4.3.7 |
| Uniform-memory control | `uniform_memory_control` | 6 | §4.1, §4.3 |
| **Total** | | **183** | |

The exploration phase (`pass2_explore` 4, `pass2_factorial` 26, `pass3_freeze_confirm` 8,
`baseline_tuning` 55 — 93 runs) runs at the held-out α = 0.3 and is **not** counted among
the 183, nor is the conditional 6-run recheck that fires if the frozen formulation is not
Formulation 3. The factorial, the freeze confirmation, and each baseline's shipped-value
cell all deepen their reference to five seeds, because that cell's spread *is* the margin
everything else in its stage is judged against (manuscript §4.4).

Six of `benchmark_grid`'s 42 rows — the CIFAR-10 `fedavg` label — are dispatched early
at Stage 1c (`--only fedavg`), because they define the formulation study's accuracy
floor and the study's verdict is what freezes the config the other 36 run. They are the
only confirmatory runs preceding the §4.3.1 tag; Stage 4's `--skip_completed` passes
over them. Decision 71, and Decision 72 for why nothing else may join them.

The three `benchmark_grid*` files share one `experiment_group`, so `scripts/analysis.py`
reads them as a single 105-run grid. Before 2026-07-30 only the CIFAR-10 file existed and
the other 63 runs lived as a commented `--multirun` in `conf/config.yaml` that also omitted
`algorithm.post_process=true`; `test_primary_grid_files_dispatch_all_105_runs` and the
parametrized `test_primary_grid_turns_the_post_process_pipeline_on` now guard both.

---

## Open Analysis Deliverables

Analyses the grid's outputs support but which are not yet built. Not experiments —
no runs attach to these.

- **Matched-bit-budget Pareto comparison** (Decision 69, carried forward from the
  retired `formal-experiment-plan.md`). FedMAQ against the pure-quantization
  baselines FedPAQ and DAdaQuant on an accuracy-vs-compression frontier, at
  *matched* bit budgets. Plotting the frontier across unmatched budgets is
  apples-to-oranges and is the reading an examiner will reach for first.
  `conf/matrix/baseline_tuning.yaml` brackets FedPAQ's `q` at {4, 8, 16} partly so
  this is constructible from runs that already exist.

---

## Reference Links

| Document                                                                                                                                                    | Purpose                                                     |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)                                                         | Resolved decisions log (single source of truth)             |
| [docs/adr/0002-hardware-telemetry-grounding.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/adr/0002-hardware-telemetry-grounding.md) | Late-2023 Hardware Grounding & Telemetry specification      |
| [docs/RUNBOOK.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/RUNBOOK.md)                                                             | Execution model, dispatch order, operational controls       |
| [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)                                       | Chronological experiment registry with per-experiment links |
| [docs/audits/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/README.md)                                                 | Codebase and algorithm audit registry & archive             |
| [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)                                                                       | Canonical glossary (resolves naming drift between repos)    |
