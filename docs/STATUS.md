# FedMAQ Project Status

Single source of truth for current project state. Updated after each experiment batch.

**Last updated**: 2026-08-06 (Table 4.1's provenance is reproducible from committed code — `baseline_tuning_margin()` replaces the analyser the Stage 1b matrix named but which filters `algorithm == "fedmaq"` and could never read its own runs — and the `pre-registration` tag moves from `951f96a` to `b6b17b9`, the earliest commit carrying all three of §4.3.1's artifacts correctly and the last before the ablation dispatched — Decisions 87, 88. Before it: One iso-byte scorer now serves the formulation study, the baseline table and the ablation; measured against it, **FedMAQ does not beat FedAvg at equal bytes** — −5.44pp paired at α=1.0, null at α=0.1, against 38–40% fewer bytes — Decision 86. Before it: Formulation 2 frozen, ablation arms re-expressed, degenerate recheck skipped — Decisions 84, 85. Before it: 2026-08-02 (Stage 3 (`pass3_freeze_confirm`) skipped as degenerate — its surviving-set arm would have been config-identical to unrefined — and the pre-registered empty-freeze branch executed directly: `conf/algorithm/fedmaq.yaml` now freezes `soft_voting`/`ema_student`/`grad_norm_ema` all `false`, Ablation Configuration 8 dropped from `conf/matrix/ablation.yaml` (confirmatory total 183→177), `docs/freeze/resolved_configs.yaml` regenerated, full suite 188 passed — Decision 80. **The manuscript's chapter_6.tex §6.2 contribution bullet resting on Configuration 8 still needs withdrawing in the sibling `fedmaq-manuscript` repo — not done, out of scope here.** Before it: `pass2_factorial` completed 26/26 clean under the Decision 78 fix — the surviving set came back **empty**: none of the seven refinement cells clears the noise margin, `soft_voting` is the closest at +3.3pp against a 5.6pp margin, `grad_norm_ema` and `ema_student` both trend negative individually. Per-seed data checked (not just the aggregate) to rule out a corrupted run given σ's wide 5-seed CI; the spread is genuine, not an artifact — Decision 79. Before it: Stage 1.2's re-dispatch aborted again at tasks 12–14 — not Decision 77's trigger recurring, but a teardown gap: a timeout-killed run's Ray processes could survive `ray stop` on Linux with nothing to force them down, contaminating the next task's `ray.init()`. `kill_ray_processes()` now force-kills on non-Windows too, matching the Windows branch — Decision 78. Before it, 2026-08-01: Stage 1.2's first dispatch died on its first three tasks; `client_gpus: 0.5` was measured and exonerated, the real defect was partition-ID resolution with no retry, which now aborts loudly rather than failing at the first blip — Decision 77. Before it, pass 5, pre-dispatch: the three exploration matrices were writing every cell of a stage into one output directory — fixed with `variant:` and a guard test across all twelve matrices, Decision 76. Pass 4 before it: Stage 1c pulls the grid's FedAvg reference rows forward, the calendar moves every other confirmatory run behind the tag, DAdaQuant's ceiling is unmixed from FedPAQ's units, Chapter 6 learns its own failure branches, and the uniform-memory arm is reframed as the memory-blind condition it actually is — Decisions 71–75)

---

## Important Context

> [!IMPORTANT]
> **All experiments conducted so far are exploratory smoke tests** — short-round sweeps (40–50R) on single seeds to validate the algorithm direction and identify which hyperparameters matter. They are **not** the formal thesis results. **Nothing in the 177-run confirmatory grid has been executed.** Of the three-stage exploration phase that gated it, Stage 1 (`pass2_explore`, screening) ran on 2026-07-31 and its four arms collided into one output directory, so its screening comparison is unrecoverable and is **not** being re-run, because nothing consumes it (Decision 76). Stage 2 (`pass2_factorial`) aborted twice (Decisions 77, 78) before completing all 26 runs clean on 2026-08-02 — the noise-margin verdict was an **empty surviving set** (none of `soft_voting`/`ema_student`/`grad_norm_ema` clears the margin; per-seed data checked to rule out a data-quality cause — Decision 79). Stage 3 (`pass3_freeze_confirm`) was skipped as degenerate and the pre-registered empty-freeze branch executed directly (Decision 80): **the refinement layer is now frozen unrefined.** Manuscript Chapters 5 and 6 are ~90% `{[PLACEHOLDER]}` for that reason; nothing there may be written as though results exist.
>
> **Superseded 2026-08-06.** Stage 1b (`baseline_tuning`, 55 runs) completed — FedProx `mu` 1.0→0.01 and FedDistill `reg_alpha` 1.0→0.5 (Decision 81). Stage 2 (`formulation_study`, 30 runs) completed; its selection criterion was amended before the verdict was computed (Decision 83) and froze **Formulation 2** (Decision 84). The `ablation` matrix (36 of the 177) is executing as of this date, so "nothing in the grid has been executed" above is no longer true.
>
> **Headline comparison, 2026-08-06 (Decision 86).** Scored at equal bytes, FedMAQ at Formulation 2 loses to FedAvg by **5.44pp paired** at α=1.0 and is **indistinguishable** from it at α=0.1, while transmitting 38–40% fewer bytes. Chapters 5 and 6 may not be written around a FedMAQ-over-FedAvg accuracy claim. FedAvg is the uncompressed reference; the contribution rests on FedPAQ/DAdaQuant/FedKD at equal bytes, and those runs do not exist yet.

> [!IMPORTANT]
> **The refinement layer is frozen unrefined (2026-08-02, Decision 80).** `soft_voting`, `ema_student`, and `grad_norm_ema` all ship `false` in `conf/algorithm/fedmaq.yaml`, following Stage 2's empty surviving-set verdict (Decision 79) and the pre-registered empty-freeze branch (Decision 60). Both gates on the single tag at RUNBOOK.md step 10 cleared (Stage 1b, Decision 81; Stage 2, Decision 84) and **the `pre-registration` tag is cut, at `0dd7ef1`** (Decision 88; it previously pointed at `951f96a` and said "Formulation 3") — do not flip these flags back on without a new labelled exploration round; the freeze is a pre-registered outcome, not a placeholder. Manuscript §3.5 and Chapter 5's $T = 1.0$ justification, and the chapter_6.tex §6.2 contribution bullet resting on Configuration 8, need revisiting in the sibling `fedmaq-manuscript` repo (not done from this repo).

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
  since §4.3.2 was written. Table 4.1 gains DAdaQuant's quantization-level bounds
  and FedDistill's regularizer weight — the latter one of the five constants the
  matched-tuning stage may rewrite, with nowhere to land until now — plus a
  footnote on the levels-versus-bit-widths trap that had DAdaQuant capped at 5
  bits beside FedPAQ's 8 (Decision 73). The table is also now `\resizebox`'d; it
  had been running ~187pt past the text block since before this pass. §6.2's two
  contribution bullets now carry the branches §4.3.1 and §4.3.6 pre-register for
  them, and Chapters 1, 5 and 6 stop calling the 183 reported runs a
  "183-run confirmatory grid" — 30 of them are the formulation study, which
  precedes the grid rather than sitting in it (Decision 74). §4.1 stops claiming
  the uniform-memory control arm isolates server-side distillation — distillation
  is on and identical on both sides of that contrast — and states what the arm is:
  at 8192 MB with `c_unit = 512` the Tier-1 ceiling reaches `q_max` and binds on
  no client, so the arm is Ablation Configuration 2's memory-blind condition read
  in the post-processing regime. §4.1 also now reports the binding fraction
  (three sampled clients in seven) and §5.1 gains a bullet requiring the two
  deltas be cross-checked (Decision 75).
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

## The Confirmatory Grid — 177 runs, all pending

Every one is dispatched through a matrix file. See [docs/RUNBOOK.md](RUNBOOK.md) ("Dispatch Order") for the sequence, which is load-bearing, and ("Execution Model") for where runs actually happen.

| Stage | Matrix | Runs | Manuscript |
| :---- | :----- | ---: | :--------- |
| Primary grid, CIFAR-10 | `benchmark_grid` | 42 | §4.5 |
| Primary grid, CIFAR-100 | `benchmark_grid_cifar100` | 42 | §4.5 |
| Primary grid, FEMNIST | `benchmark_grid_femnist` | 21 | §4.5 |
| Formulation study | `formulation_study` | 30 | §4.3.6 |
| Ablation (leave-one-out) | `ablation` | 36 | §4.3.7 |
| Uniform-memory control | `uniform_memory_control` | 6 | §4.1, §4.3 |
| **Total** | | **177** | |

Ablation dropped from 42 to 36 runs (7 → 6 net-new arms) on 2026-08-02: Stage 2's
noise-margin verdict was an empty surviving refinement set, so Configuration 8
(`fedmaq_no_refinements`) has nothing left to remove and was dropped per the
pre-registered branch (Decision 60, executed as Decision 79/80). **The
manuscript's `chapter_6.tex` §6.2 contribution bullet resting on Configuration
8's contrast still needs withdrawing in the sibling `fedmaq-manuscript` repo —
not done as part of this change**, which is scoped to this repo only.

The exploration phase (`pass2_explore` 4, `pass2_factorial` 26, `pass3_freeze_confirm` 8,
`baseline_tuning` 55 — 93 runs) runs at the held-out α = 0.3 and is **not** counted among
the 177. The conditional 6-run recheck fired (the freeze is Formulation 2) but is not
spent — it compares an empty refinement layer against itself; Decision 85. The factorial, the freeze confirmation, and each baseline's shipped-value
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
| [docs/DECISIONS.md](../docs/DECISIONS.md)                                                         | Resolved decisions log (single source of truth)             |
| [docs/adr/0002-hardware-telemetry-grounding.md](../docs/adr/0002-hardware-telemetry-grounding.md) | Late-2023 Hardware Grounding & Telemetry specification      |
| [docs/RUNBOOK.md](../docs/RUNBOOK.md)                                                             | Execution model, dispatch order, operational controls       |
| [docs/experiments/README.md](../docs/experiments/README.md)                                       | Chronological experiment registry with per-experiment links |
| [docs/audits/README.md](../docs/audits/README.md)                                                 | Codebase and algorithm audit registry & archive             |
| [CONTEXT.md](../CONTEXT.md)                                                                       | Canonical glossary (resolves naming drift between repos)    |
