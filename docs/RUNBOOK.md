# Experiment Runbook

How the FedMAQ experiments are executed: where runs happen, in what order, and the
operational controls that keep a sweep recoverable. Durable operational reference —
not session context. Per-session orientation belongs in a temporary handoff file,
not in this repo.

**Last updated**: 2026-08-06

---

## Execution Model

**Read this before the dispatch order below. An agent cannot run any of it.**

- **Every reported run executes on the Linux datacenter allocation, reached through a
  JupyterHub gateway only** — no SSH, no VM, no shell an agent can drive. Hardware and
  software are specified in manuscript §4.3.4 (A100 40GB PCIe, dual Xeon Platinum 8276,
  64 GB RAM, shared with co-tenants). That section is canonical; do not restate it in an
  ADR or a second registry (`.claude/rules/docs-management.md`).
- **Sweeps are a hand-off.** The agent emits paste-ready commands, the user runs them in
  JupyterHub, the user pastes results back. Never write a plan step in which an agent
  dispatches a sweep or polls for its completion.
- **`outputs/` being empty locally proves nothing.** Results live on the allocation. Do
  not infer which stages have run from the local filesystem — ask, or ask for that
  matrix's `sweep_status.json`.
- **Experiments are not blocked on hardware.** The allocation exists and is in use
  (`pass2_explore` completed 4/4 on 2026-07-31). What gates the remaining stages is
  dispatch order, not availability.
- **The local Windows rig is for smoke tests only** — `run-minitest`, `--dry_run`,
  pre-dispatch validation, and config-composition checks. It is the validated fallback
  of §4.3.4, not where the grid runs. The Ray crash mitigations in
  `.claude/rules/flower-patterns.md` are scoped to it and apply nowhere else.

---

## Dispatch Order

**The order below is load-bearing, not a convenience.** Manuscript §4.5 states it, and
running out of order costs runs rather than just time. Each matrix file's own header
carries the reason it sits where it does; read it before dispatching that stage.
**Per the execution model above, "dispatch" means handing the command to the user.**

### Every dispatch on the allocation carries the Ray host flags

```
-o ray.temp_dir=/tmp/ray-cjb -o ray.object_store_gb=4
```

Not optional, and omitted from every command in this file until 2026-08-01. Both
default to `null`, which reproduces Ray's stock behaviour — and stock behaviour is
wrong on this host in two ways `conf/config.yaml` already documents. Ray sizes its
object store at ~30% of *total* RAM, so ~19 GB on a 64 GB box that has ~39 GB
actually free, on top of the sweep's own footprint; 4 GB is ample, since the store
only carries parameter payloads and MobileNetV2GN is ~27 MB. And the allocation is
one shared VM with per-user accounts and a **shared `/tmp`**, so a co-tenant's Ray
session collides with ours at the default `/tmp/ray`. A co-tenant is not
hypothetical: `nvidia-smi` on 2026-08-01 showed 11.6 GB of VRAM held by a process
this account cannot see.

Keep the path short and outside `$HOME` — Ray builds Unix domain sockets beneath it
and those paths cap near 107 characters.

Neither is proven to have caused the Stage 1.2 aborts (Decision 77 records what was
and was not established there). Both are cheap, documented, and remove the two known
ways this host starves a sweep that the runner's own docstring already warned about.

### Stage 1 — Exploration (gates everything downstream)

Runs at the held-out $\alpha = 0.3$, absent from the confirmatory grid by design, so
no mechanism is ever selected at a skew it is later reported on. Not counted among
the 177.

1. `--matrix pass2_explore` — screening, R=50, one seed, 4 runs. **Done 2026-07-31**
   (~70 min at `client_gpus: 1.0`, 4/4 completed). Its four cells collided into one
   output directory and the screening comparison is unrecoverable; it is not re-run,
   because nothing consumes it (Decision 76). **This line read `0.5` until
   2026-08-01 and was wrong** — `f216abf` did not change that value until 16:46 that
   day and this run ended at 10:13. The error was load-bearing: it made a
   never-exercised setting look like the one a completed sweep had validated, and
   sent the first triage of the Stage 1.2 failure at the wrong suspect (Decision 77).
2. `--matrix pass2_factorial` — keep-or-drop, fully crossed $2^3$, three seeds, **26
   runs**: the unrefined reference cell carries five seeds, not three, because its
   spread *is* the margin every other cell is judged against and at n=3 that
   estimate is uncertain by roughly a factor of twelve. Use
   `--run_timeout_seconds 2100`. Then `scripts/analysis.py:exploration_noise_margin`
   writes `scripts/analysis_output/exploration_margin.json`.
   **It takes an `experiment_group` and refuses to pool stages** — the default is
   `pass2_factorial`; pass `experiment_group="pass3_freeze_confirm"` for the R=100
   gate. It cannot produce a margin from `pass2_explore` (one seed, by design).
3. **Edit `pass3_freeze_confirm.yaml`** — replace the placeholder overrides in the
   `fedmaq-surviving-set` arm with `surviving_refinement_set` from that JSON. The
   values shipped there are today's defaults, not a prediction. That field is the
   smallest *cell* that cleared the margin, never a union across cells.
4. `--matrix pass3_freeze_confirm` — R=100, **8 runs** (unrefined arm at five seeds,
   surviving-set arm at three). Use `--run_timeout_seconds 4200`.
   **Skipped 2026-08-02** (Decision 79/80): Stage 1's surviving set came back
   empty, making both arms config-identical to unrefined — nothing to confirm.
   Went straight to step 5's empty-freeze branch instead.
5. **Freeze the refinement layer.** Write the surviving set into
   `conf/algorithm/fedmaq.yaml` — **and only there.** The §4.3.7 arms inherit that
   file via their Hydra defaults list and restate only their own removal, so one edit
   reaches all of them (Decision 58). Then
   `uv run python scripts/dump_frozen_configs.py` to refresh
   `docs/freeze/resolved_configs.yaml` (the arms are no longer self-describing, and
   that generated snapshot is what makes the tagged state readable — Decision 59) and
   run `uv run python -m pytest tests/test_simulation.py`.
   **Do not tag here.** §4.3.1 locks and tags three things together — the fixed
   mechanism set, the selected formulation, and the baseline hyperparameter table
   (Table 4.1) — and two of them do not exist until step 9. The single tag is step 10.
   **If nothing clears at R=100** the surviving set is empty — pre-registered, not a
   judgement call (Decision 60, manuscript §4.3.1 and §4.3.7). FedMAQ freezes
   unrefined, Configuration 8 drops from `conf/matrix/ablation.yaml`, and the
   `chapter_6.tex` contribution bullet resting on its contrast goes with it. No subset
   retries, no tuning to rescue a mechanism.
   `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` enforces it.

### Stage 1b — Baseline matched-tuning (55 runs)

Independent of everything above — no baseline shares configuration with FedMAQ — so
it may run concurrently with Stage 1. It is placed after it only because Decision 29
sequenced it that way and because nothing is lost by the ordering. Held-out
α = 0.3, uncounted among the 177, and its verdict enters the same tag at step 10.

6. `--matrix baseline_tuning`. R=100, 55 runs (each of five baselines: a five-seed
   reference cell at its shipped Table 4.1 value, plus two three-seed challengers).
   Use `--run_timeout_seconds 4200`. Then
   `scripts/analysis.py:baseline_tuning_margin`, same √2σ rule as the factorial.
   **Not `exploration_noise_margin`**, which this line named until 2026-08-06:
   that one filters `algorithm == "fedmaq"` and reports a completed 55-run stage
   as no runs at all.
   **Write any challenger that clears into `conf/algorithm/<baseline>.yaml` and into
   Table 4.1.** The expected outcome is that none clears and every baseline keeps its
   published value — that is a result, not a null sweep, and §4.3.2 reports it as one.
   FedAvg is absent by design: it is the uncompressed control and has no knob.

### Stage 1c — The grid's FedAvg reference rows (6 runs, not net-new)

The formulation study's accuracy floor is 90% of the uncompressed FedAvg reference
at the same dataset and skew, *"reusing the FedAvg runs already present in the
benchmark grid"* (§4.3.6). Dispatched in file order, those runs arrive at step 13 —
after the freeze they are supposed to decide. The dependency is circular unless
FedAvg's rows are pulled forward, so they are (Decision 71).

7. `--matrix benchmark_grid --only fedavg`. CIFAR-10, α ∈ {0.1, 1.0}, 3 seeds.
   **These are six of Stage 4's own 42 rows, not additional runs**, dispatched
   early under the same `experiment_group` and into the same directories; step 13's
   `--skip_completed` passes over them. `--only` refuses an unrecognized label
   rather than scheduling an empty sweep.
   They are the one confirmatory cell that runs before the step-10 tag. That is
   disclosed in §4.3.6 rather than finessed, and it is admissible because FedAvg is
   invariant to all three artifacts the tag locks: it carries no refinement layer,
   no formulation, and no tunable constant (which is why Stage 1b excludes it).
   Nothing else may move across the tag on this argument.

### Stage 2 — Formulation study (30 runs)

8. `--matrix formulation_study`. Must carry Stage 1's surviving layer: its
   Formulation 1 cell is Ablation Configuration 4's parity anchor, and an anchor only
   anchors if it carries the same refinement layer as the arm.
9. **Resolve the verdict, then write the formulation.** `select_winner` returns one
   verdict *per skew* — two, structurally, since the study runs both. Collapse them
   with `scripts/analysis.py:resolve_frozen_formulation`, which implements the
   pre-registered rule (Decisions 64–65, §4.3.6): skews agreeing freezes that
   formulation; skews diverging freezes the α = 0.1 winner and reports the split as a
   finding; one skew disqualifying its whole field defers to the other; both
   disqualifying falls back to highest mean top-1 at R=100 at α = 0.1 **and withdraws
   §4.3.6's contribution claim**. Write the result to `conf/algorithm/fedmaq.yaml`.
   **If the frozen formulation is not 3**, the reserved 6-run recheck fires — surviving
   layer vs. unrefined, 3 seeds, under the winning formulation, at α = 0.3. It is a
   veto on that layer, never a second search: the factorial is not re-opened and no
   mechanism is reconsidered. If it no longer clears, `soft_voting` leaves the frozen
   set, `fedmaq.yaml` and every §4.3.7 arm are rewritten, and Configuration 8 shrinks
   or drops. Header of `conf/matrix/formulation_study.yaml` has the full rule.

### Stage 2b — Tag the pre-registration

10. Re-run `scripts/dump_frozen_configs.py` and `pytest tests/test_simulation.py`, then
   **git-tag once.** The tag carries all three of §4.3.1's locked artifacts: the fixed
   mechanism set (step 5), the baseline hyperparameter table (step 6), and the selected
   formulation (step 9). Manuscript §6.2 promises this tag. Nothing downstream of here
   may edit a frozen config; an anomaly during confirmation opens a new labelled
   exploration round instead (§4.3.1).
   **Done 2026-08-06** — `pre-registration` at `0dd7ef1`, moved from `951f96a`
   (whose message said "Formulation 3" and which predated the baseline table).
   Decision 88 records the move, the SHA it came from, and why the tag sits after
   the ablation dispatch rather than before it.

### Stage 3 — Ablation (36 runs)

11. **Before dispatch**, if Formulation 1 or 2 was frozen at step 9, revisit
    `fedmaq_no_data` and `fedmaq_no_state` plus `ABLATION_ARM_DIFFS` —
    `fedmaq_no_data`'s removal becomes `gamma2=0`, and `fedmaq_no_state` drops its
    `formulation` override and stops being the fallback arm. Each is now a one-line
    change in a file that contains only its removal. The arms are currently pinned to
    Formulation 3 through `fedmaq.yaml`. Re-run `scripts/dump_frozen_configs.py`
    afterwards.
12. `--matrix ablation`.

### Stage 4 — Primary grid (105 runs) and control arm (6 runs)

The six baselines here need Stage 1b's verdict, not Stage 1's; FedMAQ's own rows need
the freeze.

13. `--matrix benchmark_grid --skip_completed` (CIFAR-10, 42), `--matrix benchmark_grid_cifar100` (42),
    `--matrix benchmark_grid_femnist` (21). All three share one `experiment_group`, so
    `analysis.py` reads them as the single 105-run grid the manuscript describes.
14. `--matrix uniform_memory_control` (6).

**42 + 42 + 21 + 36 + 6 = 147 confirmatory, plus the 30-run formulation study = 177
reported.** (Ablation dropped from 42 to 36 on 2026-08-02 — Stage 1's noise-margin
verdict was an empty surviving set, so Configuration 8 has nothing left to remove;
Decisions 60, 79/80. `chapter_6.tex` §6.2's contribution bullet resting on that
contrast still needs withdrawing in the sibling manuscript repo.) The exploration
phase's own runs are outside that total: `pass2_explore`
4, `pass2_factorial` 26, `pass3_freeze_confirm` 8, `baseline_tuning` 55 — 93 runs at
the held-out α = 0.3. The conditional 6-run recheck at step 9 fired (the freeze is
Formulation 2, not 3) but is **not spent**: it compares the surviving refinement
layer against unrefined, and Decisions 79/80 froze that layer empty, so both arms
are byte-identical. It is a veto on the layer, never a second search, so it cannot
re-populate what the ablation arms inherit and does not gate them — Decision 85. The formulation study declares `phase: explore`, not `formal`: §4.3.1
makes it the culmination of the exploration phase, whose verdict is frozen and
tagged, so it necessarily precedes the grid it configures. `test_primary_grid_files_dispatch_all_105_runs`
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
  Prefer `--skip_completed` for recovery: it re-dispatches only runs missing a
  final-round `final_global_model.pt`, so a sweep that lost tasks 57 and 91 is repaired
  by one re-invocation with no index arithmetic. `--start_at N` still exists for
  deliberately resuming at a point (1-indexed, and the index is into that matrix's own
  task list — re-read the dry run before using it). Both are previewable: `--dry_run`
  labels each task it would skip and why.
- **Every sweep writes `sweep_status.json`** to its experiment-group directory
  (`outputs/<phase>/<dataset>_<model>/<exp_group>/`), rewritten after each task so it
  survives a sweep that never reaches its summary. It carries `failed_indices`, plus the
  label, exit code, and full command of each failure. Read it before deciding what to
  re-run; it is scoped to one invocation and replaced on the next.
- **Dry-run first**: `--dry_run` prints every composed command without executing. On a
  shared host this is the cheapest way to catch a wrong `experiment=` or output path.
