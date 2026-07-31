# Experiment Runbook

How the FedMAQ experiments are executed: where runs happen, in what order, and the
operational controls that keep a sweep recoverable. Durable operational reference —
not session context. Per-session orientation belongs in a temporary handoff file,
not in this repo.

**Last updated**: 2026-07-31

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

### Stage 1 — Exploration (gates everything downstream)

Runs at the held-out $\alpha = 0.3$, absent from the confirmatory grid by design, so
no mechanism is ever selected at a skew it is later reported on. Not counted among
the 183.

1. `--matrix pass2_explore` — screening, R=50, one seed, 4 runs. **Done 2026-07-31**
   (~70 min at `client_gpus: 0.5`, 4/4 completed).
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
5. **Freeze.** Write the surviving set into `conf/algorithm/fedmaq.yaml` — **and only
   there.** The §4.3.7 arms inherit that file via their Hydra defaults list and
   restate only their own removal, so one edit reaches all of them (Decision 58). Then
   `uv run python scripts/dump_frozen_configs.py` to refresh
   `docs/freeze/resolved_configs.yaml` (the arms are no longer self-describing, and
   that generated snapshot is what makes the tagged state readable — Decision 59), run
   `uv run python -m pytest tests/test_simulation.py`, and git-tag. Manuscript §6.2
   promises that tag.
   **If nothing clears at R=100** the surviving set is empty — pre-registered, not a
   judgement call (Decision 60, manuscript §4.3.1 and §4.3.7). FedMAQ freezes
   unrefined, Configuration 8 drops from `conf/matrix/ablation.yaml`, and the
   `chapter_6.tex` contribution bullet resting on its contrast goes with it. No subset
   retries, no tuning to rescue a mechanism.
   `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` enforces it.

### Stage 2 — Formulation study (30 runs)

6. `--matrix formulation_study`. Must carry Stage 1's surviving layer: its
   Formulation 1 cell is Ablation Configuration 4's parity anchor, and an anchor only
   anchors if it carries the same refinement layer as the arm.

### Stage 3 — Ablation (42 runs)

7. **Before dispatch**, if Formulation 1 or 2 won Stage 2, revisit `fedmaq_no_data`
   and `fedmaq_no_state` plus `ABLATION_ARM_DIFFS` — `fedmaq_no_data`'s removal
   becomes `gamma2=0`, and `fedmaq_no_state` drops its `formulation` override and
   stops being the fallback arm. Each is now a one-line change in a file that
   contains only its removal. The arms are currently pinned to Formulation 3 through
   `fedmaq.yaml`. Re-run `scripts/dump_frozen_configs.py` afterwards.
8. `--matrix ablation`.

### Stage 4 — Primary grid (105 runs) and control arm (6 runs)

The six baselines here are independent of Stages 1–3 and may run at any time;
FedMAQ's own rows need the freeze.

9. `--matrix benchmark_grid` (CIFAR-10, 42), `--matrix benchmark_grid_cifar100` (42),
   `--matrix benchmark_grid_femnist` (21). All three share one `experiment_group`, so
   `analysis.py` reads them as the single 105-run grid the manuscript describes.
10. `--matrix uniform_memory_control` (6).

**42 + 42 + 21 + 42 + 6 = 153 confirmatory, plus the 30-run formulation study = 183
reported.** The formulation study declares `phase: explore`, not `formal`: §4.3.1
makes it the culmination of the exploration phase, whose verdict is frozen and
tagged, so it necessarily precedes the grid it configures. The headline total is
unchanged; only the labelling is. `test_primary_grid_files_dispatch_all_105_runs`
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
