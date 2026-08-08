# Execution Model and Dispatch Order

How FedMAQ experiments are executed: where runs happen, in what order, and the
operational controls that keep a sweep recoverable.

**This file is durable reference, not state.** It carries no run counts and no
"done/pending" markers — those move, and every one of them lives in the pinned
dispatch-state Issue. If you need to know what has actually run, read that Issue or
ask for the relevant matrix's `sweep_status.json`. Do not add status here.

> **Provenance.** This was `docs/RUNBOOK.md` until 2026-08-07. It moved and was
> rewritten in the same commit, so `git log --follow` does not traverse the rename —
> the prior history is at `75df164^:docs/RUNBOOK.md`.

---

## Execution model

**Read this before the dispatch order below. An agent cannot run any of it.**

- **Every reported run executes on the Linux datacenter allocation, reached through a
  JupyterHub gateway only** — no SSH, no VM, no shell an agent can drive. Hardware and
  software are specified in manuscript §4.3.4 (A100 40GB PCIe, dual Xeon Platinum
  8276, 64 GB RAM, shared with co-tenants). **That section is canonical; do not
  restate it in an ADR or a second registry.**
- **Sweeps are a hand-off.** The agent emits paste-ready commands, the user runs them
  in JupyterHub, the user pastes results back. Never write a plan step in which an
  agent dispatches a sweep or polls for its completion.
- **`outputs/` being empty locally proves nothing.** Results live on the allocation.
  Do not infer which stages have run from the local filesystem.
- **Experiments are not blocked on hardware.** The allocation exists and is in use.
  What gates the remaining stages is dispatch order, not availability.
- **The local Windows rig is for smoke tests only** — `run-minitest`, `--dry_run`,
  pre-dispatch validation, config-composition checks. It is the validated fallback of
  §4.3.4, not where the grid runs. The Ray crash mitigations in
  `.agent/rules/engineering.md` are scoped to it and apply nowhere else.

---

## Dispatch order

**The order below is load-bearing, not a convenience.** Manuscript §4.5 states it,
and running out of order costs runs rather than just time. Each matrix file's own
header carries the reason it sits where it does; read it before dispatching that
stage. **Per the execution model above, "dispatch" means handing the command to the
user.**

### Every dispatch on the allocation carries the Ray host flags

```
-o ray.temp_dir=/tmp/ray-cjb -o ray.object_store_gb=4
```

Not optional. Both default to `null`, which reproduces Ray's stock behaviour — and
stock behaviour is wrong on this host in two ways `conf/config.yaml` already
documents. Ray sizes its object store at ~30% of *total* RAM, so ~19 GB on a 64 GB
box that has ~39 GB actually free, on top of the sweep's own footprint; 4 GB is
ample, since the store only carries parameter payloads and the model is ~27 MB. And
the allocation is one shared VM with per-user accounts and a **shared `/tmp`**, so a
co-tenant's Ray session collides with ours at the default `/tmp/ray`. A co-tenant is
not hypothetical — `nvidia-smi` has shown multiple GB of VRAM held by a process this
account cannot see.

Keep the path short and outside `$HOME`: Ray builds Unix domain sockets beneath it
and those paths cap near 107 characters.

Neither flag is proven to have caused any past abort ([ADR-0013](../adr/0013-execution-infrastructure-failures.md)
records what was and was not established). Both are cheap, documented, and remove the
two known ways this host starves a sweep.

### Stage 1 — Exploration (gates everything downstream)

Runs at the held-out α = 0.3, absent from the confirmatory grid by design, so no
mechanism is ever selected at a skew it is later reported on. Not counted among the
reported runs.

1. `--matrix pass2_explore` — screening, R=50, one seed.
2. `--matrix pass2_factorial` — keep-or-drop, fully crossed 2³, three seeds; **the
   unrefined reference cell carries five seeds, not three**, because its spread *is*
   the margin every other cell is judged against and at n=3 that estimate is uncertain
   by roughly a factor of twelve. Use `--run_timeout_seconds 2100`. Then
   `scripts/analysis.py:exploration_noise_margin` writes
   `scripts/analysis_output/exploration_margin.json`.
   **It takes an `experiment_group` and refuses to pool stages.** It cannot produce a
   margin from `pass2_explore` (one seed, by design).
3. **Edit `pass3_freeze_confirm.yaml`** — replace the placeholder overrides in the
   `fedmaq-surviving-set` arm with `surviving_refinement_set` from that JSON. The
   values shipped there are defaults, not a prediction. That field is the smallest
   *cell* that cleared the margin, never a union across cells.
4. `--matrix pass3_freeze_confirm` — R=100. Use `--run_timeout_seconds 4200`.
   **Skip this stage if the surviving set is empty** — both arms would be
   config-identical to unrefined, so there is nothing to confirm. Go straight to
   step 5's empty-freeze branch.
5. **Freeze the refinement layer.** Write the surviving set into
   `conf/algorithm/fedmaq.yaml` — **and only there.** The ablation arms inherit that
   file via their Hydra defaults list and restate only their own removal, so one edit
   reaches all of them. Then `uv run python scripts/dump_frozen_configs.py` to refresh
   `docs/freeze/resolved_configs.yaml`, and run
   `uv run python -m pytest tests/test_simulation.py`.
   **Do not tag here.** §4.3.1 locks and tags three things together — mechanism set,
   selected formulation, baseline hyperparameter table — and two of them do not exist
   until step 9. The single tag is step 10.
   **If nothing clears at R=100** the surviving set is empty: pre-registered, not a
   judgement call. FedMAQ freezes unrefined, Configuration 8 drops from
   `conf/matrix/ablation.yaml`, and the `chapter_6.tex` contribution bullet resting on
   its contrast goes with it. **No subset retries, no tuning to rescue a mechanism.**
   `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` enforces it.
   See [ADR-0008](../adr/0008-exploration-protocol-and-the-empty-refinement-layer.md)
   and [ADR-0010](../adr/0010-freeze-machinery-and-pre-registration.md).

### Stage 1b — Baseline matched-tuning

Independent of everything above — no baseline shares configuration with FedMAQ — so
it may run concurrently with Stage 1. Held-out α = 0.3, uncounted among the reported
runs, and its verdict enters the same tag at step 10.

6. `--matrix baseline_tuning`. R=100: each of five baselines gets a five-seed
   reference cell at its shipped Table 4.1 value plus two three-seed challengers. Use
   `--run_timeout_seconds 4200`. Then `scripts/analysis.py:baseline_tuning_margin`,
   same √2σ rule as the factorial.
   **Not `exploration_noise_margin`** — that one filters `algorithm == "fedmaq"` and
   reports a completed Stage 1b as no runs at all.
   **Write any challenger that clears into `conf/algorithm/<baseline>.yaml` and into
   Table 4.1.** The expected outcome is that none clears and every baseline keeps its
   published value — **that is a result, not a null sweep**, and §4.3.2 reports it as
   one. FedAvg is absent by design: it is the uncompressed control and has no knob.
   See [ADR-0011](../adr/0011-baseline-matched-tuning.md).

### Stage 1c — The grid's FedAvg reference rows (not net-new)

The formulation study's accuracy floor is defined against the uncompressed FedAvg
reference at the same dataset and skew, *"reusing the FedAvg runs already present in
the benchmark grid"* (§4.3.6). Dispatched in file order those runs arrive at step 13
— after the freeze they are supposed to decide. The dependency is circular unless
FedAvg's rows are pulled forward, so they are.

7. `--matrix benchmark_grid --only fedavg`. CIFAR-10, α ∈ {0.1, 1.0}, 3 seeds.
   **These are rows of Stage 4's own matrix, not additional runs**, dispatched early
   under the same `experiment_group` and into the same directories; step 13's
   `--skip_completed` passes over them. `--only` refuses an unrecognized label rather
   than scheduling an empty sweep.
   They are the one confirmatory cell that runs before the step-10 tag. That is
   disclosed in §4.3.6 rather than finessed, and it is admissible because FedAvg is
   invariant to all three artifacts the tag locks. **Nothing else may move across the
   tag on this argument** — see [ADR-0009](../adr/0009-run-identity-and-analysis-scoping.md).

### Stage 2 — Formulation study

8. `--matrix formulation_study`. Must carry Stage 1's surviving layer: its
   Formulation 1 cell is Ablation Configuration 4's parity anchor, and an anchor only
   anchors if it carries the same refinement layer as the arm.
9. **Resolve the verdict, then write the formulation.** `select_winner` returns one
   verdict *per skew* — two, structurally, since the study runs both. Collapse them
   with `scripts/analysis.py:resolve_frozen_formulation`, which implements the
   pre-registered rule: skews agreeing freezes that formulation; skews diverging
   freezes the α = 0.1 winner and reports the split as a finding; one skew
   disqualifying its whole field defers to the other; both disqualifying falls back to
   highest mean top-1 at α = 0.1 **and withdraws the contribution claim**. Write the
   result to `conf/algorithm/fedmaq.yaml`.
   **If the frozen formulation is not the incumbent**, the reserved recheck fires —
   surviving layer vs. unrefined under the winning formulation at α = 0.3. It is a
   **veto on that layer, never a second search**: the factorial is not re-opened and
   no mechanism is reconsidered. It is degenerate if the layer is empty. Header of
   `conf/matrix/formulation_study.yaml` has the full rule; see
   [ADR-0012](../adr/0012-formulation-selection-and-the-iso-byte-amendment.md).

### Stage 2b — Tag the pre-registration

10. Re-run `scripts/dump_frozen_configs.py` and `pytest tests/test_simulation.py`,
    then **git-tag once.** The tag carries all three of §4.3.1's locked artifacts:
    the fixed mechanism set (step 5), the baseline hyperparameter table (step 6), and
    the selected formulation (step 9). Manuscript §6.2 promises this tag.
    **Nothing downstream of here may edit a frozen config**; an anomaly during
    confirmation opens a new labelled exploration round instead.

### Stage 3 — Ablation

11. **Before dispatch**, if Formulation 1 or 2 was frozen at step 9, revisit
    `fedmaq_no_data` and `fedmaq_no_state` plus `ABLATION_ARM_DIFFS` — the data-removal
    arm's removal becomes `gamma2=0`, and the state-removal arm drops its `formulation`
    override and stops being the fallback arm. Each is a one-line change in a file
    containing only its removal. Re-run `scripts/dump_frozen_configs.py` afterwards.
12. `--matrix ablation`.

### Stage 4 — Primary grid and control arm

The six baselines here need Stage 1b's verdict, not Stage 1's; FedMAQ's own rows need
the freeze.

13. `--matrix benchmark_grid --skip_completed` (CIFAR-10),
    `--matrix benchmark_grid_cifar100`, `--matrix benchmark_grid_femnist`. All three
    share one `experiment_group`, so `analysis.py` reads them as the single primary
    grid the manuscript describes.
14. `--matrix uniform_memory_control`.

The formulation study declares `phase: explore`, not `formal`: §4.3.1 makes it the
culmination of the exploration phase, whose verdict is frozen and tagged, so it
necessarily precedes the grid it configures. `test_primary_grid_files_dispatch_all_105_runs`
asserts the primary-grid share of the run arithmetic — **that test, not this file, is
where the count is pinned.**

---

## Key operational controls

- **Declarative matrix runner mandate.** Hydra `--multirun` causes CUDA VRAM leaks and
  lands runs in a date-keyed tree with no `experiment_group`. Always launch sweeps
  with `uv run python scripts/run_matrix.py --matrix <name>`. Every confirmatory run
  has a matrix file; if you find yourself hand-typing a `--multirun` for one, the file
  is missing and should be written instead.
- **`post_process` follows the comparison partner, not the algorithm.** ON for the
  three `benchmark_grid*` files and `uniform_memory_control`; OFF for
  `formulation_study` and every `ablation` arm. Both directions are enforced in
  `tests/test_simulation.py`. See [ADR-0004](../adr/0004-confirmatory-grid-design.md).
- **Prefer `--skip_completed` for recovery.** It re-dispatches only runs missing a
  final-round `final_global_model.pt`, so a sweep that lost tasks 57 and 91 is
  repaired by one re-invocation with no index arithmetic. `--start_at N` still exists
  for deliberately resuming at a point (1-indexed, into that matrix's own task list —
  re-read the dry run before using it). Both are previewable with `--dry_run`.
- **Every sweep writes `sweep_status.json`** to its experiment-group directory,
  rewritten after each task so it survives a sweep that never reaches its summary. It
  carries `failed_indices` plus the label, exit code and full command of each failure.
  Read it before deciding what to re-run; it is scoped to one invocation and replaced
  on the next.
- **Check system RAM headroom** before Flower simulations, not just VRAM.
- **Dry-run first.** On a shared host this is the cheapest way to catch a wrong
  `experiment=` or output path.
