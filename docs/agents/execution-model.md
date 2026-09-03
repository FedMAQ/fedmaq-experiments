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
- **The local Windows rig is for smoke tests only** — matrix-runner `--dry_run`,
  pre-dispatch validation, config-composition checks. It is the validated fallback of
  §4.3.4, not where the grid runs. The Ray crash mitigations in
  `.agents/rules/engineering.md` are scoped to it and apply nowhere else.

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

### Gate 0 — Current-code correctness

No allocation run precedes this gate.

1. Resolve the implementation and literature audits, run `just check`, and verify
   each expected-run manifest with its generator. `scripts/check_freeze.py --check`
   compares the machine-readable source certificate against the current tree and
   reports changed, missing, or newly discovered campaign sources. A method or
   accounting change after a prior bit-exact capture makes that capture historical
   evidence, even when the CSV schema is unchanged; the source certificate does not
   replace the GPU golden comparison.
2. On the GPU host, capture and compare the golden set for the exact commit that will
   be tagged. This is a user-run gate; an agent prepares the commands and waits for
   the returned evidence. Do not reuse a capture from before a quantizer or byte-axis
   correction.
3. Dry-run every matrix named below. Dry-run output is validation, not dispatch.
   A matrix still carrying an unresolved selection placeholder is the exception: the
   dispatch planner refuses it and `run_matrix.py --dry_run` exits 2, naming the run
   labels and the unresolved keys. That named exit **is** this step's pass signal for
   such a matrix — it is evidence the interlock is live, not a gate failure. Every
   other matrix must dry-run clean. A pre-selection matrix becomes dry-runnable only
   once the stage that resolves it has been run and its verdict written in — by
   construction, after this gate — or never, if its branch is retired rather than
   resolved, as `pass3_freeze_confirm` was under ADR-0008.

### Stage A — Widened matched tuning (145 validation cells)

4. `--matrix baseline_tuning_wide`, at held-out α = 0.3 and R=100. Each of the six
   tunable algorithms contributes five registered validation seeds per arm. FedAvg
   is absent because it has no tunable communication knob. FedMAQ varies
   `q_max ∈ {4, 6, 8, 16}` while `c_unit=1024` remains fixed.

5. Run `baseline_tuning_margin`, retain every five-point table and seed-level curve,
   and write only challengers that strictly clear `sqrt(2) * sigma` into the shipped
   configs. A highest point that does not clear is not adopted. `paper_default_variant`
   is provenance metadata and may be null; it is never substituted with the shipped
   reference.

These 145 cells are the replacement pipeline's matched-tuning stage. They must
finish before Stage 1a because their verdicts configure FedMAQ and the baseline
arms.

### Gate 1 — `pre-registration-stage1a`

6. Freeze the audited code, corrected byte instrument, widened-tuning verdicts,
   baseline table, `power_mean_design` matrix, expected 84-cell manifest, and Stage-1
   selection rule. Re-run `just check`, the expected-run generator in check mode, and
   the current-code golden compare, then tag that exact pushed commit
   `pre-registration-stage1a`.

No reported replacement cell may precede this tag. Earlier provisional tags do not
authorize this campaign.

### Stage 1a — Power-mean degree and structural controls (84 reported cells)

7. `--matrix power_mean_design`. It contains the seven degree settings at
   `omega=0.5`, the resource-only control, and six structural-rule settings across
   two skews and three seeds.
8. Run `scripts/select_power_mean.py`. It refuses an incomplete closure certificate
   and resolves skew disagreement through the severe-skew rule. Record the selected
   `p`; do not edit any Stage-1a row after observing the result.

### Stage 1b — Selected-p omega follow-up (12 reported cells)

9. Materialize the canonical selected-`p` follow-up for `omega ∈ {0.25, 0.75}` at
   both skews and three seeds. The already-run `omega=0.5` cells are reused, not
   repeated. The matrix and expected manifest must be committed and dry-run before
   dispatch; never hand-type twelve independent commands.
10. Dispatch the 12 cells, apply the same minimum-common-byte and severe-skew rule,
    and write the selected `(p, omega)` into the shipped FedMAQ configuration.

### Gate 2 — selected formulation and downstream manifest

11. Freeze the selected pair, resolved configs, ablation arm diffs, and the complete
    174-cell downstream manifest. Run `just check`, manifest checks, and the golden
    compare, then tag the exact pushed commit before any downstream cell runs.
    A material change after this gate opens a new labelled exploration amendment; it
    is not folded silently into the frozen campaign.

### Stage 2 — Downstream confirmation (174 reported cells)

12. `--matrix benchmark_grid`, `--matrix benchmark_grid_cifar100`, and
    `--matrix benchmark_grid_femnist`. The three files share one experiment group and
    contribute 105 cells.
13. `--matrix ablation`, contributing 36 net-new cells after the inherited FedAvg
    rows and dropped empty-refinement Configuration 8 are accounted for.
14. `--matrix uniform_memory_control`, contributing six cells, plus the registered
    FedPAQ pipeline and memory-sensitivity matrices.

The full scientific workload is therefore 415 cells: 145 matched-tuning cells,
96 formulation cells, and 174 downstream confirmation cells. Assurance executions
are recorded separately and are not part of this scientific total.

---

## Key operational controls

- **Declarative matrix runner mandate.** Hydra `--multirun` causes CUDA VRAM leaks and
  lands runs in a date-keyed tree with no `experiment_group`. Always launch sweeps
  with `./.venv/bin/python scripts/run_matrix.py --matrix <name>`. Every confirmatory run
  has a matrix file; if you find yourself hand-typing a `--multirun` for one, the file
  is missing and should be written instead.
- **`post_process` follows the comparison partner, not the algorithm.** ON for the
  three `benchmark_grid*` files and `uniform_memory_control`; OFF for
  `formulation_study` and every `ablation` arm. Both directions are enforced in
  `tests/test_config_and_dispatch.py`. See [ADR-0004](../adr/0004-confirmatory-grid-design.md).
- **Prefer `--skip_completed` for recovery.** It re-dispatches only runs missing a
  final-round `final_global_model.pt`, so a sweep that lost tasks 57 and 91 is
  repaired by one re-invocation with no index arithmetic. `--start_at N` still exists
  for deliberately resuming at a point (1-indexed, into that matrix's own task list —
  re-read the dry run before using it). Both are previewable with `--dry_run`.
- **Use `--shard I/N` for multi-host dispatch.** Sharding is a round-robin partition
  of the fully expanded canonical matrix list, independent of host, completion state,
  and wall clock. Every host must use the same matrix and `N`, with a distinct `I`:
  `./.venv/bin/python scripts/run_matrix.py --matrix benchmark_grid --shard 2/4`.
  The dry run prints canonical indices and the shard's status file is
  `sweep_status.shard-I-of-N.json`, so hosts do not race on one JSON file.
- **Merge shard status only after collection.** From the shared sweep-group directory,
  run `./.venv/bin/python scripts/merge_sweep_status.py --group-dir <group-dir>`.
  The merger refuses missing or overlapping canonical indices before writing the
  aggregate `sweep_status.json`. Each run manifest records its producing host and
  shard; these are provenance fields, not substitutes for the modeled execution
  telemetry required by Issue #28.
- **Every unsharded sweep writes `sweep_status.json`** to its experiment-group directory,
  rewritten after each task so it survives a sweep that never reaches its summary. It
  carries `failed_indices` plus the label, exit code and full command of each failure.
  Read it before deciding what to re-run; it is scoped to one invocation and replaced
  on the next.
- **Check system RAM headroom** before Flower simulations, not just VRAM.
- **Dry-run first.** On a shared host this is the cheapest way to catch a wrong
  `experiment=` or output path.
