---
name: user-run-matrix
description: Prepare and interpret a declarative FedMAQ matrix sweep that the user dispatches.
disable-model-invocation: true
---

# User-Run Matrix

Use this explicitly invoked skill when the user wants to dispatch a matrix
experiment or inspect its returned run evidence. The agent prepares commands and
interprets results; the user runs every experiment on the intended checkout.

## Prepare

1. Read the relevant execution Issue and any dated assurance envelope it names,
   then the current matrix file under `conf/matrix/`. Confirm the intended
   checkout, exact commit, stage, and output group with the user. When the Issue
   names an already sealed exact candidate, dispatch it detached; a later
   documentation-only `main` commit does not replace that candidate.
2. Emit a portable dry-run command before the dispatch command. Use the current runner:

       uv run python scripts/run_matrix.py --matrix <name> --dry_run

   Then provide the live dispatch command wrapped with `bash ./scripts/notify_run.sh`
   (e.g., `bash ./scripts/notify_run.sh ./.venv/bin/python scripts/run_matrix.py --matrix <name> --run_timeout_seconds 7200 ...`)
   only after the user has reviewed the plan. Read [the execution model](../../../docs/agents/execution-model.md)
   for the checkout interpreter and allocation-specific Ray overrides; use the
   checkout's documented interpreter when `uv` is unavailable.
3. Use `--skip_completed` for artifact-based recovery, `--shard I/N` for
   deterministic multi-host dispatch, and `--only LABEL` only for a deliberate
   label-scoped run. The runner rejects `--only` with `--shard`.
4. Keep command logs and returned evidence outside the checkout. Do not edit
   source, configuration, frozen artifacts, or result files to make a run fit.

## Interpret returned evidence

- Read the matrix group's `sweep_status.json`; for sharded work, collect every
  `sweep_status.shard-I-of-N.json` and use `scripts/merge_sweep_status.py` only
  after all shards are present.
- Report the matrix, commit, status state, completed/skipped/failed counts,
  failed indices, and any abort reason. A command being prepared is not a run.
- Route dead-run handling to `sweep-recovery`; route exact-commit golden capture
  and comparison to `jupyterhub-golden-gate`.

Completion requires a reviewed dry-run plan, user-returned status/evidence, and
an explicit completed, failed, or blocked disposition. The agent never declares
an experiment dispatched from command preparation alone.
