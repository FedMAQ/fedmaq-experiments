---
name: sweep-recovery
description: Recover a failed or interrupted declarative matrix sweep from its status and checkpoints.
disable-model-invocation: true
---

# Sweep Recovery

Use this explicitly invoked skill after the user reports an interrupted matrix
run, a failed task, or a Windows Ray/Flower crash. Recovery commands are
prepared for the user; recovery is not inferred from a missing local output.

## Status-first recovery

- Locate the matrix group's current `sweep_status.json` and copy it plus the
  original logs outside the checkout before choosing a rerun. Read `state`,
  `failed_indices`, each failure command, and `abort_reason`; the next
  invocation rewrites the status file.
- Before any rerun, preserve each failed cell's run dir outside the group, then
  move that dir aside. A run refuses to append to an earlier attempt's
  `experiment_log.jsonl` or `v2_diagnostic.jsonl`, and the dry run marks those
  cells `WILL REFUSE`.
- If a sweep refuses because a lock is held, the named PID is a live sweep. Do not
  delete the lockfile, and do not start Ray cleanup around it.
- Re-run the same matrix with `--skip_completed` after a dry run. It keys off
  each run's final-round checkpoint and fills arbitrary gaps; it is the default
  recovery path. Wrap live allocation commands with `bash ./scripts/notify_run.sh`
  per [the execution model](../../../docs/agents/execution-model.md).
- Use `--start_at N` only when intentionally resuming the canonical list from
  index N after reviewing the dry-run plan. It assumes no completed task before
  N needs repair.
- For sharded work, repeat the same `--shard I/N` on the affected host and
  preserve each shard's status file. Run
  `scripts/merge_sweep_status.py --group-dir <group-dir>` only after every
  expected shard file is collected; it rejects overlap, gaps, and drift.

- A `PartitionResolutionError` is a correct fail-closed outcome. Re-dispatch the
  identified lost run after the user reviews its status; never guess a partition
  ID or add a fallback ([ADR-0013](../../../docs/adr/0013-execution-infrastructure-failures.md)).

## Windows crash branch

This branch applies only to the local Windows fallback workstation and its
smoke/pre-dispatch checks. Reported runs execute on the Linux allocation; see
[docs/agents/execution-model.md](../../../docs/agents/execution-model.md).

If a Flower/Ray simulation dies with a raylet `SIGSEGV`, `SYSTEM_ERROR`, or actor
death:

- Check system RAM headroom before VRAM; GPU headroom does not establish that Ray
  and PyTorch can initialize.
- Treat a repeatable Windows Ray failure as an environment branch and report it;
  do not change the experiment or silently accept partial output.

Completion requires the preserved status/log evidence, a reviewed dry-run
recovery plan, and a user-returned rerun disposition for every failed index.
