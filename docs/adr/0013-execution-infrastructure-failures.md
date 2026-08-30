# ADR-0013 — Sweep failures: measure the suspect before reverting it

**Status**: Accepted · 2026-08-01
**Supersedes**: `docs/DECISIONS.md` Decisions 77, 78 (file deleted; see ADR-0014)

## Context

The incident chronology, measurements, and recovery archaeology are retained in
the [issue-31 concurrency research record](../research/2026-08-27-issue-31-concurrency-mechanism.md).
This ADR keeps only the durable recovery decisions and their scope.

## Decision

### `client_gpus: 0.5` was measured, and is innocent

The paired control exonerated `client_gpus: 0.5`; the setting remains the matrix
default. The research record owns the measurements and the limits of that evidence.

### The actual defect: partition-ID resolution had no retry

Membership changes can invalidate a cached partition-ID lookup.

**The fix is five attempts with linear backoff, then `PartitionResolutionError` —
and it aborts deliberately.** A guessed ID or silently dropped client would change
the sampled partitions and violate the reproducibility contract. Aborting costs a
recoverable run; silently changing partitions costs reproducibility.

### Ray teardown isolates subsequent runs

After a hard timeout, surviving Ray processes can prevent the next task from
starting. `kill_ray_processes()` force-kills `raylet`, `gcs_server`, and
`plasma_store` on non-Windows as well as Windows. This isolates the cascade; it does
not claim to diagnose the original shared-host hang.

`max_consecutive_failures` remains unchanged; the abort threshold is not a recovery
knob for unrelated failures.

### Pre-registered contingency (not active)

The **tolerate-and-record** response was pre-registered as a contingency, but it
was not adopted and has not fired. Current behavior remains fail-closed: after
five attempts with linear backoff, partition resolution raises
`PartitionResolutionError` and aborts. Re-dispatch the lost run; do not enable
tolerate-and-record until a separately assured implementation exists. Any future
tolerate-and-record implementation must drop the unresolvable client, continue,
and write the deviation to that run's `experiment_log.csv` for analysis.

## Consequences

- **Measure the suspect against a same-session control before reverting it.** Both
  the `client_gpus` exoneration and the false runbook attribution turned on
  timestamps and a paired measurement, not on plausibility.
- Resume on an artifact, not an index: `--skip_completed` keys on each run's
  final-round checkpoint, which is correct regardless of *which* tasks died.
- A partition-ID abort is a correct outcome, not a bug to suppress. Re-dispatch the
  lost run; do not add a fallback that guesses.
