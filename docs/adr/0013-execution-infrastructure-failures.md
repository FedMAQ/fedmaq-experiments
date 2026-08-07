# ADR-0013 — Sweep failures: measure the suspect before reverting it

**Status**: Accepted · 2026-08-01
**Supersedes**: `docs/DECISIONS.md` Decisions 77, 78 (file deleted; see ADR-0014)

## Context

The first FedMAQ factorial sweep died on its first three tasks with
`ValueError: Message contains an Error (... the destination SuperNode was removed
from the federation ...)` raised from partition-ID resolution, then a TTL error and
a timeout. The obvious suspect was `client_gpus: 0.5`, which had been flipped from
`1.0` across nine matrices days earlier and had never been exercised on a FedMAQ
sweep. The runbook made it more plausible still by crediting a *completed*
screening run to `0.5` — a false claim, since that run predated the flip by hours.

## Decision

### `client_gpus: 0.5` was measured, and is innocent

Two 5-round FedMAQ runs, back to back in one session so the shared host's co-tenant
load could not drift between them, differing only in `client_gpus`, with overrides
otherwise identical to the failing task. Both completed cleanly. Peak VRAM was
~5.0 GB of our own at 0.5 and ~3.0 GB at 1.0 on a 40 GB card against a co-tenant
holding ~11 GB, leaving ~24 GB spare. **There is no VRAM ceiling near where this
operates, and 0.5 was *faster*.**

**A revert of the pre-tag matrices to 1.0 was drafted and not taken** — it would
have fixed nothing and cost throughput. Every matrix stays at 0.5.

Two corrections fell out. The runbook's false attribution is fixed, with the
timestamps that settle it. And the original justification for the flip was measured
with **FedAvg**, not FedMAQ, which additionally carries a server-side grad-norm
probe model and KD: the end-to-end FedMAQ speed-up measured 1.16×, not the 1.88×
on record. **No schedule estimate should be re-derived from 1.88×.**

### The actual defect: partition-ID resolution had no retry

`_partition_id` caches by node ID, so all 100 `get_properties` round trips happen in
round 1 and every later round is a cache hit — and a cache hit cannot raise. A
50-round run takes ~1050 s, so a failure at 495 s is roughly round 20, long after
the cache was warm. The only way back into that code path mid-run is a node ID that
is not in the cache: a SuperNode that left the federation, exactly as the error text
said. **This is a membership change on a shared host, not a cold-start storm and
not resource exhaustion.** It met a single unguarded attempt with no retry.

**The fix is five attempts with linear backoff, then `PartitionResolutionError` —
and it aborts deliberately.** This is the opposite of its sibling in
`strategy_hooks/_partition.py`, which falls back to `hash(cid) % num_clients`. A
guessed ID here would not mislabel one client; it would change *which* partitions
are drawn, silently and differently each run, destroying the one property
`SeededPartitionClientManager` exists to provide (ADR-0006) and with it the paired
methodology of ADR-0004. Dropping the proxy and sampling 99 is no better: that is
not the protocol the manuscript's table states, and nothing downstream would record
the deviation.

**Aborting costs one run, recoverable with `--skip_completed`. A silently
non-reproducible run costs the reproducibility claim and cannot be detected
afterwards.**

*What this does not establish:* the probes never reproduced the failure at either
setting — 5 rounds cannot, since the trigger is a mid-run membership change. The
measurement exonerates `client_gpus`; it does not validate the fix.

### Teardown was the cascade, not the trigger

The re-dispatched sweep aborted again, but not from the same trigger. One task hung
silently and was killed by the timeout (cause unknown; a shared-host stall is the
likeliest explanation and is not diagnosable from those logs). The *next* task
failed within 30 s because Ray's GCS server never started —
`kill_ray_processes()` ran `ray stop` first, as always, but **on Linux `ray stop`
was the only teardown step**, while the unconditional force-kill that already
existed for Windows had no Linux counterpart. A raylet or GCS wedged by the
preceding hard timeout-kill (the driver dies; Ray's grandchildren do not) survives
a polite `ray stop` and blocks the next task's `ray.init()`.

`kill_ray_processes()` now force-kills `raylet`, `gcs_server` and `plasma_store`
(`pkill -9 -f`) on non-Windows, unconditionally, mirroring the Windows branch.
**This hardens the cascade — one bad task no longer contaminates the next — not the
hang itself**, which may be an irreducible shared-host event.

**Explicitly not done:** `max_consecutive_failures` was not raised. The abort is
working as designed; the bug was that unrelated failures were chaining into it, not
that the threshold is wrong.

### Pre-registered, before the re-dispatch

If aborts recur — node loss being routine on this shared host rather than rare —
the response is **tolerate-and-record**: drop the unresolvable client, proceed, and
write the deviation into that run's `experiment_log.csv` so affected rounds are
auditable and analysis can flag them. It is not a silent drop, and it was not
adopted then, because a fix adopted before the failure rate is known would trade
the reproducibility guarantee for a problem that may not exist. It has not fired.

## Consequences

- **Measure the suspect against a same-session control before reverting it.** Both
  the `client_gpus` exoneration and the false runbook attribution turned on
  timestamps and a paired measurement, not on plausibility.
- Resume on an artifact, not an index: `--skip_completed` keys on each run's
  final-round checkpoint, which is correct regardless of *which* tasks died.
- A partition-ID abort is a correct outcome, not a bug to suppress. Re-dispatch the
  lost run; do not add a fallback that guesses.
