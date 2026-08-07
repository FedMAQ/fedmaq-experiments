# ADR-0006 — Determinism is real, and the golden-diff gate is literal bit-exactness

**Status**: Accepted · 2026-07-16, confirmed 2026-07-23
**Supersedes**: `docs/DECISIONS.md` Decisions 18, 19, 40 (file deleted; see ADR-0014)

## Context

Two things depend on this repo being bit-reproducible. ADR-0004's paired-seed
methodology needs arms at a matched seed to draw *identical* partitions and
clients, or seed variance stops cancelling and ~3pp ablation deltas stop being
detectable at n = 3. And the architecture-deepening programme (ADR-0007) needed a
refactor gate strong enough to prove behaviour preservation across a dozen seams.

During Step 1 validation, re-running an identical seeded `ci_test` config appeared
to produce varying `test/loss` and `test/accuracy` between runs, suggesting
GPU/cuDNN non-determinism would block a bit-exact gate. Three fallback designs
(force determinism, CPU-only validation, tolerance-based comparison) were drafted
in response.

## Decision

### Determinism is closed in three halves

1. **Per-Ray-worker torch-flag re-pinning plus a seeded DataLoader** — training
   reproducibility inside each actor.
2. **`SeededPartitionClientManager`** — client sampling keyed by *partition ID*
   with a per-round RNG, so paired arms at a matched seed draw identical clients.
3. **Deterministic partitioning**, locked by
   `test_partition_seed_invariant_for_paired_arms`.

**`generate_partition_indices` is a pure function** of `(dataset, num_clients,
alpha, num_public_samples, seed, partition)` with **no algorithm input**. Its
single call site passes the *global* `cfg.experiment.num_public_samples`, so every
arm gets byte-identical partitions.

> **Footgun.** `num_public_samples` is sliced *before* Dirichlet advances the RNG.
> Divergent per-arm values would silently diverge partitions. This is safe only
> because it is one global config value — never make it per-algorithm.

### The golden-diff gate is literal bit-exactness; no fallback was needed

Before committing to any of the three fallback designs, the suspicion was retested
directly: `experiment=ci` was run twice each for `fedavg` and `fedmaq` — the exact
pair implicated — as independent processes, same seed, on GPU, and the resulting
`experiment_log.csv` files diffed. `test/loss`, `test/accuracy`, precision/recall/f1,
communication bytes, simulated `system/*_time` columns and FedMAQ-specific metrics
were **byte-identical** across both reruns for both algorithms. Only
`wall_time_sec` — real wall-clock, never a golden-diff candidate — differed.

The earlier observation was the artifact, not the platform. The existing
`strict_determinism` infrastructure (`set_seed` / `configure_torch_determinism`:
seeded Python/NumPy/torch/CUDA, `cudnn.deterministic=True`,
`use_deterministic_algorithms(True, warn_only=False)`, seeded DataLoader workers,
per-Ray-worker reseed in `client_fn`) was already in place at Step 1 and is
sufficient on its own; the Step 1 validation *comparison* was what misled.

**`scripts/golden_diff.py` therefore implements literal bit-exact comparison** —
old code vs. new code, same seed, all CSV columns except `wall_time_sec`. No
determinism-forcing work, no CPU-only fallback, no tolerance thresholds.

### A harness contamination class this gate exposes

`fedkd` and `fedmd` persist client/teacher state to disk keyed only by client ID,
not by run. The harness's capture and compare runs were silently inheriting each
other's trained weights instead of each starting cold, producing a 21-column
mismatch on a file the refactor had not touched. Fixed by wiping `PERSISTENCE_DIR`
at the top of every `_run()` call, both capture and compare. **A golden-diff
failure on an untouched baseline is a harness question first.**

## Consequences

- **If non-determinism appears to reappear, re-run the two-independent-runs diff
  first** to confirm it is real before reaching for a tolerance-based design. Do
  not assume GPU training is inherently non-reproducible in this codebase — it is
  not, and the one time it looked that way the measurement was wrong.
- Any change that reorders RNG-consuming operations breaks the gate by
  construction. `run_epochs` (ADR-0003) preserves the exact per-batch sequence and
  epoch/batch nesting order for precisely this reason, and several client hooks
  keep loss accumulators *external* to it so floating-point sums are not
  reassociated — see the comments in `cfd.py` and `fedmd.py`.
- The golden set is one config *per branch*, not per algorithm: `standard.py` is
  the shared `fit()` entry for six different branches, and a harness that only
  diffs plain FedAvg passes green while silently breaking FedProx or KD.
- FedMD is excluded from the default golden set on cost grounds (ADR-0005).
