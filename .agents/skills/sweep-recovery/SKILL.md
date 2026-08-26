---
name: sweep-recovery
description: Recover from a dead sweep run or a Windows Ray/Flower crash.
disable-model-invocation: true
---

# Sweep Recovery

## Sweep resumption and failure handling

- **Resume on an artifact, not an index.** `run_matrix.py --skip_completed` keys
  on each run's final-round checkpoint, which is correct regardless of *which*
  tasks died; `--start_at N` assumes the survivors are a prefix.
- **Record failures as they happen** (`sweep_status.json`). An unattended sweep
  that dies before its summary otherwise leaves the failure list only in a log
  stream.
- A `PartitionResolutionError` abort is a **correct outcome**, not a bug to
  suppress. Re-dispatch the lost run; never add a fallback that guesses a
  partition ID ([ADR-0013](../../../docs/adr/0013-execution-infrastructure-failures.md)).

## Windows Ray crashes

**Scope: the local Windows fallback workstation only** — smoke tests,
`run-minitest`, and pre-dispatch validation. Reported runs execute on the Linux
allocation, where none of this applies. See
[docs/agents/execution-model.md](../../../docs/agents/execution-model.md).

If a Flower+Ray sim dies unexpectedly (raylet `SIGSEGV`, `SYSTEM_ERROR`, actor
deaths):

- **Check system RAM headroom first, not just VRAM.** `nvidia-smi` can show ample
  GPU headroom while the OS has only ~4 GB free, starving Ray/PyTorch init.
- Some crashes are a known Windows Ray instability class independent of RAM
  pressure — Flower's own docs recommend WSL2. Not always resource-explainable.
