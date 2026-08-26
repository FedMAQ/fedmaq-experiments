# Engineering

## Hydra config

- Root `conf/config.yaml` composes the `dataset/`, `heterogeneity/`, `algorithm/`
  and `experiment/` groups. Add a new algorithm as `conf/algorithm/<name>.yaml`,
  never inline in the root config.
- Override from the CLI: `python scripts/run.py dataset=cifar10 heterogeneity.alpha=0.5`
- Outputs go to `outputs/` (single run) or `multirun/` (sweeps).

**Never use Hydra's `--multirun` flag** for multiple sequential federated jobs.
PyTorch GPU memory caching and Ray actor accumulation inside one parent process
cause CUDA OOM. Use the process-isolated runners in `scripts/` — for sweeps,
`scripts/run_matrix.py --matrix <name>`.

**Nothing downstream of the `pre-registration` tag may edit a frozen config.**
Thirteen files under `conf/` are frozen; treat them as read-only.

## Flower patterns

- Separate client app, server app and strategy modules per baseline or phase.
- Keep dataset loading and model definitions out of strategy classes.
- Use Flower's `ClientApp` / `ServerApp` patterns for simulation.
- Hyperparameters come from Hydra configs, never hardcoded in Python.

## Sweep recovery

- Read `.agents/skills/sweep-recovery/SKILL.md` when a sweep run dies, a
  `PartitionResolutionError` aborts, or a Flower+Ray sim crashes (raylet
  `SIGSEGV`, `SYSTEM_ERROR`, actor deaths).
