# Flower Patterns

- Separate **client app**, **server app**, and **strategy** modules per baseline or phase.
- Keep dataset loading and model definitions out of strategy classes.
- Use Flower's recommended `ClientApp` / `ServerApp` patterns for simulation.
- Hydra configs select algorithm and dataset; avoid hardcoding hyperparameters in Python.

## Windows Ray crash mitigation

If a Flower+Ray sim dies unexpectedly (raylet `SIGSEGV`, `SYSTEM_ERROR`, actor deaths):

- **Check system RAM headroom first**, not just GPU VRAM — `nvidia-smi` can show ample headroom while `Get-CimInstance Win32_OperatingSystem` shows only ~4GB free out of 16GB, starving Ray/PyTorch init. Want several GB free before launching.
- Some crashes are a known Windows Ray instability class independent of RAM pressure (Flower's own docs recommend WSL2) — not always resource-explainable.
- Multi-run scripts should support resumption plus a retry-with-full-Ray-teardown loop per run, rather than restarting a whole chain from scratch on failure. Resume on an artifact, not an index: `run_matrix.py --skip_completed` keys on each run's final-round checkpoint, which is correct regardless of *which* tasks died, whereas `--start_at N` assumes the survivors are a prefix. Record failures to a file as they happen (`sweep_status.json`) — an unattended sweep that dies before its summary otherwise leaves the failure list only in a log stream.
