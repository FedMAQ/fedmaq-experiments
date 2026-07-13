# Hydra Config

- Root config: `conf/config.yaml` composes `dataset/`, `heterogeneity/`, `algorithm/`, `experiment/` groups.
- Outputs go to `outputs/` (single run) or `multirun/` (sweeps).
- **Do NOT use Hydra's `--multirun` CLI flag directly** for running multiple sequential federated learning jobs. PyTorch GPU memory caching and Ray actor accumulation inside the same parent process cause CUDA Out-of-Memory (OOM) errors. Always use the process-isolated python runners in `scripts/` instead.
- Override from CLI: `python scripts/run.py dataset=cifar10 heterogeneity.alpha=0.5`
- Add new algorithms as `conf/algorithm/<name>.yaml`, not inline in root config.
