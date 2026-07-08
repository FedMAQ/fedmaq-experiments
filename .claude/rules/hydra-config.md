# Hydra Config

- Root config: `conf/config.yaml` composes `dataset/`, `heterogeneity/`, `algorithm/`, `experiment/` groups.
- Outputs go to `outputs/` (single run) or `multirun/` (sweeps) — both gitignored.
- Override from CLI: `python scripts/run.py dataset=cifar10 heterogeneity.alpha=0.5`
- Add new algorithms as `conf/algorithm/<name>.yaml`, not inline in root config.
