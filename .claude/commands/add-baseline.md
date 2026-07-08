---
description: scaffolds a new FL baseline implementation and updates registry
---

To implement a new SOTA baseline:

1. Create a Hydra configuration file under `conf/algorithm/{name}.yaml`.
2. Implement strategy logic, client training loops, or algorithm-specific components inside [strategy.py](../../src/fedmaq/core/strategy.py) and [client.py](../../src/fedmaq/core/client.py). Use [baselines/](../../src/fedmaq/baselines/) for standalone compression/quantization helper modules.
3. Integrate the baseline strategy selection into the main training runner in [run.py](../../scripts/run.py).
4. Update [.claude/project/baseline_registry.md](../project/baseline_registry.md) with the new algorithm details, setting status to `[In Progress]` or `[Complete]`.
5. Run `uv run pytest` to ensure correct initialization.
