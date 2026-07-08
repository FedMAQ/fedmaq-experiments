# Ablation Phases

The ablation study progresses sequentially through the following four research phases. These represent conceptual and experimental milestones (configured dynamically via Hydra under `conf/`), not separate python packages:

| Phase | Focus               | Key Objectives                                                                                                    |
| ----- | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1     | FL Environment      | PyTorch/Flower env, vision datasets, Dirichlet non-IID, bandwidth stragglers (`core/`)                            |
| 2     | Uplink Quantization | Isolate resource/data/state-aware (gradient norm) uplink quantization vs static baselines (`baselines/`, `core/`) |
| 3     | Server-Side KD      | Server-side logit/feature KD to recover accuracy from quantization noise (`baselines/`, `core/`)                  |
| 4     | Benchmarking        | Unified FedMAQ vs SOTA across datasets (`scripts/run.py` & configs)                                               |

Do not skip phases when adding features. Maintain common components in `core/` and comparator algorithms in `baselines/`. Do not create separate top-level folders for different phases.
