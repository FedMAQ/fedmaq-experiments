# FedMAQ Experiment Registry & Roadmap

This directory houses all experimental sweeps, ablation studies, and hyperparameter tuning results conducted for the FedMAQ thesis.

Every experiment is self-contained within its own directory and adheres to a strict organization standard:

- `results.md`: Detailed tabular data (accuracy, loss, communication footprint, simulated latency) and Hydra configuration paths.
- `comments.md`: In-depth empirical analysis, physical mechanisms, and Master's thesis narrative alignment.

> [!IMPORTANT]
> **Archived: ResNet18GN-era smoke tests (July 13–15, 2026).** Nine exploratory smoke-test experiments (40–50 round, single-seed) were run on ResNet18GN to validate the algorithm direction. They are **deprecated** following the switch to MobileNetV2GN as the iso-architecture (see [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)). Directory index below.

## Archived (ResNet18GN, deprecated)

|  #  | Experiment           | Directory                                                                                                                                                    |
| :-: | :------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1  | Baseline Smoke Test  | [archive/smoke-test-7-13/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/smoke-test-7-13/)                           |
|  2  | Formulation Study    | [archive/pilot-formulation-study-7-14/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/pilot-formulation-study-7-14/) |
|  3  | EMA Decay Sweep      | [archive/ema-decay-sweep-7-14/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/ema-decay-sweep-7-14/)                 |
|  4  | Soft-Voting Sweep    | [archive/soft-voting-sweep-7-14/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/soft-voting-sweep-7-14/)             |
|  5  | Temperature Ablation | [archive/temperature-ablation/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/temperature-ablation/)                 |
|  6  | ResNet18GN Baselines | [archive/baseline-comparison-resnet18/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/baseline-comparison-resnet18/) |
|  7  | Client KD Reg Sweep  | [archive/client-kd-reg-sweep-7-15/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/client-kd-reg-sweep-7-15/)         |
|  8  | Stacked Reg Sweep    | [archive/stacked-reg-sweep-7-15/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/stacked-reg-sweep-7-15/)             |
|  9  | No-EMA (ResNet18GN)  | [archive/fedmaq-normal-no-ema-50r/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/fedmaq-normal-no-ema-50r/)         |

Consolidated historical accuracy standings and best-known configs for these runs live in [archive/RESNET18GN-SUMMARY.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/archive/RESNET18GN-SUMMARY.md).

## Current (MobileNetV2GN)

|  #  | Experiment                     | Directory                                                                                                                      | Description                                                                      |
| :-: | :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
|  1  | MobileNetV2GN Smoke Test (50R) | [mobilenetv2-smoke-50r/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/mobilenetv2-smoke-50r/) | 50-round sweeps of FedAvg, FedProx, and FedMAQ across $\alpha \in \{0.1, 1.0\}$. |

New experiments land as top-level dirs in this directory following the same `results.md` / `comments.md` structure. See [docs/plans/formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md) for the exploration/confirmation pipeline.

## Run Execution & Context

- **Process-Isolated Execution**: All experiments use process-isolated runner scripts under the `scripts/` directory to prevent CUDA Out-of-Memory (OOM) leaks from sequential multi-runs.
- **Hardware Grounding**: Simulates edge client memory sizes matching **Raspberry Pi variants (2GB/4GB/8GB)** and **Jetson Edge Nodes (16GB)** capping quantization bit-widths (4/8/16/32-bit).
- **Data Paths**: Runner scripts output to `experiments/` by default; data is manually transferred to `multirun/` after completion. Paths referencing `experiments/` in older documents may be outdated.
