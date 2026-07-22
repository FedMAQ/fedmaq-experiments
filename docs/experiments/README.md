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

|  #  | Experiment                     | Directory                                                                                                                                          | Description                                                                                                                                                                                                  |
| :-: | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1  | MobileNetV2GN Smoke Test (50R) | [mobilenetv2-smoke-50r/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/mobilenetv2-smoke-50r/)                     | 50-round sweeps of FedAvg, FedProx, FedMAQ, DAdaQuant, FedPAQ, and FedKD across $\alpha \in \{0.1, 1.0\}$.                                                                                                   |
|  2  | Soft-Voting Explore (Pass 1)   | [soft-voting-explore-mobilenetv2/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/soft-voting-explore-mobilenetv2/) | Priority 1 exploration Pass 1: `entropy_weight` × `precision_weight` sweep + soft-voting ablation, explore-α=0.3, 50R single-seed. Provisional pick ew=2.0/pw=0.5/sv_on, pending multi-seed re-verification. |

> [!NOTE]
> **Smoke-run caveats, resolved as of 2026-07-18** (see [docs/audits/distillation-direction-audit.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/audits/distillation-direction-audit.md)): FedKD's near-chance smoke result was a rank-starvation bug (F10), fixed and re-confirmed on a real 50R MobileNetV2GN run — FedKD is unblocked for comparison tables. F13 (KD-baseline coverage gap) closed 2026-07-17: FedDistill/FedAvg+KD ran clean; CFD collapsed to chance both α and was **dropped from the formal stack** (F15, structural — `docs/DECISIONS.md` Decision 26), same disposition as **FedMD** (infeasible pretrain cost, Decision 25). Formal baseline stack is now 6 + FedMAQ.

New experiments land as top-level dirs in this directory following the same `results.md` / `comments.md` structure. See [docs/plans/formal-experiment-plan.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/plans/formal-experiment-plan.md) for the exploration/confirmation pipeline.

## Run Execution & Declarative Matrix Runner

- **Declarative Matrix Runner**: All sweeps are defined in YAML manifests (`conf/matrix/*.yaml`) and executed using `uv run python scripts/run_matrix.py --matrix <name>`.
- **Process-Isolated Execution**: `scripts/run_matrix.py` enforces process isolation and calls `kill_ray_processes()` between runs to eliminate CUDA VRAM leaks and Ray worker accumulation.
- **Hardware Grounding**: Simulates edge client memory sizes matching **Raspberry Pi variants (2GB/4GB/8GB)** and **Jetson Edge Nodes (16GB)** capping quantization bit-widths (1–16 bits).
- **Canonical Output Hierarchy**: All raw experiment logs and Hydra configs land strictly in:
  `outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<heterogeneity>/seed_<seed>/`
  _(Phases: `ci` [2R], `smoke` [50R], `explore` [50R], `formal` [100R])._
