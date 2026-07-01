# Baseline Registry

Maps each SOTA target to implementation status. Update when porting into `src/fedmaq/baselines/`.

| Algorithm  | Group             | Paper / Citation        | Config                           | Status        | Notes                                      |
| ---------- | ----------------- | ----------------------- | -------------------------------- | ------------- | ------------------------------------------ |
| FedAvg     | Seminal Controls  | McMahan et al., 2017    | conf/algorithm/fedavg.yaml       | [Complete]    | Verified 2-round local dry run simulation. |
| FedProx    | Seminal Controls  | Li et al., 2020         | conf/algorithm/fedprox.yaml      | [Complete]    | Verified 2-round local dry run simulation. |
| FedPAQ     | Pure Quantization | Reisizadeh et al., 2020 | conf/algorithm/fedpaq.yaml       | [Complete]    | Verified 2-round local dry run simulation. |
| DAdaQuant  | Pure Quantization | Hönig et al., 2022      | conf/algorithm/dadaquant.yaml    | [Complete]    | Verified 2-round local dry run simulation. |
| FedMD      | Pure KD           | Li et al., 2019         | `conf/algorithm/fedmd.yaml`      | [Complete]    | Verified 2-round local dry run simulation. |
| FedDistill | Pure KD           | Song et al., 2024       | `conf/algorithm/feddistill.yaml` | [Not Started] |                                            |
| FedKD      | Hybrid Q+KD       | Wu et al., 2022         | `conf/algorithm/fedkd.yaml`      | [Complete]    | Verified 2-round local dry run simulation. |
| CFD        | Hybrid Q+KD       | Sattler et al., 2022    | `conf/algorithm/cfd.yaml`        | [Not Started] |                                            |
| FedMAQ     | Proposed SOTA     | Bunyi et al., 2026      | `conf/algorithm/fedmaq.yaml`     | [Complete]    | Verified 2-round local dry run simulation. |

**Status:** `[Not Started]` | `[In Progress]` | `[Complete]`
