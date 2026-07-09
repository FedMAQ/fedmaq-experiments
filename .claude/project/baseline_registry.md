# Baseline Registry

Maps each SOTA target to implementation status. Update when porting into `src/fedmaq/baselines/`.

| Algorithm  | Group             | Paper / Citation        | Config                           | Status        | Notes                                      |
| ---------- | ----------------- | ------------------------ | --------------------------------- | -------------- | ------------------------------------------- |
| FedAvg     | Seminal Controls  | McMahan et al., 2017    | conf/algorithm/fedavg.yaml       | [Complete]    | Verified 2-round local dry run simulation. |
| FedProx    | Seminal Controls  | Li et al., 2020         | conf/algorithm/fedprox.yaml      | [Complete]    | Verified 2-round local dry run simulation. |
| FedPAQ     | Pure Quantization | Reisizadeh et al., 2020 | conf/algorithm/fedpaq.yaml       | [Complete]    | Verified 2-round local dry run simulation. |
| DAdaQuant  | Pure Quantization | Hönig et al., 2022      | conf/algorithm/dadaquant.yaml    | [Complete]    | Verified 2-round local dry run simulation. |
| FedMD      | Pure KD           | Li et al., 2019         | `conf/algorithm/fedmd.yaml`      | [Complete]    | Verified 2-round local dry run simulation. |
| FedDistill | Pure KD           | Jeong et al. (FedGen/FEDDISTILL+) | `conf/algorithm/feddistill.yaml` | [Complete]    | FEDDISTILL+ variant (shares FedAvg weights AND label-wise logits). Client `FedDistillFit` (`core/client_hooks/feddistill.py`) tracks per-class mean logits (LogitTracker, counts init to 1 to stay finite under non-IID) and trains with `CE + reg_alpha*KLDiv(log_softmax(z), softmax(global_logits[y]))`; server `FedDistillHook` averages client logit matrices (weights still FedAvg-aggregated) and rebroadcasts. Logits travel as bytes via FitRes.metrics / FitIns.config. Deviation: per-round tracker (Flower recreates clients each round), not the reference's cross-round cumulative sum. Verified via unit tests + a 2-round `run(cfg)` smoke exercising the reg path. Ref: `references/feddistill/` (FedGen). |
| FedKD      | Hybrid Q+KD       | Wu et al., 2022         | `conf/algorithm/fedkd.yaml`      | [Complete]    | Verified 2-round local dry run simulation. |
| CFD        | Hybrid Q+KD       | Sattler et al., 2022    | `conf/algorithm/cfd.yaml`        | [Not Started] | No `references/cfd/` source yet — needs a reference implementation sourced before porting. |
| FedMAQ     | Proposed SOTA     | Bunyi et al., 2026      | `conf/algorithm/fedmaq.yaml`     | [Complete]    | Verified 2-round local dry run simulation. |

**Status:** `[Not Started]` | `[In Progress]` | `[Complete]`
