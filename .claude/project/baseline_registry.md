# Baseline Registry

Maps each SOTA target to implementation status. Update when porting into `src/fedmaq/baselines/`.

| Algorithm  | Group             | Paper / Citation        | Config                           | Status        | Notes                                      |
| ---------- | ----------------- | ------------------------ | --------------------------------- | -------------- | ------------------------------------------- |
| FedAvg     | Seminal Controls  | McMahan et al., 2017    | conf/algorithm/fedavg.yaml       | [Complete]    | Verified 2-round local dry run simulation. |
| FedProx    | Seminal Controls  | Li et al., 2020         | conf/algorithm/fedprox.yaml      | [Complete]    | Verified 2-round local dry run simulation. |
| FedPAQ     | Pure Quantization | Reisizadeh et al., 2020 | conf/algorithm/fedpaq.yaml       | [Complete]    | Verified 2-round local dry run simulation. |
| DAdaQuant  | Pure Quantization | Hönig et al., 2022      | conf/algorithm/dadaquant.yaml    | [Complete]    | Verified 2-round local dry run simulation. |
| FedMD      | Pure KD           | Li et al., 2019         | `conf/algorithm/fedmd.yaml`      | [Complete]    | Verified 2-round local dry run simulation. |
| FedDistill | Pure KD           | Jeong et al. (FedGen/FEDDISTILL+) | `conf/algorithm/feddistill.yaml` | [Complete]    | FEDDISTILL+ variant: shares FedAvg weights and label-wise logits. |
| FedKD      | Hybrid Q+KD       | Wu et al., 2022         | `conf/algorithm/fedkd.yaml`      | [Complete]    | Server-side SVD compression fixed 2026-07-13: was truncating full aggregated weight matrices (collapsed to rank-1, pinned accuracy at chance level); now compresses the delta against a tracked client-side reference, matching the upload path and the paper's gradient/update-compression design. 20-round CIFAR-10 verification: alpha=1.0 reached 24.19% acc / loss 2.017 (from chance-level 10%/ln(10)); alpha=0.1 reached 14.72% acc, noisier. Needs a longer run (40-100 rounds) to confirm full convergence, especially under heterogeneity. |
| CFD        | Hybrid Q+KD       | Sattler et al., 2022    | `conf/algorithm/cfd.yaml`        | [Complete]    | Paper-faithful port (no reference implementation to diff against, tested at elevated rigor). Implementation detail in `changelog.md`. |
| FedMAQ     | Proposed SOTA     | Bunyi et al., 2026      | `conf/algorithm/fedmaq.yaml`     | [Complete]    | Verified 2-round local dry run simulation. |

**Status:** `[Not Started]` | `[In Progress]` | `[Complete]`
