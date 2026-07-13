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
| CFD        | Hybrid Q+KD       | Sattler et al., 2022    | `conf/algorithm/cfd.yaml`        | [Complete]    | Root-caused accuracy plateau to unshuffled public loader sequence causing catastrophic server-model forgetting (gradient updates chasing homogeneous class batches in sequence). Fixed by joint shuffling of images and targets in server distillation. Verified on 8-round CIFAR-10 simulation (alpha=1.0) with accuracy climbing from 10% to 25.56% in round 7 (targets_row_std healthy at ~0.11-0.13), resolving the issue. |
| FedMAQ     | Proposed SOTA     | Bunyi et al., 2026      | `conf/algorithm/fedmaq.yaml`     | [Complete]    | Verified 2-round local dry run simulation. |

**Status:** `[Not Started]` | `[In Progress]` | `[Complete]`

**Caveat (2026-07-13):** "Verified 2-round local dry run simulation" only confirms the job runs without crashing -- it is not evidence of algorithmic correctness. FedKD passed this exact bar while its SVD compression was silently pinned at chance-level accuracy (see FedKD row). Do not treat `[Complete]` + dry-run note as a correctness signal for any baseline until it has been re-checked under the `run-minitest` sweep (all baselines, multi-round, post-CFD-fix) planned next.
