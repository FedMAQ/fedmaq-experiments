# Baseline Registry

Maps each SOTA target to implementation status. Update when porting into `src/fedmaq/baselines/`.

| Algorithm | Group              | Paper / Citation     | Config                          | Status        | Notes |
| --------- | ------------------ | -------------------- | ------------------------------- | ------------- | ----- |
| FedAvg    | Vanilla            | McMahan et al., 2017 | `conf/algorithm/fedavg.yaml`    | [Not Started] |       |
| FedProx   | Vanilla            | Li et al., 2020      | `conf/algorithm/fedprox.yaml`   | [Not Started] |       |
| DAdaQuant | Static compression | Hönig et al., 2022   | `conf/algorithm/dadaquant.yaml` | [Not Started] |       |
| LAQ-HC    | Adaptive           | Cui et al., 2026     | `conf/algorithm/laq_hc.yaml`    | [Not Started] |       |
| FedMD     | Pure KD            | Li et al., 2019      | `conf/algorithm/fedmd.yaml`     | [Not Started] |       |
| FedKD     | Pure KD            | Wu et al., 2022      | `conf/algorithm/fedkd.yaml`     | [Not Started] |       |
| DynFed    | SOTA Q+KD          | He et al., 2025      | `conf/algorithm/dynfed.yaml`    | [Not Started] |       |
| FedDT     | SOTA Q+KD          | He et al., 2025      | `conf/algorithm/feddt.yaml`     | [Not Started] |       |
| FedMAQ    | Proposed SOTA      |                      | `conf/algorithm/fedmaq.yaml`    | [Not Started] |       |

**Status:** `[Not Started]` | `[In Progress]` | `[Complete]`
