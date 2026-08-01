# Baseline Registry

Maps each algorithm target to its configuration. **Status** column: see
[docs/audits/baseline-status-audit.md](../../docs/audits/archive/baseline-status-audit.md)
for the reasoning behind each verdict (🟢 ready · 🟡 config-ready but unmeasured on
MobileNetV2GN · 🟠 needs attention before formal grid · 🔴 broken · ⚫ dropped).

| Algorithm  | Group                 | Paper / Citation                  | Config                                                                                                         | Status |
| ---------- | --------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- | :----: |
| FedAvg     | Seminal Controls      | McMahan et al., 2017              | [fedavg.yaml](../../conf/algorithm/fedavg.yaml)         |   🟢   |
| FedProx    | Seminal Controls      | Li et al., 2020                   | [fedprox.yaml](../../conf/algorithm/fedprox.yaml)       |   🟠   |
| FedPAQ     | Pure Quantization     | Reisizadeh et al., 2020           | [fedpaq.yaml](../../conf/algorithm/fedpaq.yaml)         |   🟢   |
| DAdaQuant  | Pure Quantization     | Hönig et al., 2022                | [dadaquant.yaml](../../conf/algorithm/dadaquant.yaml)   |   🟢   |
| ~~FedMD~~  | Pure KD (dropped)     | Li et al., 2019                   | [fedmd.yaml](../../conf/algorithm/fedmd.yaml)           |   ⚫   |
| FedDistill | Pure KD               | Jeong et al. (FedGen/FEDDISTILL+) | [feddistill.yaml](../../conf/algorithm/feddistill.yaml) |   🟡   |
| FedKD      | Hybrid Q+KD           | Wu et al., 2022                   | [fedkd.yaml](../../conf/algorithm/fedkd.yaml)           |   🟠   |
| ~~CFD~~    | Hybrid Q+KD (dropped) | Sattler et al., 2022              | [cfd.yaml](../../conf/algorithm/cfd.yaml)               |   ⚫   |
| FedMAQ     | Proposed SOTA         | Bunyi et al., 2026                | [fedmaq.yaml](../../conf/algorithm/fedmaq.yaml)         |   🟢   |

**FedMD dropped** from the formal baseline stack (8 → 7) — infeasible pretrain
cost (see `docs/DECISIONS.md` Decision 25). Config/hook code retained for
reproducibility, excluded from all sweeps.

**CFD dropped** from the baseline stack (7 -> 6) - collapses to chance
accuracy at production client-count scale; per-client partitions too small for
CFD's 1-bit hard-vote protocol at 100 clients (see `docs/DECISIONS.md`
Decision 26). Config/hook code retained for reproducibility, excluded from
all sweeps.
