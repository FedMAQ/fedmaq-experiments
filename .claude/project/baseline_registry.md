# Baseline Registry

Maps each algorithm target to its configuration. **Status** column: see
[baseline-status-audit.md](../../docs/audits/archive/baseline-status-audit.md)
(archived 2026-07-18; it is the source of every verdict below)
for the reasoning behind each verdict (🟢 ready · 🟡 config-ready but unmeasured on
MobileNetV2GN · 🟠 needs attention before formal grid · 🔴 broken · ⚫ dropped).

| Algorithm  | Group                 | Paper / Citation                  | Config                                                                                                         | Status |
| ---------- | --------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- | :----: |
| FedAvg     | Seminal Controls      | McMahan et al., 2017              | [fedavg.yaml](../../conf/algorithm/fedavg.yaml)         |   🟢   |
| FedProx    | Seminal Controls      | Li et al., 2020                   | [fedprox.yaml](../../conf/algorithm/fedprox.yaml)       |   🟢   |
| FedPAQ     | Pure Quantization     | Reisizadeh et al., 2020           | [fedpaq.yaml](../../conf/algorithm/fedpaq.yaml)         |   🟢   |
| DAdaQuant  | Pure Quantization     | Hönig et al., 2022                | [dadaquant.yaml](../../conf/algorithm/dadaquant.yaml)   |   🟢   |
| ~~FedMD~~  | Pure KD (dropped)     | Li et al., 2019                   | [fedmd.yaml](../../conf/algorithm/fedmd.yaml)           |   ⚫   |
| FedDistill | Pure KD               | Jeong et al. (FedGen/FEDDISTILL+) | [feddistill.yaml](../../conf/algorithm/feddistill.yaml) |   🟢   |
| FedKD      | Hybrid Q+KD           | Wu et al., 2022                   | [fedkd.yaml](../../conf/algorithm/fedkd.yaml)           |   🟡   |
| ~~CFD~~    | Hybrid Q+KD (dropped) | Sattler et al., 2022              | [cfd.yaml](../../conf/algorithm/cfd.yaml)               |   ⚫   |
| FedMAQ     | Proposed SOTA         | Bunyi et al., 2026                | [fedmaq.yaml](../../conf/algorithm/fedmaq.yaml)         |   🟢   |

**FedMD dropped** from the formal baseline stack (8 → 7) — infeasible pretrain
cost (see `docs/DECISIONS.md` Decision 25). Config/hook code retained for
reproducibility, excluded from all sweeps.

**Two configs moved after Stage 1b** (`baseline_tuning`, 55 runs, 2026-08-06,
Decision 81): FedProx `mu` 1.0 → 0.01 and FedDistill `reg_alpha` 1.0 → 0.5, both
clearing the √2σ margin. FedPAQ, DAdaQuant and FedKD retained their published
constants. Manuscript Table 4.1 carries the revised values; the tag freezes them.

**CFD dropped** from the baseline stack (7 -> 6) - collapses to chance
accuracy at production client-count scale; per-client partitions too small for
CFD's 1-bit hard-vote protocol at 100 clients (see `docs/DECISIONS.md`
Decision 26). Config/hook code retained for reproducibility, excluded from
all sweeps.
