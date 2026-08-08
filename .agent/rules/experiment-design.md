# Experiment Design

Task: discrete image classification only. Benchmark datasets: CIFAR-10,
CIFAR-100, FEMNIST.

## Heterogeneity

- **CIFAR-10 / CIFAR-100**: Dirichlet (α) partitioning.
- **FEMNIST**: writer-based natural partitioning (`partition: writer`), not
  Dirichlet — use `heterogeneity=femnist experiment=femnist`.

Exact α values, memory/bandwidth/compute settings and the control-group config
live in `conf/heterogeneity/` and `conf/experiment/default.yaml`. Those files are
authoritative over any prose describing them.

## Baseline stack

Implement under `src/fedmaq/baselines/`. Six baselines plus FedMAQ; the two
dropped entries keep their code for reproducibility and are excluded from every
sweep. Rationale for each verdict: [ADR-0005](../../docs/adr/0005-baseline-stack-membership.md).

| Algorithm | Group | Paper | Config | Status |
| --- | --- | --- | --- | :-: |
| FedAvg | Seminal control | McMahan et al., 2017 | `fedavg.yaml` | 🟢 |
| FedProx | Seminal control | Li et al., 2020 | `fedprox.yaml` | 🟢 |
| FedPAQ | Pure quantization | Reisizadeh et al., 2020 | `fedpaq.yaml` | 🟢 |
| DAdaQuant | Pure quantization | Hönig et al., 2022 | `dadaquant.yaml` | 🟢 |
| FedDistill | Pure KD | Jeong et al. | `feddistill.yaml` | 🟢 |
| FedKD | Hybrid Q+KD | Wu et al., 2022 | `fedkd.yaml` | 🟢 |
| ~~FedMD~~ | Pure KD | Li et al., 2019 | `fedmd.yaml` | ⚫ dropped |
| ~~CFD~~ | Hybrid Q+KD | Sattler et al., 2022 | `cfd.yaml` | ⚫ dropped |
| FedMAQ | Proposed | Bunyi et al., 2026 | `fedmaq.yaml` | 🟢 |

Update this table when adding or porting a baseline.

**Tuned constants** are frozen behind the `pre-registration` tag and carried by
the manuscript's Table 4.1 — FedProx `mu: 0.01` and FedDistill `reg_alpha: 0.5`
moved off their published values during Stage 1b; FedPAQ, DAdaQuant and FedKD
retained theirs. See [ADR-0011](../../docs/adr/0011-baseline-matched-tuning.md).

**FedKD's absolute accuracy sits well below the other baselines. That is
architectural** — a width-0.5 student against a full-size teacher — **not a bug
and not a tuning failure.**

**FedMD is excluded from smoke and regression sweeps.** Keep it out of
`conf/matrix/*.yaml` and out of `scripts/golden_diff.py`'s default `GOLDEN_SET`;
it is the slowest config by a wide margin (disk-persisted, up to 4× `run_epochs`
per round). Re-add it only for a change that actually touches its code path.

**Client model persistence.** Baselines where the server does not aggregate
weights (FedMD and other prediction-averaging / distillation baselines) persist
client state dicts to `.data_partitions/fedmd_models/client_{cid}.pth`, inside the
gitignored partition cache. Without this, local weights are lost across simulated
rounds in Flower. Note that this state is keyed by client ID and **not by run** —
wipe it between golden-diff capture and compare, or runs silently inherit each
other's weights ([ADR-0006](../../docs/adr/0006-determinism-and-the-golden-diff-gate.md)).

## Metrics

Log to WandB for every run:

1. Top-1 test accuracy (%)
2. Cross-entropy loss, and distillation loss when KD is active
3. Precision, recall, F1 (macro-averaged)
4. Cumulative communication overhead (MB/GB, per client and aggregate)
5. Wall-clock runtime (seconds)
6. **Convergence stability — accuracy vs. rounds *and* accuracy vs. transmitted
   bytes.** The second curve is not optional: it is the primary comparison axis
   for every selection verdict ([ADR-0012](../../docs/adr/0012-formulation-selection-and-the-iso-byte-amendment.md)).
