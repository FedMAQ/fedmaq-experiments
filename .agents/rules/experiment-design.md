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

Implement under `src/fedmaq/baselines/`. Six baselines plus FedMAQ were
executed in the replacement campaign; FedKD is excluded from its primary
comparison. The two dropped entries keep their code for reproducibility and
are excluded from every sweep. Rationale:
[ADR-0005](../../docs/adr/0005-baseline-stack-membership.md).

| Algorithm | Group | Paper | Config | Status |
| --- | --- | --- | --- | :-: |
| FedAvg | Seminal control | McMahan et al., 2017 | `fedavg.yaml` | 🟢 |
| FedProx | Seminal control | Li et al., 2020 | `fedprox.yaml` | 🟢 |
| FedPAQ | Pure quantization | Reisizadeh et al., 2020 | `fedpaq.yaml` | 🟢 |
| DAdaQuant | Pure quantization | Hönig et al., 2022 | `dadaquant.yaml` | 🟢 |
| FedDistill+ | Distillation-augmented parameter sharing | Zhu et al., 2021 | `feddistill.yaml` | 🟢 |
| FedKD | Hybrid low-rank compression + KD | Wu et al., 2022 | `fedkd.yaml` | ⚠️ first-study comparison excluded |
| ~~FedMD~~ | Pure KD | Li et al., 2019 | `fedmd.yaml` | ⚫ dropped |
| ~~CFD~~ | Hybrid Q+KD | Sattler et al., 2022 | `cfd.yaml` | ⚫ dropped |
| FedMAQ | Proposed | Bunyi et al., 2026 | `fedmaq.yaml` | 🟢 |

Update this table when adding or porting a baseline.

**Tuned constants.** Read the shipped values from `conf/algorithm/` and the
replacement Stage-A verdicts from
[ADR-0011](../../docs/adr/0011-baseline-matched-tuning.md). Historical
Stage-1b tuning values are superseded for this campaign.

**FedKD first-study validity.** The replacement campaign produced nonfinite
test losses in three CIFAR-10 severe-skew seeds. Treat the FedKD arm as
attempted but excluded from the first-study primary comparison; do not
attribute the failure to architecture or implementation until the bounded
forensics in [ADR-0016](../../docs/adr/0016-v2-evaluation-protocol-and-advance-rule.md)
are complete. Its code and evidence remain available for diagnosis.

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

## Paired-seed selection statistics

**Count exact zeros in the paired differences before reporting a paired t.**
With \(n-1\) of \(n\) differences exactly zero, \(t \equiv 1\) identically for
any \(n\), independent of the nonzero difference's magnitude — the statistic
carries no effect-size information. A high tie count more generally caps the
achievable \(t\). This is degeneracy, distinct from significance: sign
inconsistency across seeds (some seeds favor one arm, some the other) is a
separate, non-degenerate failure mode — genuinely computed, genuinely
inconclusive — not weaker evidence but evidence of a different kind. Worked
instance: [2026-09-14 Stage 1a power-mean selection analysis](../../docs/research/2026-09-14-stage1a-power-mean-selection-analysis.md#3-behavioral-observations).
