# Soft-Voting Explore Sweep — MobileNetV2GN (Priority 1, Pass 1)

**Date**: 2026-07-18
**Config**: CIFAR-10, Dirichlet α=0.3 (explore-α, `conf/heterogeneity/dirichlet_alpha_0.3.yaml`), 50 rounds, seed=0, MobileNetV2GN.
**Runner**: `scripts/run_soft_voting_explore.py` (+ `scripts/run_soft_voting_explore_resume.py` for crash recovery).
**Output**: `multirun/2026-07-18/03-30-59-soft-voting-explore-mobilenetv2/{0..17}/`

## Ablation arm (soft_voting on/off, ew=1.0/pw=1.0)

| idx | label | round-50 top-1 |
| :-: | :-- | :-: |
| 0 | sv_on | 0.4920 |
| 1 | sv_off | 0.4796 |

## Sweep arm (entropy_weight × precision_weight, soft_voting=true)

| idx | ew | pw | round-50 top-1 |
| :-: | :-: | :-: | :-: |
| 2 | 0.5 | 0.5 | 0.4982 |
| 3 | 0.5 | 1.0 | 0.4996 |
| 4 | 0.5 | 2.0 | 0.4891 |
| 5 | 0.5 | 4.0 | 0.4903 |
| 6 | 1.0 | 0.5 | 0.4906 |
| 7 | 1.0 | 1.0 | 0.5021 |
| 8 | 1.0 | 2.0 | 0.4908 |
| 9 | 1.0 | 4.0 | 0.4838 |
| 10 | 2.0 | 0.5 | **0.5179** (max) |
| 11 | 2.0 | 1.0 | 0.4959 |
| 12 | 2.0 | 2.0 | 0.4763 |
| 13 | 2.0 | 4.0 | 0.4785 |
| 14 | 4.0 | 0.5 | 0.4916 |
| 15 | 4.0 | 1.0 | 0.4768 |
| 16 | 4.0 | 2.0 | 0.4933 |
| 17 | 4.0 | 4.0 | 0.4601 (min) |

Sweep-grid spread: 0.4601-0.5179 (5.78pp), median 0.4907. 14/16 cells cluster in 0.4838-0.4996 (~1.6pp band); idx10 and idx17 are the two outliers.

## Data-quality note

Runs 14 and 17 had duplicate rows in their raw `experiment_log.csv` from Ray-crash retries (Windows Ray instability, `flower-patterns.md`) — deduped in place post-hoc (kept first occurrence per round). Run 17 needed 5 relaunches to finish; treat its result (sweep floor) as lower-confidence than the other 17 runs pending a clean rerun.
