# V1 round telemetry for FedMAQ diagnosis

Date: 2026-09-24. Purpose: record what the replacement V1 evidence can support when explaining how it informs V2 formulation exploration. This is an evidence inventory and diagnostic guide; it does not select a V2 change.

## Evidence and scope

The local certified bundle is [`campaign-evidence-2026-09-22`](../../../campaign-evidence-2026-09-22/README.md), for execution candidate `94d44c1079ff4e7f505ebabc005fd38ea8602d08`. It contains 499 cells: 325 selection/validation cells under `outputs/explore/` and 174 downstream/test cells under `outputs/formal/`. The bundle guide states that telemetry is local and final model weights (3.91 GB total) remain on the JupyterHub host; the local checkpoint index records their paths, sizes, and hashes ([bundle README, lines 16–17](../../../campaign-evidence-2026-09-22/README.md#L16)).

I checked all 499 local `experiment_log.csv` files: each has one row for every round 0–100. Round 0 is the initial evaluation; rounds 1–100 are trained rounds. The per-cell files include `run_manifest.json`, `experiment_log.csv`, and `experiment_log.jsonl`; the bundle guide also directs readers to `.hydra/overrides.yaml` and the referenced `.data_partitions/*.json` files ([bundle README, lines 29–32](../../../campaign-evidence-2026-09-22/README.md#L29)). A representative FedMAQ pair is the full and no-KD CIFAR-10, alpha 0.1, seed 123 ablation:

- [`fedmaq/experiment_log.csv`](../../../campaign-evidence-2026-09-22/outputs/formal/cifar10_mobilenetv2/ablation/fedmaq/dirichlet_alpha_0.1/seed_123/experiment_log.csv)
- [`fedmaq_no_kd/experiment_log.csv`](../../../campaign-evidence-2026-09-22/outputs/formal/cifar10_mobilenetv2/ablation/fedmaq_no_kd/dirichlet_alpha_0.1/seed_123/experiment_log.csv)

The formal `benchmark_grid` has 15 FedMAQ runs: six each for CIFAR-10 and CIFAR-100, and three for FEMNIST. Across rounds 1–100, the 1,500 rows have mean logged `algorithm/fedmaq/avg_q` 6.439 and mean `tier1_binding_fraction` 0.417. The applied-width histogram totals 18,000 client-round decisions: `q_count_8` is 9,722 (54.0%) and `q_count_16` is zero. The denominator is 18,000 because CIFAR runs sample 10 of 100 clients per round, while FEMNIST runs sample 20 of 200; these settings are in the per-run manifests/configurations. This pooled description is a prompt to inspect conditions and rounds separately, not evidence that 8-bit concentration caused an accuracy result.

Protocol boundaries matter when reusing the logs: the matched-tuning, stage 1a, and stage 1b cells are validation-only; downstream uses test after validation verdicts are frozen ([`replacement-v1.yaml`, lines 5–24](../../conf/protocol/replacement-v1.yaml#L5)). This reread also inspects V1 downstream/test trajectories for exploratory diagnosis and V2 hypothesis generation. Record that second use: the V1 test remains the registered V1 outcome, but is exploratory evidence for any V2 decision informed by it. Do not present it as independent confirmation.

## What each round records

The representative CSV header contains these groups (the full header is in the linked raw file):

| Group | Persisted fields |
|---|---|
| Evaluation | `round`; `test/{loss,accuracy,precision,recall,f1}`; `val/{loss,accuracy}` |
| Communication | `communication/{round_bytes,round_payload_bytes,round_upload_bytes,round_secondary_bytes,cumulative_bytes,cumulative_mb,cumulative_upload_bytes,cumulative_upload_mb}`; `communication/client_bytes_uploaded_{mean,min,max,std}` |
| Timing | `system/{round_time_sec,cumulative_time_sec,client_sim_time_sec,cumulative_client_time_sec,server_sim_time_sec,cumulative_server_time_sec,wall_time_sec,cumulative_wall_time_sec}` |
| Client aggregates | `client/{avg_train_loss,avg_train_acc,avg_local_loss,avg_task_loss,avg_distill_loss,avg_task_loss_student,avg_task_loss_teacher,avg_kd_loss_student,avg_kd_loss_teacher,avg_teacher_acc,avg_epochs_trained,avg_q,modeled_capacity_mb_mean,modeled_capacity_mb_min,modeled_capacity_mb_max}` |
| FedMAQ planner/KD | `algorithm/fedmaq/server_kd_loss`; `algorithm/fedmaq/{avg,min,max,std}_grad_norm`; `algorithm/fedmaq/{avg,min,max,std}_q`; `algorithm/fedmaq/{avg,min,max,std}_q_k_max`; `algorithm/fedmaq/tier1_binding_fraction`; for each `b` in `{2,3,4,5,6,7,8,16}`, `algorithm/fedmaq/q_count_b` and `algorithm/fedmaq/q_hat_count_b` |

The JSONL snapshots additionally retain `algorithm/fedmaq/dropped_teachers` and `algorithm/fedmaq/kd_skipped` in trained rounds. Read JSONL when a metric may be added dynamically: the telemetry writer notes that CSV columns are fixed from the first row and later keys can be omitted ([`telemetry.py`, lines 421–466](../../src/fedmaq/core/telemetry.py); [traceability audit, persistence caveat](2026-09-02-reporting-evidence-traceability-audit.md)).

The hook defines `q_count_b` as counts of final applied client widths and `q_hat_count_b` as counts of ladder-snapped targets before the Tier-1 cap; it also logs width moments, raw `q_k_max` moments, binding fraction, and gradient-norm moments ([`fedmaq.py`, lines 140–223](../../src/fedmaq/core/strategy_hooks/fedmaq.py)). These are round summaries, not client-keyed assignments: they do not say which client moved from a given pre-cap bin to an applied bin.

## What V1 can and cannot diagnose

| Existing signal | It can help screen | It cannot establish from these logs |
|---|---|---|
| Per-round evaluation loss/accuracy alongside `server_kd_loss`, `kd_skipped`, and `dropped_teachers` | Whether outcome changes coincide in time with KD loss/status changes or skipped/dropped teachers | Pre-KD versus post-KD accuracy on the same round; teacher/proxy logits, agreement, calibration, or which examples drove the loss. Test metrics evaluate the returned global state after the round. |
| `q_hat_count_b` versus `q_count_b`, `q_k_max` moments, `tier1_binding_fraction`, `avg_q`, and gradient-norm summaries | Whether the planner’s snapped targets and applied-width mix differ across rounds; whether Tier 1 appears to bind often; whether allocation summaries move with outcomes | Per-client width assignments or paired pre-cap-to-applied transitions; layer/tensor quantization error; whether a width histogram caused an accuracy change. |
| Client aggregate training/local loss and accuracy versus global test/validation loss and accuracy | Whether the logged local-training and global-evaluation measures move differently at run level | Per-client distributions, client identities tied to metrics/widths, or a direct measure of client drift. Generic KD/client-teacher columns are blank or absent for FedMAQ; FedMAQ performs server-side KD. |
| Round and cumulative bytes alongside applied-width histograms | How communication cost changes with the observed allocation summaries; whether an outcome comparison is sensitive to the selected byte budget | Quantization error or a causal accuracy-per-bit effect isolated from other round changes. |
| Full FedMAQ and existing no-KD paired runs | Whether removing KD aligns with a different trajectory or paired result in the tested CIFAR-10 ablation conditions | Generalization beyond those datasets/conditions/seeds, or that KD alone caused instability. |

There are no saved per-round teacher/proxy predictions or logits, individual teacher outputs, per-client applied-width records, quantization-error/residual norms, or per-round model checkpoints in the local telemetry. The final checkpoint files are host-only per the bundle guide. `server_kd_loss` is useful as a process signal, but it is not a pre/post outcome comparison.

One scoped example shows why the analysis unit matters. In the CIFAR-10 alpha 0.1, seed 123 no-KD ablation, the registered paired readout reports no-KD accuracy 0.2264 at common bidirectional budget 8,449,728,298 bytes (round 99), with paired delta −0.1000 versus full FedMAQ; at the terminal point no-KD is 0.3471, with paired delta +0.1355. The raw full-FedMAQ accuracy over rounds 96–100 is `.2951, .2910, .3101, .3264, .2116`; no-KD is `.3223, .3003, .3805, .2264, .3471`. The paired manifests differ in `kd_epochs` (1 versus 0); the no-KD arm's zero KD loss is the disabled setting. Both are 100-round closed-loop runs, so later gradients and bit plans can also diverge. Both arms jump between adjacent rounds, which supports trajectory-level inspection but does not isolate a direct same-round KD effect or KD as the source of the jumps. Sources: [`paired_seed_scores.csv`, row for seed 123](../../../fedmaq-analyses/reports/study1_replacement_2026_09_23/paired_seed_scores.csv); the two raw logs linked above.

## Quantitative reread of V1 trajectories

The reproducible script [2026-09-24-v1-round-diagnostics.py](2026-09-24-v1-round-diagnostics.py) reads local CSV/JSONL plus each run manifest, checks rounds 0–100, and emits 27 run rows: 15 formal benchmark FedMAQ runs and six paired CIFAR-10 full/no-KD ablations (two arms × three seeds × two alpha values). From the experiments repo root, run `uv run --offline python docs/research/2026-09-24-v1-round-diagnostics.py`; stdout is CSV. It records `kd_epochs` from the manifest so the no-KD zero-loss/status fields remain identifiable as a disabled treatment.

Formulas: late accuracy is mean global `test/accuracy` over rounds 91–100. Late instability is the mean and maximum of `|accuracy_r − accuracy_(r−1)|` for endpoints `r=91…100`, in percentage points. The local-train/global-evaluation difference is `client/avg_train_acc − test/accuracy`; its drift is the rounds 91–100 mean minus the rounds 1–10 mean, also in percentage points. This difference is descriptive, not a conventional generalization gap: the client training and global test measurements use different samples and evaluation processes. The KD association is per-run Pearson `r` between `server_kd_loss_r` and `test/accuracy_r − test/accuracy_(r−1)` for rounds 2–100. Applied/snapped-pre-cap q8 shares pool their respective histogram counts over rounds 1–100. Histogram movement is the mean per-round total-variation distance, `0.5 × Σ_b |q_count_b/n − q_hat_count_b/n|`; it is the minimum assignment movement compatible with the two marginal histograms, not observed client-level transitions. Binding is the mean logged `tier1_binding_fraction`; communication is logged cumulative MB at round 100.

### Formal benchmark outcomes by cell

Each row is a separate run; no dataset or alpha pooling was used.

| Condition / seed | Late accuracy % | Mean / max late step pp | Local-train/global-eval difference drift pp | Cumulative MB at round 100 |
|---|---:|---:|---:|---:|
| CIFAR-10 α=.1 / 0 | 19.26 | 5.17 / 15.73 | +3.25 | 9,048.87 |
| CIFAR-10 α=.1 / 42 | 25.05 | 5.30 / 14.54 | +2.13 | 9,063.57 |
| CIFAR-10 α=.1 / 123 | 23.27 | 2.71 / 9.21 | −7.58 | 8,991.62 |
| CIFAR-10 α=1 / 0 | 52.87 | 2.14 / 3.44 | +18.54 | 9,440.07 |
| CIFAR-10 α=1 / 42 | 50.74 | 1.65 / 3.80 | +20.33 | 9,405.64 |
| CIFAR-10 α=1 / 123 | 50.98 | 2.36 / 6.95 | +16.31 | 9,414.17 |
| CIFAR-100 α=.1 / 0 | 11.14 | 1.25 / 2.55 | +19.59 | 9,883.44 |
| CIFAR-100 α=.1 / 42 | 14.39 | 1.56 / 4.31 | +23.38 | 9,863.02 |
| CIFAR-100 α=.1 / 123 | 12.89 | 1.57 / 4.28 | +24.08 | 9,842.35 |
| CIFAR-100 α=1 / 0 | 23.91 | .73 / 1.05 | +39.48 | 9,950.33 |
| CIFAR-100 α=1 / 42 | 22.17 | .76 / 1.81 | +44.92 | 9,933.75 |
| CIFAR-100 α=1 / 123 | 22.07 | 1.70 / 3.79 | +37.08 | 9,927.99 |
| FEMNIST α=1 / 0 | 62.47 | .98 / 2.86 | +16.75 | 13,228.87 |
| FEMNIST α=1 / 42 | 63.15 | .70 / 2.00 | +14.94 | 13,176.46 |
| FEMNIST α=1 / 123 | 64.05 | 1.24 / 2.49 | +18.51 | 13,182.71 |

CIFAR-10 α=.1 has the largest late step changes in this grid; α=1, CIFAR-100, and FEMNIST are generally smoother by the same measure. The local-train/global-evaluation difference widened in every CIFAR-100 and FEMNIST seed and all CIFAR-10 α=1 seeds, while CIFAR-10 α=.1 was mixed. These patterns identify questions for follow-up; they do not identify a cause.

### Mechanism summaries by condition

Ranges below are across the three listed seeds. Correlations are ordered seed `0 / 42 / 123`. All 15 benchmark runs had `kd_skipped=0` and zero dropped-teacher rounds.

| Condition | Late KD-loss mean range | Per-seed `r(KD loss, Δaccuracy)` | Applied / snapped pre-cap q8 share range | Mean histogram movement range | Binding range |
|---|---:|---|---:|---:|---:|
| CIFAR-10 α=.1 | .0547–.0689 | +.064 / +.133 / −.084 | 32–40% / 58–64% | 28.8–36.4% | 32.8–40.8% |
| CIFAR-10 α=1 | .0621–.0673 | −.502 / −.351 / −.449 | 58–61% / 97–98% | 38.8–42.0% | 39.8–43.2% |
| CIFAR-100 α=.1 | .1570–.1772 | −.323 / −.235 / −.298 | 58–61% / 95.8–97.9% | 38.8–42.1% | 40.5–43.8% |
| CIFAR-100 α=1 | .1107–.1324 | −.338 / −.368 / −.592 | 58–61% / 91.9–92.6% | 38.9–42.2% | 43.4–46.9% |
| FEMNIST α=1 | .0283–.0407 | −.205 / −.310 / −.225 | 51–57% / 91.1–94.2% | 35.2–44.7% | 36.8–45.7% |

The pre-cap q8 share exceeds the applied q8 share in every condition, often by 30–40 percentage points; binding is present in roughly one-third to just under one-half of rounds' sampled clients. This supports stratifying the planner-to-cap seam by condition and round. The per-round histogram difference does not identify which clients changed widths. The KD-loss correlation is negative in 13/15 benchmark runs, but ranges from near zero/mixed at CIFAR-10 α=.1 to moderately negative elsewhere; shared training trends and serial dependence can produce this association. No run recorded a skipped KD round or dropped teacher, so V1 status flags do not explain the observed seed-to-seed accuracy differences.

The mean logged raw `q_k_max` is 8.155–9.204 across these runs; a sampled client reached cap 15 in 806 of 1,500 trained rounds. The configured width ladder is `{2,3,4,5,6,7,8,16}` and `_snap_floor` maps feasible caps from 9 through 15 down to 8 ([`quantization_planner.py`, lines 126–141](../../src/fedmaq/core/quantization_planner.py#L126); [replacement protocol ladder](../../conf/protocol/replacement-v1.yaml#L38)). V1 does not retain a per-client cap histogram, so these summaries do not quantify how often each intermediate cap occurred. They motivate a diagnostic hypothesis: test whether adding feasible widths in the 9–15 gap improves planner-to-cap alignment or the accuracy/communication trade-off, with actual distortion and transmitted bytes measured. A zero applied q16 count is structurally compatible with caps below 16 and is not by itself evidence of a defect.

### Paired CIFAR-10 no-KD ablation

The registered paired deltas below are no-KD minus full FedMAQ. The common bidirectional budgets are 8,449,728,298 bytes at α=.1 and 8,596,027,939 bytes at α=1.

| α / seed | Late accuracy full / no-KD % | Mean late step full / no-KD pp | Common-budget Δ pp | Terminal Δ pp |
|---|---:|---:|---:|---:|
| .1 / 0 | 22.80 / 33.70 | 4.69 / 6.67 | +16.61 | +6.27 |
| .1 / 42 | 26.32 / 39.43 | 3.67 / 3.38 | +14.44 | +14.58 |
| .1 / 123 | 26.69 / 30.36 | 3.14 / 6.36 | −10.00 | +13.55 |
| 1 / 0 | 53.46 / 62.10 | 1.55 / .53 | +8.68 | +7.63 |
| 1 / 42 | 51.47 / 58.76 | 1.66 / 1.29 | +5.48 | +7.21 |
| 1 / 123 | 51.20 / 56.83 | 2.13 / .82 | +5.78 | +4.83 |

No-KD scored higher at both paired readouts in all three α=1 seeds. At α=.1 it scored higher at the common budget in two seeds and lower in seed 123, while terminal deltas were positive in all three. Late step changes also occurred in both arms; no-KD was less stable in two α=.1 seeds and more stable in all three α=1 seeds. This is a small, closed-loop ablation: after round 1, changing KD can change global weights, later client gradients, planner inputs, and bit assignments. It supports a hypothesis about the KD configuration's trajectory-level association in these CIFAR-10 conditions, not a same-round direct KD effect or a general claim across tasks.

### How to use these results for V2

This quantitative reread inspects V1 downstream/test trajectories to generate hypotheses. Those V1 test results remain the registered V1 outcomes, but are exploratory evidence for any V2 choice informed by this reread. Fresh seeds on the same test images can improve replication across random seeds; they do not restore an untouched test set after V1 test results and labels have been inspected. Validation can screen candidate formulations; a pristine, still-unseen holdout is ideal for strong confirmation. If no such holdout is available, label V2 results on the reused test images as exploratory/conditional, not independent confirmation. Existing V1 baseline curves are historical references, so V2 screening does not require a broad baseline rerun. A new-seed superiority claim needs matched targeted comparators, or must be framed explicitly as descriptive/unpaired.

The no-KD comparison has an additional boundary: the formal benchmark-grid manifests set `post_process=true`, while the paired CIFAR-10 ablation manifests set `post_process=false` ([benchmark manifest](../../../campaign-evidence-2026-09-22/outputs/formal/cifar10_mobilenetv2/benchmark_grid/fedmaq/dirichlet_alpha_0.1/seed_123/run_manifest.json); [ablation manifest](../../../campaign-evidence-2026-09-22/outputs/formal/cifar10_mobilenetv2/ablation/fedmaq/dirichlet_alpha_0.1/seed_123/run_manifest.json)). Thus the ablation's paired differences are evidence about its pipeline-free CIFAR-10 conditions; do not transfer their effect size to the post-processed benchmark runs.

## Evidence-to-hypothesis map

These are diagnostic interpretations to check against the per-cell traces, not selected V2 changes.

| Observed pattern to verify | Hypothesis it could motivate | What must be checked before treating it as support |
|---|---|---|
| KD loss correlates negatively with same-round accuracy changes in 13/15 formal benchmark runs | KD behavior may be worth probing as a trajectory-level factor | This is a within-run time correlation with serial dependence and common training trends; V1 has no same-round pre-KD evaluation or teacher outputs. All formal runs had zero skipped KD or dropped-teacher rounds. |
| Snapped pre-cap and applied histograms differ, binding is frequent, and sampled caps reach 15 while the ladder jumps from 8 to 16 | Planner-to-cap alignment or feasible-width resolution in the 8–16 gap may be worth testing | Recover per-client caps if available; quantify candidate widths' distortion and transmitted bytes. Current V1 aggregate telemetry does not identify exact cap counts or client transitions. |
| Local-train/global-evaluation difference widens in a condition | A mismatch between client training measurements and global evaluation may be worth investigating | Check validation curves and paired conditions; this difference is not a conventional generalization gap, and V1 lacks per-client or per-class outcome diagnostics. |
| Ranking changes between a fixed byte budget and terminal round | The conclusion depends on where along the communication trajectory it is measured | Report both registered budget and endpoint, paired by seed; do not label either scalar as the full trajectory. |
| Pipeline-free no-KD arm scored higher at both readouts in all three α=1 seeds; at α=.1 it was higher at 2/3 common budgets and 3/3 endpoints | The KD configuration may merit a targeted matched comparison | This is a small closed-loop ablation on CIFAR-10 with `post_process=false`; its effect size does not transfer to post-processed benchmark runs. |

## Reproducible analysis checklist

1. Resolve each cell using the certified stage ledger and `run_manifest.json`; record dataset, alpha, seed, algorithm/configuration, split, stage, commit, and protocol identity. Keep selection/validation rows separate from downstream/test rows.
2. Load `experiment_log.jsonl` as the scalar source and cross-check the CSV. Verify the expected sequence 0–100; analyze rounds 1–100 as trained rounds and preserve round 0 as initialization.
3. For a hypothesis, first plot per-seed round-wise evaluation loss/accuracy and its candidate mechanism signal (KD status/loss, q histograms/binding, or bytes). Inspect traces before averaging; then report per-condition summaries across seeds.
4. For paired comparisons, join only matching conditions and seeds. Report the registered common-budget round/value and terminal round/value separately; retain paired deltas and identify the budget metric (bidirectional bytes here).
5. V1 downstream/test traces may be reread to generate exploratory V2 hypotheses, but record that reuse and do not treat the same test results as confirmatory evidence. Use validation for screening; use a pristine holdout for strong confirmation when available. If V2 reuses the same test images, label its evidence exploratory/conditional.
6. Record the exact source paths, manifest identities, row/round selections, aggregation rule, and any exclusions so a panel can reproduce the reasoning from V1 evidence.

## Primary implementation references

- [FedMAQ telemetry hook](../../src/fedmaq/core/strategy_hooks/fedmaq.py#L140) defines planner/KD summaries and histogram semantics.
- [Telemetry schema and writer](../../src/fedmaq/core/telemetry.py#L39) defines stable fields and CSV/JSONL persistence.
- [Server KD metrics](../../src/fedmaq/core/kd_utils.py#L192) emits dropped-teacher and skipped-step status.
- [Replacement V1 protocol](../../conf/protocol/replacement-v1.yaml#L5) defines validation-only selection and frozen-verdict downstream evaluation.
