# KD-repair screen (`v2_kd_screen`): registered settings

[ADR-0028](../adr/0028-kd-repair-branch-reopened-and-fedkd-re-entry.md) (items 2–4) owns the screen's design, its advance rule, and the claim rule. This note records what the three `v2_kd_screen_*` matrices fix beyond that. It covers each family's three settings, the order that breaks ties between them, a one-line rationale for each setting, and the guard family's proxy-label disclosure. `conf/protocol/replacement-v1.yaml` pins the matrices' resolved hashes. Changing a setting here without re-registering the matrices changes nothing that runs.

The screen is exploratory. A family's best setting is its **entrant**. An entrant advances to confirmation only under ADR-0028 item 4. A result from this screen is an association on the validation split and does not show that a repair works.

## Arms

Every arm runs the FedMAQ pipeline (`algorithm.post_process=true`) on the five first-study conditions, with seeds 0, 42, and 123:

- **`fedmaq_no_kd`**: the reference each repair is compared against.
- **`fedmaq`**: unrepaired KD. It is descriptive only, so each repair can be read against the KD pass it modifies.
- **18 repair arms**: each is `fedmaq` with exactly one family enabled through `+algorithm.kd_repair.<family>.*`. The runner refuses an arm that enables two families.

## Tie order

When two settings of a family tie on the advance rule's statistic, the one listed first wins. In the five parametric families, settings are listed from the smallest departure from the frozen KD pass to the largest, so a tie keeps the smaller change. The schedule family tests two hypotheses, not one parameter. Its early-window settings, which the #119 pilot trace motivates, come before the late-start setting, which no evidence yet supports. Within the early window, the longer window departs less and comes first.

## Settings and rationales

Rationales draw on the 2026-09-24 server-KD shortlist in fedmaq-literature (`docs/audits/2026-09-24-server-kd-paper-shortlist.md`) and on the pilot trace in [fedmaq-experiments#119](https://github.com/FedMAQ/fedmaq-experiments/issues/119). Neither source tests any of these rules in FedMAQ's configuration. A rationale explains why a setting is worth screening. It does not predict that the setting will work.

### Schedule

The KD weight is interpolated in parameter space, `θ = θ_avg + w·(θ_kd − θ_avg)`, and the pass is skipped when `w = 0`. `start` and `stop` are inclusive 1-based rounds.

| Order | Variant | Setting | Rationale |
|---|---|---|---|
| 1 | `sched_r1to40` | `start=1, stop=40` | In the #119 pilot, KD led no-KD at rounds 1–3 and trailed from round 20. Stopping KD at 40 keeps the early lead and ends the later drag. |
| 2 | `sched_r1to20` | `start=1, stop=20` | The same reading with the stop moved to the round where KD began to trail. |
| 3 | `sched_r25ramp10` | `start=25, ramp_rounds=10` | The opposite hypothesis: KD helps only once the parameter average is good enough to serve as its warm start. This is a conjecture. No shortlisted paper tests a delayed KD start. |

### Temperature

The KD loss is scaled by T², following Hinton et al. (2015), as settled with the author in [fedmaq-experiments#127](https://github.com/FedMAQ/fedmaq-experiments/issues/127). Gradient magnitudes therefore stay comparable across settings, and each setting changes only how soft the targets are. The frozen value is T = 1.

| Order | Variant | Setting | Rationale |
|---|---|---|---|
| 1 | `temp_t2` | `T=2` | Softer targets pass more of the teachers' non-argmax mass (Hinton et al., 2015). Softening is listed first as the conventional KD direction. |
| 2 | `temp_t0p5` | `T=0.5` | Sharper targets, the same factor of two in the other direction. This tests whether overconfident pooled targets from skewed teachers are the problem. |
| 3 | `temp_t4` | `T=4` | A larger departure in the softening direction. |

Fan et al. (ICML 2024) tie KD's benefit to target calibration, but they study centralized, labelled KD, and the shortlist cautions against citing them to tune temperature before teacher error and calibration are measured. They are therefore not a rationale for any setting here. The frozen T = 1 stays justified in `conf/algorithm/fedmaq.yaml`.

### Teacher selection

For each proxy batch, the teachers kept are those whose batch-mean predictive entropy, normalized by log C, is at or below the threshold. The kept teachers get equal weight. A batch with no teacher at or below the threshold is skipped. The no-op threshold is 1.0.

| Order | Variant | Setting | Rationale |
|---|---|---|---|
| 1 | `select_h0p90` | threshold 0.90 | Selective-FD (Shao et al., 2024) filters inaccurate or ambiguous predictions from skewed clients before they teach. A high threshold drops only near-uniform teachers. |
| 2 | `select_h0p75` | threshold 0.75 | A stricter filter. |
| 3 | `select_h0p60` | threshold 0.60 | The strictest filter. It tests whether a few confident teachers teach better than the whole ensemble at severe skew. |

Selective-FD selects per client and per sample, in a different topology. The entropy threshold here is a server-only adaptation.

### Teacher weighting

Teacher weights are proportional to `exp(−β · JS(teacher, leave-one-out ensemble))`, computed per proxy batch. The no-op is β = 0.

| Order | Variant | Setting | Rationale |
|---|---|---|---|
| 1 | `weight_b0p5` | β = 0.5 | Kovalchuk et al. (UAI 2026) evaluate reliability-weighted aggregation of client predictions under skew. Down-weighting teachers that disagree with the rest is an unlabelled stand-in for that reliability signal. |
| 2 | `weight_b1` | β = 1 | A stronger down-weighting. |
| 3 | `weight_b2` | β = 2 | The strongest down-weighting. |

The Jensen–Shannon divergence in natural log is at most ln 2, so two teachers' weights differ by at most a factor of 2^−β: about 0.71 at β = 0.5, 0.5 at β = 1, and 0.25 at β = 2. The spread in practice is smaller, so `weight_b0p5` may behave close to unrepaired `fedmaq`. With two teachers, each leave-one-out ensemble is the other teacher, the two divergences are equal, and the weights tie.

Kovalchuk et al. fit their reliability model on a labelled client calibration split, which FedMAQ does not have. AE-KD (Du et al., 2020) is the counterpoint: teachers that disagree may carry the complementary knowledge a student needs. Agreement weighting is therefore an adaptation and is not the published estimator.

### Accept-if-better guard

After the KD pass, the distilled student is kept only if `proxy_ce_after ≤ (1 − margin) · proxy_ce_before`, where CE is the labelled cross-entropy on the proxy set. Otherwise the pre-KD parameter average is returned unchanged.

| Order | Variant | Setting | Rationale |
|---|---|---|---|
| 1 | `guard_m0` | margin 0 | FedBE (Chen & Chao, 2021) separates ensemble quality from the student's ability to recover it. FedDF's supplement (Table 7) reports an SGD student below FedAvg at α = 0.1. A guard rejects the rounds where the student fails. Margin 0 accepts ties. |
| 2 | `guard_m0p01` | margin 0.01 | Requires a 1% lower proxy CE. |
| 3 | `guard_m0p05` | margin 0.05 | Requires a 5% lower proxy CE. |

**Proxy-label disclosure.** The guard reads the proxy set's labels. Frozen FedMAQ uses those labels only to draw a class-balanced proxy set, never in the KD pass. The guard is the one family that uses them to decide what the server keeps. Weighted-FD's released implementation also scores client predictions with proxy labels, and the shortlist records that using proxy labels this way is a method change even though no new data is acquired. Any thesis claim for a guard entrant must state that it relies on labelled server proxy data.

### Class-balanced KD

Each teacher gets a separate weight for each class c, computed from the participating clients' label histograms and normalized over teachers within the class. `mode: hard` gives equal weight to the teachers whose client holds examples of c and zero weight to the rest. `mode: soft` uses `max(floor, r)`, where `r` is the client's share of c divided by the participants' pooled share of c in that round. A class that no participant holds falls back to equal weights. When every participant holds every class in the same proportion, both modes reproduce the unrepaired pass.

| Order | Variant | Setting | Rationale |
|---|---|---|---|
| 1 | `classbal_soft0p3` | soft, floor 0.3 | In the #119 pilot, KD erased minority classes at severe skew (class 7 fell from 0.358 to 0.028 by round 100). FedGO (Jang et al., ICML 2025; Theorems 3.4 and 3.6) ties the right client weights to per-client sample counts and data density. A high floor departs least from uniform weights. |
| 2 | `classbal_soft0p1` | soft, floor 0.1 | Closer to label-share weighting. |
| 3 | `classbal_hard` | hard | Only teachers that saw a class teach it. DaFKD (Wang et al., CVPR 2023) is the domain-aware precedent. |

FedGO and DaFKD estimate relevance from input density using generator and discriminator machinery. The label-histogram weight here is a simpler adaptation that needs each participant's class counts at the server.
