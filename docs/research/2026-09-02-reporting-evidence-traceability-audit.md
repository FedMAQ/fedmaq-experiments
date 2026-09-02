# Reporting-evidence traceability and campaign-capability audit

Date: 2026-09-02. Scope: all three named lenses of
[issue #102](https://github.com/FedMAQ/fedmaq-experiments/issues/102): trace each
manuscript table, figure, and numerical reporting commitment to its exact evidence;
test whether the campaign can produce the comparison it promises; and determine
whether the preregistration artifact can detect dispatch drift. An additional
analysis-readout lens was added after the source trace exposed a concrete downstream
split defect. Matching metric names alone do not count as closure.

No experiment was dispatched and no configuration, source, manuscript, or committed
evidence file was changed. The findings were routed into the open owning issues;
[issue #103](https://github.com/FedMAQ/fedmaq-experiments/issues/103) was opened for
the otherwise unowned v2 implementation surface.

## Verdict

The replacement-campaign round telemetry is sufficient for the central accuracy, loss,
bidirectional-communication, upload-only, rounds-to-target, simulated-time, and
capacity-summary plots. It also retains the per-seed rows needed for correct
paired analysis. The 415-cell design can support its core descriptive comparisons
without adding scientific cells. However, the currently promised reporting surface
is not implementation-closed:

1. formulation identity omits `p` and `omega` from the analysis record, and the
   headline/ablation matched-budget aggregation pools seeds rather than computing
   a paired per-seed budget;
2. applied-`q` and raw Tier-1-ceiling moments cannot reconstruct the promised
   pre-cap and applied discrete bit-width distributions;
3. a realized label distribution is reconstructible from a partition cache but
   is not a manifest-bound run artifact;
4. macro precision/recall/F1 cannot support the manuscript's promised note about
   class-level imbalance;
5. the FedPAQ-pipeline analyzer currently rejects the downstream `test` split,
   admits unrelated experiment groups, and silently intersects incomplete evidence;
6. the memory-sensitivity matrix has no dedicated exact-set readout;
7. the preregistration contract does not bind exact stage identities/treatments and
   accepts runtime split or wire-protocol overrides without equality checks;
8. t-SNE checkpoint availability is enforced, but the exact comparator/seed and
   projection protocol remain under-specified and the manifest does not bind the
   checkpoint digest; and
9. the v2 repair screen, freeze, and confirmatory reporting surface has not yet
   been implemented, including KD-only server-time and realized proxy-pass fields.

The current-campaign findings are now closed by route: #96 owns analysis identity,
exact readouts, and preregistration closure; #97 owns discrete precision evidence;
#99 owns partition identity; #100 verifies the new fields in a real smoke run; and
#101 owns the reporting choices that can be resolved by checkpoint re-analysis or
claim narrowing. #103 owns the distinct future v2 campaign. These routes are work
still to be implemented, not evidence that the pipeline is already frozen.

## Authority and resolution rule

The manuscript defines what must be reported. The experiment implementation and
schemas define what was or will be recorded. For each commitment below, a route is
accepted only when the source provides:

- a persisted field or a deterministic, version-bound derivation;
- enough granularity for the promised statistic (especially seed pairing and
  discrete distributions); and
- an explicit aggregation whose semantics match the manuscript.

A scalar mean is therefore not evidence for a histogram, a pooled cross-seed
budget is not evidence for a paired seed-wise budget, and a file written on a
best-effort basis is not a manifest-bound artifact.

## Reporting-surface inventory

The empirical commitments occur in these manuscript surfaces:

- formulation selection and robustness table
  ([`chapter_5.tex:15-38`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L15-L38));
- effectiveness, loss, macro-metric, t-SNE, and predictive-utility surfaces
  ([`chapter_5.tex:54-73`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L54-L73));
- bytes curve, bytes-to-target, matched-budget scalar, and communication reduction
  ([`chapter_5.tex:81-88`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L81-L88));
- simulated delay/energy discussion and convergence curve
  ([`chapter_5.tex:100-122`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L100-L122));
- realized environment and variable-versus-uniform table
  ([`chapter_5.tex:134-146`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L134-L146));
- ablation matrix and component attributions
  ([`chapter_5.tex:158-184`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L158-L184));
- v2 repair margins, freeze artifact, and confirmatory results
  ([`chapter_5.tex:196-226`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L196-L226)); and
- Dirichlet-alpha sensitivity curve
  ([`chapter_5.tex:239-245`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_5.tex#L239-L245)).

The underlying metric definitions and seed aggregations are committed in
[`chapter_4.tex:299-378`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_4.tex#L299-L378), while the
formulation and ablation designs are specified in
[`chapter_4.tex:386-462`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_4.tex#L386-L462). The complete
per-cell and reproducibility appendices are also commitments
([`appendix_per_cell_results.tex:5`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/appendix_per_cell_results.tex#L5),
[`appendix_reproducibility.tex:5`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/appendix_reproducibility.tex#L5)).
Chapter 2 literature tables and the Chapter 3/4 architecture, sequence, workflow,
hyperparameter, candidate-setting, and Gantt graphics are static explanatory
artifacts rather than experiment-derived reporting surfaces; they require source
and prose review, not campaign telemetry.

## Exact evidence ledger

| Manuscript commitment | Exact persisted source | Required derivation | Resolution verdict |
|---|---|---|---|
| Top-1 accuracy and cross-entropy loss | `test/accuracy`, `test/loss` in the common CSV/JSONL schema ([`telemetry.py:39-74`](../../src/fedmaq/core/telemetry.py)); evaluation is attached to each round snapshot ([`strategy.py:302-346`](../../src/fedmaq/core/strategy.py)) | Group by algorithm/config/dataset/alpha/seed; report the round requested, then mean and sample SD across seeds (`ddof=1`) as implemented by [`report_schema.py:45-61`](../../scripts/report_schema.py) | **Satisfied.** The round and seed grain is retained. |
| FedMAQ server distillation loss | `algorithm/fedmaq/server_kd_loss` emitted by [`strategy_hooks/fedmaq.py:139-196`](../../src/fedmaq/core/strategy_hooks/fedmaq.py) | Select the requested round or curve; aggregate across seeds only after retaining the run identity | **Satisfied.** The hook registers the exact key into the stable CSV schema and JSONL record. |
| FedDistill task/distillation losses | `client/avg_task_loss`, `client/avg_distill_loss` from [`client_hooks/feddistill.py:174-190`](../../src/fedmaq/core/client_hooks/feddistill.py) | Sample-count-weighted client mean per round, then seed aggregation; the weighting rule is [`telemetry.py:193-216`](../../src/fedmaq/core/telemetry.py) | **Satisfied in JSONL.** The two components are separable and weighted. |
| FedKD student/teacher task and KD losses | `client/avg_task_loss_student`, `client/avg_task_loss_teacher`, `client/avg_kd_loss_student`, `client/avg_kd_loss_teacher` from [`client_hooks/fedkd.py:152-173`](../../src/fedmaq/core/client_hooks/fedkd.py) | Sample-count-weighted client mean per round, then seed aggregation | **Satisfied in JSONL.** The four components are separable. |
| Macro precision, recall, and F1 | `test/precision`, `test/recall`, `test/f1` ([`telemetry.py:39-74`](../../src/fedmaq/core/telemetry.py)); macro averaging is computed in [`evaluation.py:17-76`](../../src/fedmaq/core/evaluation.py) | Per-round value, then mean and sample SD across seeds | **Satisfied for macro metrics; insufficient for class-level imbalance.** No per-class metric or confusion matrix is recorded. |
| Accuracy retention relative to FedAvg | `test/accuracy` plus manifest `run.algorithm`, dataset, alpha, seed, and configuration identity ([`manifest.py:111-161`](../../src/fedmaq/core/manifest.py)) | For each matched seed, `100 * A_method / A_FedAvg`; then mean and sample SD. Do not divide already-aggregated means | **Satisfied if the same seed/cell join is enforced.** |
| Accuracy-versus-cumulative-bidirectional-MB and total communication | `communication/cumulative_mb`, accumulated from measured `communication/round_upload_bytes` plus measured downloads ([`telemetry.py:243-411`](../../src/fedmaq/core/telemetry.py)) | Plot each seed trajectory or a declared cross-seed summary on cumulative MB; total is the round-`R` cumulative value | **Satisfied for the primary held-constant wire seam.** DAdaQuant's `communication/round_secondary_bytes` is a parallel as-published axis and is not folded into this field. |
| Upload-only objective-3 curve and reduction | `communication/cumulative_upload_mb`; `communication/round_secondary_bytes` separately records DAdaQuant's as-published coder ([`telemetry.py:39-74`](../../src/fedmaq/core/telemetry.py)) | Pair FedMAQ with each baseline at the same dataset/alpha/seed and round `R`; compute `100 * (U_baseline - U_fedmaq) / U_baseline`, where `U` is final primary cumulative upload MB; then summarize the three paired reductions as mean/min/max. Report bidirectional MB separately, and report DAdaQuant's secondary axis separately rather than adding it to the primary seam | **Telemetry sufficient; exact report closure routed to #96 and prose distinction to #101.** |
| Bytes-to-target and scalar accuracy at the minimum common budget | Per-round `test/accuracy`, `communication/cumulative_mb`, run seed and algorithm/config identity | Find the first round reaching `0.9 *` the matched FedAvg terminal accuracy. For the common budget, compute each paired seed's budget and scalar before averaging. The v1 discrete lookup is [`analysis.py:1170-1194`](../../scripts/analysis.py); v2 interpolation is [`analysis.py:1197-1251`](../../scripts/analysis.py) | **Telemetry sufficient; current v1 headline/ablation aggregation insufficient.** [`analysis.py:1485-1585`](../../scripts/analysis.py) pools runs/seeds into one budget. Route to #96. |
| Rounds-to-target, final-round accuracy, and not-reached status | `round`, `test/accuracy`, and matched FedAvg rows | First round meeting the paired target; otherwise explicit not-reached; select `round=100` exactly rather than silently substituting another round | **Satisfied in raw records.** Completeness checks exist in [`analysis.py:1300-1363`](../../scripts/analysis.py); promotion must enforce them. |
| Client, server, straggler, total simulated time, and energy proxy | `system/client_sim_time_sec`, `system/server_sim_time_sec`, `system/round_time_sec` and cumulative counterparts ([`telemetry.py:39-74`](../../src/fedmaq/core/telemetry.py)); client straggler is the maximum modeled client delay and total adds server time ([`telemetry.py:243-411`](../../src/fedmaq/core/telemetry.py)) | Sum per-round values or use cumulative fields; energy remains an explicitly modeled proxy rather than measured power | **Satisfied for the v1 time model.** `system/server_sim_time_sec` is not KD-only for FedMAQ because it combines KD and gradient-probe work. |
| Realized memory-capacity distribution | `client/modeled_capacity_mb_mean`, `client/modeled_capacity_mb_min`, `client/modeled_capacity_mb_max`; the exact vector deterministically regenerates from manifest `config`, seed, client count, and uniform/variable mode ([`strategy.py:39-123`](../../src/fedmaq/core/strategy.py)) | Re-run the versioned capacity sampler from the manifest identity to obtain the per-client distribution; moments support only summaries | **Qualified satisfied without rerunning training.** Exact distribution is derivable, but persisting a vector/hash would make the evidence less implementation-dependent. |
| Realized client-label distribution | Partition cache `client_indices`/`public_indices` generated by [`partitioning.py:286-379`](../../src/fedmaq/core/partitioning.py), joined to dataset targets | Count labels by client after public/validation removal; report the distribution or summary statistics | **Insufficient as a run-bound field.** The cache is not hashed or referenced by the run manifest. Route to #99. |
| Assigned/applied bit-width distribution and clamp behavior | Applied-width moments `algorithm/fedmaq/{avg,min,max,std}_q`, raw Tier-1-ceiling moments `algorithm/fedmaq/{avg,min,max,std}_q_k_max`, and `algorithm/fedmaq/tier1_binding_fraction` ([`strategy_hooks/fedmaq.py:139-196`](../../src/fedmaq/core/strategy_hooks/fedmaq.py)) | Count the ladder-snapped pre-cap target and the final applied `q` per round; compare their discrete distributions and binding events | **Insufficient.** No pre-cap-target distribution is persisted, and the applied-`q` moments cannot reconstruct its histogram. `q_k_max` is the raw memory ceiling, not the applied width. #97 owns the two new per-round counts; add a joint cross-tab only if the final claim requires per-client transition pairing. |
| Variable-versus-uniform memory table | Variable FedMAQ rows in group `benchmark_grid` and uniform rows in `uniform_memory_control`; exact manifest fields are `config.experiment_group`, `run.algorithm`, `run.algorithm_config`, `run.dataset`, `run.alpha`, `run.seed`, `run.total_rounds`, and `config.heterogeneity`; outcomes are `test/accuracy`, `communication/cumulative_upload_mb`, `communication/cumulative_mb`, `algorithm/fedmaq/tier1_binding_fraction`, and the #97 distributions | Restrict to CIFAR-10, alpha `{0.1,1.0}`, seeds `{0,42,123}`, round `R`; join same-alpha/same-seed variable and uniform rows; calculate per-seed variable-minus-uniform accuracy and communication deltas, then mean/min/max, retaining both regimes' mechanism summaries | **Raw evidence sufficient after #97; no closed exact-set readout today. Routed to #96.** |
| Formulation ranking, margin, robustness, and false-positive controls | Accuracy/bytes rows plus manifest `run.algorithm`, `run.algorithm_config`, `run.dataset`, `run.alpha`, `run.seed`, and full `config`; sigma, margin, intervals and false-positive calculations are implemented in [`analysis.py:449-624`](../../scripts/analysis.py) | Identify the candidate by `(p, omega, formulation)`, aggregate the predeclared scalar per seed, then compute ranking/margin/robustness statistics | **Insufficient identity.** `p` and `omega` exist only inside `config`, not `RunRecord` or the analysis identity ([`analysis.py:59-197`](../../scripts/analysis.py)). Route to #96. |
| Ablation matrix and paired component deltas | Accuracy/bytes/time telemetry plus exact config identity; comparison tables are assembled in [`analysis.py:2719-2921`](../../scripts/analysis.py) | Pair same-dataset/alpha/seed cells, calculate per-seed deltas, then mean/min/max (and intervals where promised) | **Telemetry sufficient; matched-budget path inherits the pooled-seed defect.** Route aggregation closure to #96 and mechanism/distribution prose to #101. |
| Final-model t-SNE | `final_global_model.pt` written at the final round by [`checkpoint.py:1-73`](../../src/fedmaq/core/checkpoint.py), plus test data and the model definition; completion requires a valid checkpoint through [`common.py:156-165`](../../scripts/common.py) | Load the declared seed/config checkpoint, extract penultimate features, and apply a predeclared comparator, sample, projection seed, and t-SNE parameter set | **Satisfied for checkpoint availability.** Remaining risk is analytical preregistration and optional manifest digest binding, not absence of the artifact. |
| Dirichlet-alpha sensitivity | Manifest `run.alpha`, `run.seed`, `run.dataset`, `run.algorithm`, and `run.algorithm_config` plus the same accuracy/bytes fields | Group the declared metric by alpha; pair seeds where comparisons are made; mean and sample SD across seeds | **Satisfied.** Alpha is first-class run identity. |
| Complete per-cell appendix and reproducibility appendix | Manifest run/protocol/environment/git/dispatch/full resolved config fields ([`manifest.py:111-161`](../../src/fedmaq/core/manifest.py)) plus all round JSONL/CSV records | One manifest and complete round sequence per cell; expose seed-level rows rather than only group means | **Satisfied in schema, subject to promotion checks.** Manifest write failures are currently swallowed ([`manifest.py:183-209`](../../src/fedmaq/core/manifest.py)), so #100 must reject missing evidence rather than treating process exit as completion. |

### Persistence caveat: JSONL is the authoritative superset

The CSV header is fixed from the first written row and later dynamic keys are
ignored, while each JSONL snapshot retains the complete dictionary
([`telemetry.py:421-466`](../../src/fedmaq/core/telemetry.py)). Because round zero
precedes client fit, algorithm-specific client-loss keys can be absent from the
CSV even though they are present in later JSONL records. The loss commitments are
therefore satisfied only if reporting reads JSONL (or if the schema is explicitly
expanded before future runs). A CSV-only analyzer cannot support all promised
loss decompositions.

## Lens 2 — can the campaign produce the claimed evidence?

This lens asks whether each comparison has a matched control, enough independent
seeds for the claim actually made, and an outcome that can respond to the treatment.
It does not reinterpret three seeds as a powered significance study: Chapter 4
already limits inferential claims and treats intervals as descriptive
([`chapter_4.tex:365-378`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_4.tex#L365-L378)).

| Comparison | Seeds and control | Can the metric move? | Capability verdict |
|---|---|---|---|
| Matched baseline tuning | Five paired selection-validation seeds per candidate and reference | Accuracy and per-seed cumulative-MB trajectories are recorded | **Capable after #96.** The evidence exists, but the selector/readout must enforce the registered per-seed support and exact candidate set. |
| Stage 1a degree and Stage 1b weight selection | Three paired validation seeds per candidate, with the family centre retained as a reference | Accuracy and byte curves vary with the formulation parameters | **Capable for the preregistered descriptive selector.** It is not a hypothesis-significance design. |
| Downstream headline benchmark | Three same-cell seeds for FedMAQ and every named baseline, with FedAvg as the uncompressed utility reference | Accuracy, loss, bidirectional bytes, upload-only bytes, and simulated time vary independently | **Capable for mean/SD and paired directional claims.** Do not promote three-seed summaries into population-level superiority tests. |
| Configuration 2 resource ablation | Three paired seeds against full FedMAQ | Accuracy may move only slightly when Tier 1 binds infrequently; binding fraction and realized bit-width distributions can still show the mechanism | **Capable only as mechanism-plus-accuracy evidence after #97/#101.** An accuracy-only null cannot establish that memory awareness is inactive. |
| Other ablations and uniform-memory control | Three paired seeds and a same-pipeline full-FedMAQ anchor | The removed treatment changes planner or KD behavior after #97 restores arm distinctness | **Capable after #97.** Exact config identity and per-seed budgets remain #96 obligations. |
| FedPAQ versus pipeline-equipped FedPAQ | Three paired downstream seeds across five primary cells; ordinary FedPAQ is the matched treatment control | Coding-pipeline bytes and accuracy can differ while quantization settings remain fixed | **Campaign capable, current analyzer incapable.** The implementation defect is L4-01 below. |
| Memory-unit sensitivity | Three seeds at the centre and two bracketing units across five primary cells | Binding fraction, precision distribution, accuracy, and bytes can move | **Campaign capable, no closed readout.** The implementation gap is L4-02 below. |
| Alpha-regime sensitivity | Three matched seeds within each declared regime | All headline and mechanism metrics can move | **Capable as descriptive sensitivity.** #101 now forbids inferential overstatement. |
| v2 server-KD repairs | Not part of the 415-cell replacement ledger | Required treatment identities and KD-specific costs do not yet exist | **Not capable in this campaign.** Routed to the separate future study #103. |

No additional replacement-campaign cells are justified by this lens. Expanding the
current grid would not repair the identified failures: the expensive risks are
missing resolution or broken readout contracts, while the remaining three-seed
limits are already compatible with explicitly descriptive claims.

## Lens 3 — does preregistration constrain dispatch?

The current protocol registration proves a narrow stage-semantic hash, not the
scientific treatment set. `preregistration_contract()` contains selection-data,
reserved-test, split, wire protocol, and historical-promotion fields
([`protocol.py:43-67`](../../src/fedmaq/core/protocol.py)). `register_protocol()`
then accepts runtime `split` and `wire_protocol` overrides without asserting that
they equal the contract ([`protocol.py:97-138`](../../src/fedmaq/core/protocol.py)),
while `is_promotable_manifest()` rechecks the contract hash but not those recorded
values ([`protocol.py:141-182`](../../src/fedmaq/core/protocol.py)).

The generated 145/84/12/174 stage ledgers are valuable exact-set checks, but their
scientific identities are derived from the matrices they check. An accidental
matrix edit followed by regeneration changes both sides together. The protocol
therefore cannot yet detect drift in dataset, alpha, seed, round budget,
algorithm/variant, selected treatment, or full expected identity set.

**Disposition: routed to #96.** Its added criteria require an independently
registered exact stage/treatment authority, startup equality checks, and fail-closed
split/wire/stage behavior. #99 separately binds the realized partition rather than
only the requested split. Until those criteria land, the existence of a valid hash
must not be described as proof that the dispatched scientific design matched the
preregistration.

## Additional lens — can the analysis read the evidence it commissioned?

Two gaps surfaced only after following the expected downstream rows into their
dedicated readouts:

### L4-01 — FedPAQ-pipeline comparison rejects and contaminates real evidence → #96

`compare_fedpaq_pipeline_iso_byte()` calls a validation-only guard even though the
15 control cells and their five primary comparators are downstream reserved-test
runs ([`analysis.py:2049-2059`](../../scripts/analysis.py)). It scopes by algorithm
name rather than experiment group, overwrites duplicate seeds in a dictionary, and
silently intersects whatever algorithms/seeds happen to be present
([`analysis.py:2061-2080`](../../scripts/analysis.py)). A complete wrong-group or
incomplete result can therefore produce a plausible table. #96 now requires the
exact groups, five cells, three algorithms, three seeds, unique identities,
reserved-test split, and per-seed in-support budgets; its regression fixture must
use the real split.

### L4-02 — memory sensitivity has cells but no exact-set analyzer → #96

The matrix and experiment-group constant exist, but the group constant is otherwise
unused in `analysis.py`; no dedicated function joins the centre benchmark cells to
both bracketing units. #96 now requires that closed join and its accuracy,
communication, binding-fraction, and realized-precision outputs. This is a
re-analysis defect if raw rows are complete, not a reason to add or rerun cells.

## Closed findings and issue routes

### L1-01 — formulation identity is under-specified → #96

The manifest's full resolved configuration contains the formulation parameters,
but the analysis `RunRecord`, discovery identity, and group keys do not carry
`p` and `omega` ([`analysis.py:59-197`](../../scripts/analysis.py)). The
formulation ranking can therefore collapse or mislabel distinct candidates.
[Issue #96](https://github.com/FedMAQ/fedmaq-experiments/issues/96) already owns
first-class `p`/`omega` identity and manifest closure. Acceptance must verify the
values reach the manifest, discovered record, grouping key, and emitted table—not
merely that they appear in a YAML file.

### L1-02 — matched-budget scores pool seeds → #96

The raw telemetry retains the needed per-seed curves, but `iso_byte_scores`
selects one common budget over the pooled run list
([`analysis.py:1485-1585`](../../scripts/analysis.py)); both the headline
comparison ([`analysis.py:2525-2616`](../../scripts/analysis.py)) and ablation
analysis inherit it. The manuscript's paired-seed design requires a budget and
score per seed before summary aggregation. #96 already owns this correction and
must cover both paths.

### L1-03 — bit-width moments do not identify the distribution → #97

The existing mean/std/min/max for final applied `q`, the parallel moments for raw
Tier-1 ceiling `q_k_max`, and the binding fraction cannot recover how clients
occupy the discrete ladder before and after the resource cap. No pre-cap `q_hat`
distribution is persisted. This blocks the environment-distribution table and the
mechanism interpretation.
[Issue #97](https://github.com/FedMAQ/fedmaq-experiments/issues/97) already owns
per-round counts for both the snapped soft result and applied ladder. Its
acceptance should verify the counts sum to participating clients and preserve
round/config/seed identity. A joint soft-by-applied cross-tab is additionally
needed only if the final prose makes a per-client transition claim.

### L1-04 — realized label distribution is not manifest-bound → #99

The partition can be deterministically reconstructed from cached indices and
dataset labels, so no training rerun is necessary. But neither a per-client
class-count artifact nor its hash is attached to a run manifest. That is weaker
than the requested exact field trace and makes later reconstruction depend on
unchanged dataset and partition code. [Issue #99](https://github.com/FedMAQ/fedmaq-experiments/issues/99)
already owns post-holdout shard statistics and the validation/partition surface;
add the realized class-count matrix or a content-addressed artifact reference
there.

### L1-05 — upload and bidirectional metrics exist, but must not be conflated → #101

Both axes are recorded: `communication/cumulative_mb` is the bidirectional
headline field, while `communication/cumulative_upload_mb` is the upload-only
objective-3 field. Secondary traffic is separately measurable. No telemetry
change is required for that distinction. #96 now owns the exact paired round-`R`
uplink-reduction formula and output; [issue #101](https://github.com/FedMAQ/fedmaq-experiments/issues/101)
owns the manuscript distinction and related Config 2 mechanism reporting.

### L1-06 — macro metrics do not identify class-level imbalance → #101

The evaluator persists only macro precision, recall, and F1. Those aggregates
cannot reveal which classes are weak or support the promised instruction to
"note any class-level imbalance." The final checkpoints and reserved evaluation
loader make a class-level evaluation recoverable without rerunning training.
[Issue #101](https://github.com/FedMAQ/fedmaq-experiments/issues/101) now requires
either a checkpoint-bound, predeclared per-class evaluation or a narrower macro-only
claim. This is deliberately not routed to #100: a smoke test cannot choose the
scientific reporting contract.

### L1-07 — t-SNE checkpoint availability is satisfied; analysis identity → #101

`final_global_model.pt` is the correct artifact and execution validation already
requires a valid final checkpoint before a run counts as complete
([`common.py:156-165`](../../scripts/common.py)). Availability is therefore
**satisfied**, despite the checkpoint writer itself being best-effort. The
remaining reporting risk is that the manuscript does not yet bind the exact
comparator/configuration, seed or seed-combination rule, sampled examples,
projection seed, or t-SNE parameters. Those are preregistration/analysis choices,
not missing telemetry. #101 now requires those choices before result inspection or
withdrawal of the selective plot. Binding the checkpoint SHA-256 would strengthen
provenance, but it is not required to recover the plot from a run accepted by the
current evidence validator.

### L1-08 — the v2 repair reporting contract has no executable evidence surface → #103

ADR-0016 specifies repair-family identity, per-seed interpolated budgets, screening
and advance matrices, a frozen selection artifact, confirmatory cells, paired
intervals, KD-only server time, and proxy-pass accounting
([`0016-v2-evaluation-protocol-and-advance-rule.md:37-160`](../adr/0016-v2-evaluation-protocol-and-advance-rule.md)).
The repository currently contains no v2 repair configs/matrices, freeze artifact,
separate v2 analyzer/output namespace, KD-only server-time field, or realized
proxy-pass field. Generic accuracy and byte telemetry can be reused, but it cannot
identify or certify that protocol. [Issue #103](https://github.com/FedMAQ/fedmaq-experiments/issues/103)
now owns the implementation/evidence successor to closed
[issue #15](https://github.com/FedMAQ/fedmaq-experiments/issues/15) and ADR-0016.
It is a separate future campaign, not an enlargement of the replacement ledger or
a blocker added to #100.

### L1-09 — #98 and #100 are supporting routes, not gap sinks

[Issue #98](https://github.com/FedMAQ/fedmaq-experiments/issues/98) owns the
FedDistill global-logit norm needed for its specific mechanism interpretation;
the manuscript's current task/distillation-loss decomposition is already
separately recorded and should not be rerouted there.
[Issue #100](https://github.com/FedMAQ/fedmaq-experiments/issues/100) owns the
end-to-end smoke/evidence gate. It should verify all newly required fields and
artifacts after #96–#99 and any follow-up telemetry work land, but smoke validation
does not itself create missing resolution.

## Aggregation contract to preserve

For all manuscript tables and figures, the safe order is:

1. identify one run by manifest-bound algorithm, full configuration, dataset,
   alpha, and seed (including `p`, `omega`, and formulation where applicable);
2. select or interpolate the requested round/budget within that run;
3. calculate derived values against the same seed's comparator;
4. only then aggregate across seeds using arithmetic mean and sample SD, retaining
   `n`; and
5. report explicit not-reached/missing-cell status instead of silently replacing
   a round, seed, or comparator.

Paired deltas committed by the manuscript use per-seed differences and then
mean/min/max ([`chapter_4.tex:365-378`](https://github.com/FedMAQ/fedmaq-manuscript/blob/main/chapter_4.tex#L365-L378)).
The v2 protocol additionally specifies paired Student-t intervals and per-seed
linear interpolation; those semantics must remain isolated from the v1 discrete
analysis namespace.

## Durable routing ledger

| Finding class | Owning ticket and added closure |
|---|---|
| Analysis identity, per-seed budgets, FedPAQ-pipeline exact readout, memory-sensitivity readout, exact preregistration | [#96](https://github.com/FedMAQ/fedmaq-experiments/issues/96) |
| Soft-target and applied-bit-width distributions | [#97](https://github.com/FedMAQ/fedmaq-experiments/issues/97), already added before this final pass |
| FedDistill mechanism norm | [#98](https://github.com/FedMAQ/fedmaq-experiments/issues/98), already sufficient for its specific mechanism claim |
| Realized partition/class-count artifact and manifest digest | [#99](https://github.com/FedMAQ/fedmaq-experiments/issues/99) |
| Real-run histogram/count and partition-digest integration checks | [#100](https://github.com/FedMAQ/fedmaq-experiments/issues/100) |
| Macro-versus-class-level reporting choice, t-SNE protocol, descriptive three-seed scope, v2 prose boundary | [#101](https://github.com/FedMAQ/fedmaq-experiments/issues/101) |
| Future v2 server-KD implementation and evidence surface | [#103](https://github.com/FedMAQ/fedmaq-experiments/issues/103) |

No finding was left in a comment-only queue. #98 needed no new criterion. #100
remains an integration stop line rather than being made the source owner for fields
that belong in #96–#99. #103 is explicitly outside the current replacement ledger
and adds no blocker to #100.

## Residual-risk statement

The closed list distinguishes what a miss would cost:

| Residual condition | Cost if ignored | Disposition |
|---|---|---|
| Discrete precision histograms are absent when campaign cells run | **GPU rerun:** moments cannot reconstruct the distributions | Not accepted; #97 implementation plus #100 real-run check |
| Realized partition artifact is lost or drifts without a run-bound digest | **Potential GPU rerun:** deterministic reconstruction is possible only while the exact cache/data/code remain available | Not accepted; #99 binding plus #100 check |
| FedPAQ-pipeline or memory-sensitivity analyzer remains wrong | **Re-analysis:** raw rows contain the required curves and identities if #96 promotion closure passes | Not accepted; #96 exact readouts |
| Macro metrics cannot identify a class-level pattern | **Re-analysis or prose correction:** evaluate certified checkpoints per class or narrow the claim | Explicit decision deferred to #101; no training rerun |
| t-SNE comparator or projection protocol remains outcome-selectable | **Re-analysis or prose correction:** freeze the choices before inspection or remove the plot | Explicit decision deferred to #101; checkpoint availability is already enforced |
| Three-seed results are described as inferentially conclusive | **Prose correction:** the data support descriptive paired summaries, not a power claim | Not accepted; #101 now constrains the wording |
| v2 results are expected from the replacement campaign | **Separate future GPU campaign, not a rerun:** the evidence surface does not yet exist | Isolated to #103; replacement claims must not absorb it |
| Checkpoint SHA-256 is not embedded in the current run manifest | **Provenance hardening only:** the completion validator still requires a valid checkpoint in the identity-bound output directory | Accepted for the replacement campaign; #103 requires digests for v2 |

The remaining uncertainty is implementation risk inside the routed tickets and the
ordinary possibility of defects exposed by #100's real-run integration check. This
audit does not certify future code, any existing result row, or a pipeline freeze.
It establishes that every presently identified rerun, re-analysis, and prose risk
has an owner, and that the current campaign design needs no new scientific cells.
The author can decide to stop auditing on that closed ledger after the routed
criteria are implemented and verified; no campaign execution is authorized here.
