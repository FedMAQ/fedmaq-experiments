# ADR-0029 — The first-study arms are extended to the V2 seeds, screened post hoc, and any advantage is confirmed on fresh seeds

**Status**: Accepted · 2026-10-08
**Amends**: ADR-0016, section "2026-09-25 V2 registration", bullet "Seeds" ("used once"), and the arm choice that dropped FedProx and FedDistill from V2. ADR-0028 item 5, final sub-bullet (seed-exclusivity guard). The `v2_confirm` stage, its scalar, its claim rule and its verdict are unchanged.
**Decided by**: the thesis author, in the 2026-10-08 direction review (#145).

## Context

The first study ran on seeds 0, 42 and 123 and was scored at one common terminal byte budget (ADR-0012). V2 ran six arms on seeds 19, 37, 73, 101 and 131 and was scored with the paired per-seed interpolated budget `B*_s`. ADR-0016 calls that scalar V2-only and forbids comparing it with the first-study scalar. V2's verdict for the no-KD candidate is `not_supported`.

The author wants every first-study arm comparable with the V2 arms. Two routes were weighed:
- **Re-score both studies under one rule.** This needs no new runs, but the seeds still differ, and it overrides ADR-0016's separation for a comparison that stays confounded by seed set.
- **Run the missing arms on the V2 seeds.** Every arm then shares one seed set and one scoring rule. The author prefers this and does not treat compute as binding.

`git diff 030ad9d 6664d70 -- conf src` (first-study dispatch to V2 dispatch) changes no shared-arm behaviour:
- Most of the diff is opt-in V2 diagnostic code, gated on `v2_diagnostic.enabled`, FedMAQ and the validation split.
- It also adds stage and matrix registrations.
- The one other change moves the wall-time clock past observer work; simulated time is unaffected.

Code has changed since V2 dispatch. `git diff 6664d70 22aad3d -- conf src` touches 21 files, including KD, validation and partition code (ADR-0028 work). Reusing the V2 arms as partners therefore needs evidence that the dispatch commit reproduces them.

The author also wants an exhaustive post-hoc search for any aspect where FedMAQ, or any of its flavours, performs well. A panel member advised the author to look for such an aspect. Searching many metrics on data already seen makes false positives likely.

## Decision

1. **A new stage, `v2_extension`, runs the first-study downstream arms that V2 lacks, on the V2 seeds and the test split.** It uses 100 rounds, the frozen first-study hyperparameters, and each condition's first-study model.

   | Block | Arms | Conditions | Cells |
   | --- | --- | --- | --- |
   | E1 benchmark extras | FedProx, FedDistill | CIFAR-10 α∈{0.1, 1.0}, CIFAR-100 α∈{0.1, 1.0}, FEMNIST | 2×5×5 = 50 |
   | E2 ablation | Configurations 2–7, pipeline off as in `ablation.yaml` | CIFAR-10 α∈{0.1, 1.0} | 6×2×5 = 60 |
   | E3 ablation breadth | Configurations 2–7, pipeline off | CIFAR-100 α∈{0.1, 1.0}, FEMNIST | 6×3×5 = 90 |
   | E4 memory sensitivity | c_unit 512 and 2,048, pipeline on; full FedMAQ and no-KD | CIFAR-10 α∈{0.1, 1.0} | 2×2×2×5 = 40 |
   | E5 uniform-memory control | uniform memory, pipeline on; full FedMAQ and no-KD | CIFAR-10 α∈{0.1, 1.0} | 2×2×5 = 20 |
   | Total | | | 260 |

   - FedKD is excluded. ADR-0028 item 6 owns its re-entry.
   - Selection-stage cells (matched tuning, formulation, power-mean design, ω) are not rerun. They chose the design and were never test comparisons.
   - E3 and the no-KD arms of E4 and E5 go beyond the first study. They are added for completeness of the screen in item 4. No ablation arm has run on CIFAR-100 or FEMNIST before, so each new (arm, dataset) pair needs a local smoke cell first.
   - Uniform-memory heterogeneity exists only for CIFAR-10 α∈{0.1, 1.0}, so E5 stays there.

2. **`v2_extension` is post-hoc and descriptive.** The V2 seeds have been observed and the V2 verdict is known.
   - No `v2_extension` result changes a first-study or V2 verdict.
   - The ADR-0016 −1.0 pp rule is not applied to `v2_extension` arms.
   - Each arm is scored with the ADR-0016 `B*_s` scalar against stated partners in the same condition:
     - E1 arms pair with each V2 arm.
     - E2 and E3 Configurations 2–6 pair with Configuration 7.
     - Configuration 7 pairs with V2 `fedmaq`. This is the pipeline-off against pipeline-on contrast for full FedMAQ.
     - E4 and E5 arms pair with the V2 arm that has the same KD setting.
   - E2 Configuration 2 and E5 full FedMAQ are the same memory-blind condition in both coding regimes. They differ in `post_process`, not in memory.
   - Extending the V2 scalar to new arms on the V2 seeds is not the cross-study comparison ADR-0016 forbids. First-study and `v2_extension` results for the same arm are read only for direction. They never share a table, pooled estimate or test.

3. **N2 runs in its own stage, `v2_n2_screen`, on the validation split, seeds 0, 42 and 123.** FEMNIST, no-KD with pipeline, p ∈ {0.5, 0, −0.5}: 9 cells listed.
   - p = 0.5 matches the `v2_kd_screen_femnist` no-KD reference cells. If golden repeatability is bit-exact at the dispatch commit, those cells are reused and the stage runs 6 new cells.
   - These seeds already selected p = 0.5 on CIFAR-10, so the ADR-0026 winner's-curse caveat applies.
   - A p is nominated for item 5 when its mean paired delta against p = 0.5 at `B*_s` is ≥ +1.0 pp. A nomination is not a finding.

4. **An exploratory analysis catalogue runs on V2 and `v2_extension` cells, and every computed metric is reported, whatever its result.** Letters follow the direction review. G, a cross-study re-scoring, was replaced by item 1.
   - **A, communication:** upload-only budget; bytes to accuracy targets; partial budgets and area under the accuracy–bytes curve; per-client upload burden; bits per parameter; accuracy–bytes Pareto position.
   - **B, modeled time:** time to target; round and straggler time.
   - **C, utility beyond top-1:** F1, precision and recall; test-loss trajectories; per-class accuracy, available only from the one-seed diagnostic pilot.
   - **D, stability:** seed spread; late-round oscillation; worst seed; sensitivity to skew; collapse rate; a per-condition win/rank map.
   - **E, memory feasibility:** comparator cap violations; accuracy under tighter memory; cost of the clamp.
   - **F, distillation diagnostics:** teacher against global accuracy; the bit-assignment reading of the V2 diagnostic log.
   - **H, KD repair:** the catalogue is rerun on each ADR-0028 family once its confirmation reports.
   - **Unavailable, reported as such:** per-client accuracy (no client-side evaluation), energy (not logged).
   - The report states how many metric × condition × comparator comparisons the screen made.

5. **No aspect is called an advantage of FedMAQ until a registered confirmation on fresh seeds holds.**
   - Each candidate from item 3 or item 4 is registered before it runs, with its metric, direction, comparators, conditions and decision rule.
   - Fresh seeds are any seeds no registered matrix has used. Today that excludes 0, 7, 21, 42, 123, 19, 37, 73, 101 and 131. A test derives the exclusion from the union of all `seeds` in `conf/matrix`.
   - Each condition runs the candidate and every comparator its rule names, on at least five fresh seeds. The count is fixed at registration and is not capped by compute.
   - The thesis reports how many candidates were registered and how many were confirmed, each with its paired 95% interval, as in ADR-0028 item 4.
   - Only a confirmed candidate may be framed as an advantage in the thesis contributions or abstract. An unconfirmed one is reported as an exploratory observation.
   - RQ1's verdict stays negative, and the research question is not rewritten around a finding.

6. **The V2 seeds are used again only by `v2_confirm`, `v2_kd_confirm` and `v2_extension`.**
   - `v2_extension` matrices use phase, `experiment_group`, `stage` and `protocol_stage` `v2_extension`. `v2_n2_screen` uses the same pattern with its own name.
   - `V2_SEED_STAGES` in `tests/test_v2_registration.py` gains `v2_extension` and nothing else.
   - No other matrix may use phase `v2`, group `v2_confirm` or stage `v2_confirm`.
   - Both stages are added to `PROTOCOL_STAGES` in `src/fedmaq/core/protocol.py` and to `conf/protocol/replacement-v1.yaml`:
     - `v2_extension` with split `test`, as `v2_kd_confirm` is;
     - `v2_n2_screen` with split `val`, as `v2_kd_screen` is.

7. **ADR-0028 item 7 is unchanged.** It is read as "no first-study cell or seed is rerun". Running FedProx, FedDistill and the ablation configurations on new seeds does not breach it, and the first-study cells and verdicts stay as registered.

## Consequences

- Every downstream arm except FedKD has five-seed test evidence under one scoring rule. FedKD gets its evidence under ADR-0028.
- Before the first `v2_extension` or `v2_n2_screen` cell:
  - each stage is registered in `conf/protocol/replacement-v1.yaml` with hashed matrix contracts and a `PROTOCOL_STAGES` entry;
  - `v2_confirm` and every first-study hash stay byte-identical;
  - frozen hyperparameters are shown unchanged at the dispatch commit;
  - JupyterHub golden repeatability passes at that commit;
  - at that commit, one cell for each V2 arm (FedAvg, FedPAQ, FedPAQ+pipeline, DAdaQuant, FedMAQ, FedMAQ no-KD) reproduces its `v2_confirm` curve bit-exactly. If any differs, the V2 arms rerun as partner controls inside `v2_extension`;
  - a local smoke cell passes for each (arm, dataset) pair that has not run before;
  - a test asserts that E2 and E3 keep the pipeline off and that E4 and E5 turn it on.
- The thesis reports `v2_extension` and the catalogue as post hoc, beside the registered verdicts. Chapter 6's contributions can name an advantage only after item 5.
- `v2_extension` and `v2_n2_screen` list 269 cells, or 266 new cells when N2 reuses its p = 0.5 cells. Partner-control reruns and confirmation cells are counted when needed.

## Considered options

- **Re-score both studies' existing curves under one rule.** Rejected: the seeds still differ, and it overrides ADR-0016's separation for a comparison that stays confounded.
- **Side-by-side tables, each study under its own rule.** Rejected: readers compare the two scalars anyway.
- **Pool first-study and V2 seeds.** Rejected: the first-study seeds informed the design, so pooling dilutes the registered test.
- **Rerun the whole first study on the V2 seeds.** Rejected: the shared arms already ran in V2, and the selection stages were never test comparisons.
- **Frame a discovered advantage directly, or report it as exploratory only.** Rejected in favour of confirmation: a direct claim invites a hypothesizing-after-results objection, and exploratory-only stays as the fallback.
