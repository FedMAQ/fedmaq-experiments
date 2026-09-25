# ADR-0028 — The KD-repair branch is reopened on the accuracy endpoint, and FedKD re-enters behind a smoke gate

**Status**: Accepted · 2026-09-25
**Amends**: ADR-0016, section "2026-09-25 V2 registration", bullets "KD-repair branch" and "FedKD". The `v2_confirm` stage it registered is unchanged.
**Decided by**: the thesis author, in the 2026-09-25 decision interview.

## Context

ADR-0016 closed the KD-repair branch on the #119 exploratory pilot. The pilot bounded any repair by the ensemble-minus-average accuracy gap, which was 0.2 pp in the no-KD priority cell against a 1.0 pp margin. It allowed reopening only with loss, calibration or an early-budget scalar as the *primary* endpoint.

That bound covers only repairs that distill toward the same uniform ensemble. It does not bound two kinds of repair that the pilot itself points to:
- **Repairs that change the teacher.** At severe skew the best single teacher reached 0.264 against the ensemble's 0.212, and KD erased minority classes (class 7 fell from 0.358 to 0.028 at round 100).
- **Repairs that change the schedule.** KD led no-KD at rounds 1–3 and trailed from round 20.

The pilot also ran one seed per regime. The author wants the thesis to show that KD was given a serious chance to work before it is dropped, and compute is not a constraint.

FedKD was excluded after nonfinite losses in three CIFAR-10 α=0.1 cells. #118 traced this to an implementation defect, not a property of the algorithm:
- The hook passes non-detached softmax outputs as the KL target. When a probability underflows to 0, `nn.KLDivLoss` returns a finite loss but a NaN gradient with respect to the target logits.
- Detaching both targets kept all three seeds finite through 12 rounds.

No other baseline has a demonstrated defect. Two literature-adherence audits already corrected and disclosed the known deviations (#43).

## Decision

1. **The KD-repair branch reopens with matched-byte accuracy against no-KD as its primary endpoint.** This overrides ADR-0016's condition that a reopened branch must use loss, calibration or an early-budget scalar as primary. Loss and calibration are reported as secondary endpoints and never used as gates.
2. **Six families are tested, one mechanism per arm, three settings each:**
   - schedule;
   - teacher selection;
   - teacher weighting;
   - temperature;
   - an accept-if-better guard;
   - class-balanced KD.

   No two mechanisms are combined in this pass.
3. **The one-seed pilot is replaced by a direct screen.** It runs all 18 settings plus a no-KD+pipeline reference over the five first-study conditions, on seeds 0, 42 and 123, on the validation split. The screen is exploratory.
4. **The advance rule has no cap.** Every family's best setting advances if both hold:
   - it beats no-KD by ≥ 1.0 pp in at least one priority cell (CIFAR-10 α=1.0 or FEMNIST);
   - it is no worse than −1.0 pp in every other condition.

   The multiplicity is handled by disclosure and a pre-declared choice, not by a cap:
   - the thesis reports how many repairs reached confirmation, with each one's paired 95% interval;
   - the recommended configuration is the confirmed repair with the largest worst-case priority-cell margin over no-KD.
5. **Confirmation runs on the V2 seeds (19, 37, 73, 101, 131) on the test split.** It uses the ADR-0016 V2 claim margins plus a ≥ −1.0 pp margin against no-KD.
   - The `v2_confirm` runs serve as paired controls if JupyterHub golden repeatability is bit-exact at the repair commit. Otherwise the controls rerun.
   - The repairs are selected on other seeds and another split, so this is their first look at these seeds.
   - The seed-exclusivity guard in `tests/test_v2_registration.py` widens to admit the new stages and nothing else.
6. **FedKD re-enters with its KL targets detached, behind a smoke gate.**
   - **Gates, in order:**
     1. Local smoke of seeds 123, 0 and 42 at CIFAR-10 α=0.1 past round 12, plus a regression test for a zero-probability KL target.
     2. A 100-round preflight of the same three cells on the hub at the fix commit, finite in every round.
   - **Only after both pass:** the other 12 first-study-seed cells, then 25 confirmation-comparator cells.
   - **If a gate fails:** FedKD stays excluded, and a second fix needs the author.
   - The Stage A `tmax` is kept.
   - The first-study FedKD arm stays reported as invalid. A labelled post-hoc row reports the fixed cells beside it, and the detach is disclosed as an adaptation that the source paper leaves unspecified.
7. **No other baseline is rerun, and nothing in the first study is rerun.**

## Consequences

- The thesis frames FedMAQ as a two-tier allocation framework whose components are each tested. KD is either repaired and recommended, or dropped after six repair families were tried and failed.
- The Chapter 6 research-question answer and practice recommendations wait for the repair confirmation. §5.7 can report `v2_confirm` first.
- Every new stage is registered in `conf/protocol/replacement-v1.yaml` with hashed matrix contracts before its first cell runs. `v2_confirm` and every first-study hash stay byte-identical.

## Considered options

- **Keep the branch closed** (ADR-0016). Rejected: its bound does not cover repairs that change the teacher or the schedule, and it rests on one seed.
- **Rerun every baseline, or the whole first study.** Rejected: no baseline other than FedKD has a demonstrated defect. Replacing sealed, pre-registered evidence without a cause would weaken the gates the thesis relies on.
- **Advance at most two repairs** (the historical #15 plan). Rejected: compute does not bind, and disclosure plus a single pre-declared recommendation handles multiplicity.
