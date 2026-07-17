# Distillation-Baseline Direction & Health Audit

**Last updated:** 2026-07-17 (F10 collapse mechanism fixed & confirmed on formal 50R re-run, both α, width-0.5 MobileNetV2GN student — residual gap reclassified to still-open candidate 3, not resolved; F14 μ mislabel corrected 1.0 → 0.01 canonical; F17 FedKD student → width-0.5 MobileNetV2GN)
**Auditor:** Claude (Opus 4.8), grill-with-docs session
**Lens:** forward-looking — *are the KD baselines + FedMAQ moving in the right
direction, and which implementations look faulty?* Mines archived + recent
experiments for **surviving signal**, not obsolete numbers.
**Distinct from** [`fedmaq-code-audit.md`](fedmaq-code-audit.md) (F1–F9,
fedmaq.py craftsmanship) and [`fedmaq-audit.md`](fedmaq-audit.md) /
[`fedmaq-audit-recos.md`](fedmaq-audit-recos.md) (algorithm/math fidelity). This
pass continues the shared finding counter at **F10**.

## Method & evidence rule

Scope: the 5 KD hooks (`feddistill`, `fedmd`, `fedkd`, `cfd`, `fedavg_kd`) + the
shared `kd_utils.py` engine + `fedmaq.py`. Evidence:

- **Recent (authoritative for levels):** `docs/experiments/mobilenetv2-smoke-50r/`
  (MobileNetV2GN, 50R, α∈{0.1,1.0}) — the current iso-arch (DECISIONS.md #1).
- **Archived (ResNet18GN, deprecated):** `docs/experiments/archive/*`. Per the
  iso-arch switch, their **accuracy levels are void**. Only **relative trends /
  orderings** are read forward — an iso-arch model swap changes levels but rarely
  flips orderings.

**Report-only.** No code changed this pass (explore/confirm freeze). Findings are
inputs to a later fix pass and to the formal grid.

---

## Findings

Severity: 🔴 high · 🟠 medium · 🟡 low.

### 🟡 F10 — FedKD collapses to near-chance accuracy [correctness · collapse FIXED 2026-07-17, residual reclassified to candidate 3]

**Symptom (CONFIRMED, recent MobileNetV2GN run).** FedKD finishes **17.09%**
(α=0.1) / **31.78%** (α=1.0), peaking **20.80%** / **33.62%** — vs FedAvg
41.37% / 66.90%. On CIFAR-10 (10-class, ~10% chance) this is barely above
random. Upload comm is **36.2 / 44.7 MB** — ~100–200× below every other arm
(5000+ MB). The SVD compression path is destroying the learning signal.

**Mechanism resolved (discriminator read, 2026-07-16).** Pulled
`algorithm/fedkd/mean_rank_retained` + `algorithm/fedkd/energy` from the run logs
(`experiments/mobilenetv2-smoke-50r/fedkd/dirichlet_alpha_{0.1,1.0}/experiment_log.csv`).
Result = **rank starvation confirmed; accuracy tracks retained rank.**

| Round | energy | rank_retained | acc (α=1.0) |
| ----: | -----: | ------------: | ----------: |
| 1  | 0.117 | **0.078** | 0.100 |
| 2  | 0.134 | **0.037** | 0.101 |
| 20 | 0.44  | 0.037 | 0.178 |
| 30 | 0.61  | 0.037 | 0.165 |
| 40 | 0.78  | 0.074 | 0.235 |
| 50 | 0.95  | **0.158** | 0.318 |

Config: `tmin=0.1, tmax=0.95` (linear energy ramp). Retained rank is pinned at
**~3.7% of full rank** for rounds 2–~35 — the convergence-critical window — and
accuracy rises **only when rank finally rises** (α=1.0: acc 0.235→0.318 exactly as
rank 0.074→0.158, R40→R50). This directly explains the collapse.

**Candidate verdicts (revised — the data breaks the clean single-cause story):**

1. **Energy-schedule starvation — CONFIRMED as a driver.** The linear ramp parks
   energy in a low-rank regime for most of training, and the model only learns as
   rank recovers late.
2. **Client↔server reference desync — no server-side signature.** Server telemetry
   is internally consistent (rank/energy monotone-ish, no chaos). This *rules out
   desync from the server logs alone*; the client-side `FedKDCompressionHook`
   reference was **not inspected**, so client-side drift is not fully excluded —
   but it is not the primary mechanism.
3. **SVD too lossy for depthwise-separable weights — PARTIALLY confirmed.** Two
   facts point here, not just at the schedule: (a) energy→rank is **non-monotonic**
   (R1 energy 0.117→rank 0.078, but R2 energy 0.134→rank 0.037 — energy up, rank
   *down*), so rank is dominated by each round's delta spectrum, not the threshold;
   (b) even at the schedule's **ceiling** (0.95 energy, R50) rank is only ~13–16%
   and accuracy is still ~½ of FedAvg. The energy-truncation mapping is punishingly
   aggressive for these weights.

**Honest combined read: candidates 1 + 3 jointly** — the schedule starves early
rounds *and* the energy→rank mapping is too lossy even at its best.

**Fix implemented + validated (`diagnosing-bugs` pass, 2026-07-16).** Probed both
candidates directly against the production `FedKDHook`/`compress_tensor` code
path, driving real `MobileNetV2GN` weight deltas through 15 simulated rounds
(structured/overfit-batch gradients to reproduce a concentrated spectrum):

- **Raising `tmin` alone (0.1→0.5): insufficient.** Rank still dips
  non-monotonically mid-schedule (round 5 rank_retained fell back to 0.146) —
  confirms the audit's non-monotonicity finding; a schedule-range shift doesn't
  fix the underlying mapping.
- **Minimum-rank floor (`min_rank_frac=0.25`): fixes it.** Retained rank never
  drops below the floor and rises monotonically from there (0.278 → 0.617 across
  15 rounds), vs. the unfixed baseline dipping to 0.093 mid-schedule. This is the
  landed fix — see `compress_tensor(..., min_rank_frac=...)` in
  `src/fedmaq/baselines/compression.py`, threaded through both the client-side
  `FedKDCompressionHook` (upload path) and server-side `FedKDHook` (download/eval
  path), with `conf/algorithm/fedkd.yaml` defaulting `min_rank_frac: 0.25`.
  Regression test: `tests/test_fedkd_compression.py`.

**Real-run confirmation (`run-minitest`, 2026-07-16, preliminary/10R/α=0.1/seed=0).**
The code-path probe above is not evidence by itself — an advisor review correctly
flagged that it never measured accuracy. Ran the actual `FedKDHook` end-to-end on
CIFAR-10/MobileNetV2GN, `min_rank_frac=0.0` (old default) vs `0.25` (new default):

| Arm | rank_retained | acc R1 | acc R9 (peak) | acc R10 | mean acc R1-10 |
| :-- | :------------: | :----: | :-----------: | :-----: | :------------: |
| `min_rank_frac=0.0` (before) | pinned 3.7–4.5% | 0.100 | 0.169 | 0.131 | ~12.0% |
| `min_rank_frac=0.25` (after) | floored 26.25% | 0.100 | **0.263** | 0.164 | ~15.1% |

Rank restored as designed, and accuracy climbs with it: peak accuracy +9.4pp
(16.9%→26.3%), mean accuracy +3.1pp over the 10-round window — both arms start
identical (round-1 near-chance) and diverge as the floor keeps kicking in. Noisy
(single seed, 10 rounds, preliminary scale) but directionally unambiguous and the
first *real* evidence (not a synthetic probe) that the fix moves the actual
symptom, not just the proxy metric. Logs:
`outputs/2026-07-16/20-04-13/` (before), `outputs/2026-07-16/20-14-08/` (after).

**Action:** code fix is in and now has real-run evidence at minitest scale. FedKD
is eligible to re-enter comparison tables once F13's full MobileNetV2GN smoke
(more rounds, matches the other KD baselines' scale) confirms this holds up.

> **Superseded arch note (2026-07-16, DECISIONS #22).** The runs above trained
> FedKD's *old* SimpleCNN student. That student has since been replaced by a
> width-0.5 MobileNetV2GN (genuinely smaller, depthwise-separable) to remove the
> cross-architecture confound with the iso-arch grid. The `min_rank_frac` fix is
> architecture-agnostic and still applies, but the FedKD *accuracy* figures in
> this section are retired and will be re-measured on the new student in the GPU
> re-run wave. Item 3's "depthwise-separable" framing, previously an imperfect
> match to the SimpleCNN student, now describes the real FedKD student.

**F10 formal re-confirmation (2026-07-17, full 50R smoke, width-0.5 MobileNetV2GN
student, `min_rank_frac=0.25`).** Full-scale rerun (not the 10R minitest above)
on both heterogeneity arms:

| Arm | acc R40 | acc R50 | peak acc | `mean_rank_retained` |
| :-- | :-----: | :-----: | :------: | :-------------------: |
| α=0.1 (severe skew) | 24.04% | 26.41% | 30.10% (R45) | held at 0.278 (floor) |
| α=1.0 (moderate skew) | 34.70% | 36.29% | 38.31% (R49) | held at 0.278 (floor) |

Logs: `outputs/2026-07-16/23-36-50/` (α=0.1), `outputs/2026-07-17/10-20-23/` (α=1.0).

**Verdict: starvation mechanism CONFIRMED FIXED; residual gap reclassified,
not resolved.** Two different claims here, kept separate on purpose:

1. **The floor works** — this is the clean, architecture-independent claim.
   `mean_rank_retained` never drops below 0.278 in either 50R run, matching the
   same-model minitest A/B above (`min_rank_frac=0.0` pinned at 3.7–4.5% vs.
   `0.25` floored at 26.25%, everything else held equal). That minitest, not
   the 50R numbers below, is the causal evidence the fix works.
2. **The 50R runs characterize post-fix FedKD on the production student** —
   they are *not* a before/after comparison. The retired pre-fix figures
   (20.80%/33.62% peak) were measured on the old SimpleCNN student
   (see the superseded-arch note above); crediting the fix with a delta
   against them would conflate the `min_rank_frac` change with the separate
   student-architecture swap (DECISIONS #22). Read the table only as: this is
   what FedKD does today, on the shipped student, with the floor active.

α=1.0 climbs near-monotonically to 38.31% peak; α=0.1 stays volatile
(±10pp round-to-round swings) but does not collapse to chance — both
consistent with "starvation is gone" and inconsistent with "still starving."

**Residual gap points at candidate 3, still open.** FedKD trails every other
baseline by 15–27pp at both α (FedAvg 42.3%/66.9%, FedMAQ 53.2%/60.9%,
DAdaQuant 47.8%/65.1%). `mean_rank_retained` sitting right at the 0.278
floor in both 50R runs — not climbing past it — means the energy-truncation
schedule *wants* to go lower and the floor is doing the work throughout,
not just during early rounds. That is exactly candidate 3's prediction
(SVD too lossy for depthwise-separable weights, independent of the schedule).
**Verdict on F10 narrows to:** the schedule-starvation half of the original
"1+3 jointly" diagnosis (candidate 1) is fixed; the lossy-SVD half
(candidate 3) is unresolved and is the leading explanation for the residual
gap. Uploaded comm rose to 590–740 MB — this is a floor-mandated cost, not a
regression, and still well under FedAvg's ~8.5 GB.

**Action:** FedKD is unblocked for comparison tables — the near-chance
collapse that made it unusable is gone — but report the residual gap as an
open finding about SVD compression on depthwise-separable weights (candidate
3), not as a resolved defect. F13 (the 4 unmeasured KD baselines) is the
remaining run-gate; comparing FedKD's shape against them may help isolate
whether candidate 3 is FedKD-specific or a broader SVD-compression pattern.

### 🟠 F11 — FedMAQ's α=1.0 accuracy deficit is real (persists across models & EMA) [direction · framing]

**Not a bug — a direction finding that constrains the thesis claim.** FedMAQ
trails the uncompressed baselines under *moderate* skew, and the gap survives the
model switch:

| Model (α=1.0) | FedMAQ | FedAvg | Gap | FedMAQ EMA |
| :------------ | :----: | :----: | :-: | :--------- |
| MobileNetV2GN (recent) | 60.93% | 66.90% | −6.0pp | `ema_student=false` |
| ResNet18GN (archived, trend only) | 53.0% (R34)* | 67.57% | ~−14pp* | `ema_decay=0.7` |

*Levels void (wrong model + cancelled run); the *ordering* — FedMAQ below FedAvg
at α=1.0 — is the surviving signal.

The gap holds across **two architectures** and **both EMA settings**, so it is
structural, not a tuning artifact: quantization + KD acts as a regularizer that
*costs* accuracy when data is easy (α=1.0) and *pays off* when heterogeneity is
severe (α=0.1, where FedMAQ **peaks 53.17% vs FedAvg 42.29%**, best loss, ~40%
less comm).

**Framing consequence (consistent with DECISIONS.md #3, mechanism-primary):** lead
with **comm-efficiency + severe-skew robustness**, never raw α=1.0 accuracy. The
smoke `comments.md` proposes Student-EMA closes the moderate-skew gap — that is a
**testable hypothesis, not established**; the formal grid must sweep
`ema_student` (already Decision-flagged) rather than assert it.

### 🟡 F12 — `num_public_samples=200` dead-fallback in three KD hooks [config · latent]

`cfd.py:298`, `fedavg_kd.py:97`, `fedmaq.py:484` fall back to `200` when the key
is absent; canonical is **3000** (`conf/experiment/{default,preliminary,femnist}.yaml`).
Confirmed **latent, not active**: every shipped experiment config supplies
`num_public_samples: 3000`, so the `200` branch never executes in normal runs.
Downgraded from open-issue #1109's implied severity.

**Why it still matters (footgun):** `num_public_samples` is sliced *before* the
Dirichlet partition advances the numpy RNG (per refactor note), so a silent
divergence here would silently diverge partitions across arms. Same class as
code-audit **F8** (code default ≠ shipped yaml).

**Action:** align the three fallbacks to `3000`, or read fail-loud (raising
sentinel) so a missing key is caught, not masked. Bundle with F8's fix.

### 🟡 F13 — Evidence gap: 4 of 5 KD baselines have never run on MobileNetV2GN [coverage · gating]

The recent smoke covers **FedKD** among the KD family, but **FedMD, FedDistill,
CFD, and FedAvg+KD have zero MobileNetV2GN runs** — only deprecated ResNet18GN
archives. Their implementations were read this pass and look sound (see Verified
below), but *behavioral* health on the current iso-arch is unmeasured.

**This finding gates the others:** no KD-family comparison claim (and no
confirmation-grid entry for these four) is defensible until each has at least a
smoke run on MobileNetV2GN. **Action:** extend `mobilenetv2-smoke-50r` (or a new
smoke) to the four missing KD baselines before the freeze. `run-minitest` is the
right tool.

### 🟡 F14 — FedProx α=0.1 late-round collapse is model-specific [stability · out-of-KD-scope]

Recorded for completeness (not a KD hook). FedProx (μ=0.01, canonical) peaks
**49.04% (R45)** then diverges to **24.79%** with loss **3.30** by R50 on
MobileNetV2GN — yet was *stable and strong* on ResNet18GN (49.71%, no collapse).
So the collapse is specific to the depthwise-separable + GroupNorm architecture,
not FedProx logic — and it is **real at the shipped config**, not a bad-μ artifact.
**Action:** none in this audit; flag for the formal-grid **stability watch** —
proximal μ may need per-model tuning or the run may need convergence guards.

---

## Verified correct — *not* gaps (recorded to prevent false-positive re-raises)

- **FedDistill+ / FedMD do no server-side temperature/KL — by design, not a
  missing-KD bug.** FedDistill (Jeong et al. 2018) exchanges per-label averaged
  logits; the distillation loss lives **client-side**. FedMD (Li & Wang 2019)
  averages public-set class scores server-side; clients train to the consensus
  (L1/CE) locally — confirmed in `client_hooks/fedmd.py`. Server-side
  logit-averaging only (`feddistill.py:72`, `fedmd.py:72`) is faithful.
- **Shared KD engine `kd_utils.py` is sound.** Teachers run under `no_grad`
  (`:56`, properly detached); temperature applied symmetrically to teacher softmax
  (`:60`) and student log-softmax (`:92`); Hinton `T²` gradient rescaling (`:95`);
  entropy+precision soft-voting weights normalize over teachers (`:82-86`);
  arch/shape mismatch re-raised fail-loud (`:161-165`, code-audit F6). No
  asymmetry between the FedMAQ and FedAvg+KD paths (they share this body).
- **CFD dual-distillation path is temperature-consistent.** Server encode
  (`cfd.py:152`) and server-model training (`cfd.py:244-245`) both apply `/T` with
  `T²` KL scaling; the class-sorted public-loader shuffle-before-SGD fix is
  documented (`cfd.py:216-221`). (Prior known soft-label one-hot bootstrap concern
  — memory obs #695 — is separate and not re-litigated here.)

## Direction synthesis — trends that survive the model switch

Read from archived sweeps (orderings only) + recent smoke; these inform the frozen
config, not headline numbers:

1. **Soft-voting helps, robustly:** +2.48pp (α=0.1), +3.96pp (α=1.0) at R50.
   Keep it on.
2. **Default `temperature=1.0` is justified:** T=2.0 clearly *hurts* under severe
   skew (43.63% vs 50.20% at α=0.1); roughly neutral at α=1.0. No case for raising
   T.
3. **Hybrid > pure components under severe skew:** FedMAQ peak **53.17%** vs its
   pure-quantization cousin DAdaQuant **47.75%** (+5.42pp) on MobileNetV2GN — the
   Q+KD combination validates its own direction where it is meant to win.
4. **Severe-skew robustness is the real story:** FedMAQ + DAdaQuant stay stable at
   α=0.1 while FedProx collapses (F14) — adaptive-precision quantization prevents
   the parameter-space divergence, exactly the mechanism-primary claim.

## Summary table

| ID  | Severity | Category            | Locus                                   | Status / action |
| :-- | :------- | :------------------ | :-------------------------------------- | :-------------- |
| F10 | 🟡 | correctness (collapse fixed, residual open) | `compression.py`/`fedkd.py` SVD truncation | rank starvation (candidate 1) eliminated by `min_rank_frac=0.25`, confirmed on formal 50R re-run (2026-07-17, both α) — no more near-chance collapse. Residual 15-27pp gap vs. other baselines reclassified as still-open candidate 3 (SVD too lossy for depthwise-separable weights): `mean_rank_retained` sits at the floor throughout, not climbing past it. Unblocked for comparison tables; candidate 3 stays a finding to report, not resolve. |
| F11 | 🟠 | direction · framing | FedMAQ (mechanism)                      | not a bug; lead thesis with comm+severe-skew, treat "EMA closes α=1.0 gap" as hypothesis to sweep |
| F12 | 🟡 | config · latent     | `cfd.py:298`,`fedavg_kd.py:97`,`fedmaq.py:484` | align fallback to 3000 or fail-loud; bundle with code-audit F8 |
| F13 | 🟡 | coverage · gating   | FedMD, FedDistill, CFD, FedAvg+KD       | run MobileNetV2GN smoke for the 4 missing KD baselines before freeze (`run-minitest`) |
| F14 | 🟡 | stability (out-of-KD-scope) | FedProx (μ=0.01, canonical)     | model-specific collapse; formal-grid stability watch |

**Headline:** the *direction* is sound — soft-voting, T=1.0, and hybrid>pure all
hold as trends, and FedMAQ's severe-skew robustness is the defensible mechanism
story. Two items need action before the formal grid: **F10** (FedKD mechanism
confirmed and fix landed — a min-rank floor stops the SVD collapse on a production
code-path probe; a real formal-grid run still needs to confirm it end-to-end) and
**F13** (four KD baselines are unmeasured on the current model, gating every
comparison claim). **F11** is not a defect but the single most important *framing*
constraint. This session's code change (F10 fix) breaks from the prior
report-only/freeze posture — see `HANDOFF.md` for why.
