# Baseline Status Audit — all 8 baselines + FedMAQ

**Last updated:** 2026-07-17
**Auditor:** Claude (Sonnet 5), static-pass session
**Lens:** implementation-readiness — *for each algorithm, what is its status
(config-consistent? has a run on the current iso-arch? behaviourally sane? open
findings?) and which need revision before the formal grid.*
**Posture:** **static pass only — no runs, no fixes applied this pass.** All
run-based verdicts (F13 gaps, FedProx μ, FedKD re-confirm) are deferred to a
future run session. Recommended code fixes (F12) are documented, not applied.

## Scope & relationship to existing audits

This is the **consolidated status view** across the full baseline stack. It does
**not** re-litigate findings already owned elsewhere — it cross-links them:

- KD-family + FedMAQ mechanism findings **F10–F14** →
  [`distillation-direction-audit.md`](distillation-direction-audit.md).
- `fedmaq.py` craftsmanship findings **F1–F9** →
  [`fedmaq-code-audit.md`](fedmaq-code-audit.md) (F1–F9 largely resolved).
- Algorithm/math fidelity → [`fedmaq-audit.md`](fedmaq-audit.md) /
  [`fedmaq-audit-recos.md`](fedmaq-audit-recos.md).

**Value-add of this pass:** (a) status for the four **non-KD** baselines (FedAvg,
FedProx, FedPAQ, DAdaQuant) not covered by the KD-scoped audit; (b) a
config-consistency check of **every** algorithm against manuscript **Table 4.1**
(`fedmaq-manuscript/chapter_4.tex`, `tab:hyperparameters`) — not just the original
papers; (c) new findings **F15–F17**; (d) one consolidated per-algorithm status
table.

> **Authority note.** The **manuscript is not the source of truth** — it is a
> *manifestation* of what the project decides (code + `docs/DECISIONS.md`), and is
> subject to change to match those decisions. So Table 4.1 is used here as a
> **cross-check**, not an authority: where config/decision and manuscript disagree
> (F16, F17), the resolution is to **update the manuscript** to reflect the decision
> (or make a fresh decision), *not* to bend the code back to the manuscript.

## Evidence sources

- **Runs (authoritative for behaviour):**
  [`docs/experiments/mobilenetv2-smoke-50r/`](../experiments/mobilenetv2-smoke-50r/README.md)
  — MobileNetV2GN, 50R, single seed, α∈{0.1,1.0}. Covers FedAvg, FedProx, FedMAQ,
  DAdaQuant, FedPAQ, FedKD. **Smoke-scale, not a benchmark** (single seed, 50R vs
  the manuscript's R=100).
- **Config-consistency reference:** manuscript **Table 4.1** — the values the
  configs' comments cite. Verified this pass:

  | Table 4.1 param | Value | Config | Match |
  | :-- | :-- | :-- | :-: |
  | Rounds `R` | 100 | (smoke used 50, by design) | n/a |
  | FedProx `μ` | **0.01** | `fedprox.yaml: mu: 0.01` | ✅ |
  | FedPAQ `q` | 8-bit | `fedpaq.yaml: q: 8` | ✅ |
  | DAdaQuant `ψ` | 0.9 | `dadaquant.yaml: psi: 0.9` | ✅ |
  | DAdaQuant `φ` | 5 | `dadaquant.yaml: phi: 5` | ✅ |
  | FedKD `T_SVD` | 0.1→0.95 linear | `fedkd.yaml: tmin 0.1, tmax 0.95` | ✅ |
  | FedKD `min_rank_frac` | **(not in Table 4.1)** | `fedkd.yaml: 0.25` | ⚠️ **F16** |
  | KD Temperature `T` | 1.0 | `fedmaq/cfd temperature: 1.0` | ✅ |
  | `D_proxy` / public | 3000 | `experiment/*.yaml: 3000` | ✅ |

  Table 4.1 **does not pin** CFD `b_up/b_down`, FedMD epoch counts, FedDistill
  `reg_alpha`, or FedMAQ formulation/γ params — those are configurable per-run
  (configs note this).

---

## Per-algorithm status table

Status legend: 🟢 ready (run + sane + config-consistent) · 🟡 config-ready but
**unmeasured** on iso-arch · 🟠 needs attention before/at formal grid · 🔴 broken.

| Algorithm | Group | MobileNetV2GN run? | Sane vs FedAvg? | Config vs Table 4.1 | Open findings | Status |
| :-- | :-- | :-: | :-- | :-- | :-- | :-: |
| **FedAvg** | Seminal | ✅ (smoke) | reference (41.4% / 66.9%) | ✅ consistent | — | 🟢 |
| **FedProx** | Seminal | ✅ (smoke, **μ=0.01 confirmed**) | strong then **collapses late (R45→R50)** | ✅ (μ=0.01, verified vs hydra config) | **F14** (real collapse at shipped μ), **F15** (results.md mislabels μ) | 🟠 |
| **FedPAQ** | Pure Quant | ✅ (smoke) | sane (40.3% / 67.0% peak) | ✅ | — | 🟢 |
| **DAdaQuant** | Pure Quant | ✅ (smoke) | strong (47.8% / 65.1% peak, low comm) | ✅ | — | 🟢 |
| **FedMD** | Pure KD | ❌ **none** | unmeasured | not pinned (epochs configurable) | **F13** | 🟡 |
| **FedDistill** | Pure KD | ❌ **none** | unmeasured | not pinned (`reg_alpha` configurable) | **F13** | 🟡 |
| **FedKD** | Hybrid Q+KD | ✅ (smoke, post-F10-fix, 2026-07-17) | weakest arm, confirmed learning, no collapse (26% / 36% final, 30% / 38% peak) | ✅ + `min_rank_frac` floor | **F10 collapse fixed**; residual gap reclassified to open candidate-3 finding (SVD too lossy for depthwise-separable weights) | 🟡* |
| **CFD** | Hybrid Q+KD | ❌ **none** | unmeasured | b=1 paper-headline; Table 4.1 unpinned | **F13**, **F12** | 🟡 |
| **FedMAQ** *(contribution)* | Proposed | ✅ (smoke) | trails at α=1.0, wins at α=0.1 | ✅ | **F11** (framing), **F12** | 🟢* |
| *FedAvg+KD* *(= Ablation Config 6)* | KD ablation | ❌ **none** | unmeasured | T=1.0 ✅ | **F13**, **F12** | 🟡 |

\* FedMAQ status is 🟢 *implementation/config*; **F11** is a framing constraint on
the thesis claim, not a defect (lead with comm-efficiency + severe-skew
robustness, not raw α=1.0 accuracy). FedKD's 🟡 reflects a **partially resolved**
mechanism finding: the near-chance collapse is fixed and confirmed, but the
residual gap vs. other baselines is an open finding (candidate 3, SVD too lossy
for depthwise-separable weights) — not a defect blocking comparison tables, but
not a closed investigation either.

**Smoke numbers (α=0.1 / α=1.0 peak):** FedAvg 42.3 / 66.9 · FedProx 49.0 / 67.2
· FedPAQ 40.3 / 67.0 · DAdaQuant 47.8 / 65.1 · FedMAQ **53.2** / 60.9 · FedKD 30.1 / 38.3 (post-fix)
/ 33.6 (pre-fix). Source:
[`mobilenetv2-smoke-50r/results.md`](../experiments/mobilenetv2-smoke-50r/results.md).

---

## Per-algorithm notes (non-KD baselines — the value-add of this pass)

**FedAvg** — 🟢 the uncompressed reference. Config is a no-op (`post_process:
false`). Full-precision 32-bit; behaves as the anchor every other arm is judged
against. No action.

**FedProx** — 🟠 config is **correct** (μ=0.01 matches Table 4.1) **and the smoke
actually ran μ=0.01** — verified against the surviving hydra artifact
(`multirun/2026-07-15/mobilenetv2-smoke-50r/fedprox/.../.hydra/config.yaml` resolves
`mu: 0.01`; `overrides.yaml` carries no μ override). **`results.md`'s "μ=1.0" label
is a documentation error (F15).** The consequence is the opposite of a
reassurance: the **F14 late-round collapse — peak 49.0% (R45) diverging to 24.8%
with loss 3.30 by R50 — occurred at the shipped canonical config**, not an
off-config stress value. So F14 is a **real stability concern at μ=0.01** on the
MobileNetV2GN (depthwise-separable + GroupNorm) architecture — it was stable on
ResNet18GN, so it is model-specific, but it is **not** dismissible as a bad-μ
artifact. **Consequence:** the formal grid needs a stability watch (or convergence
guard) for FedProx on this architecture; this is the reason FedProx is 🟠, not 🟢.

**FedPAQ** — 🟢 fixed 8-bit quantization (`q: 8` = Table 4.1). Sane, mid-pack
accuracy, comm between DAdaQuant and full-precision. No open findings. Ready.

**DAdaQuant** — 🟢 doubly-adaptive (time + client). ψ=0.9, φ=5 match Table 4.1 and
the paper's best-published values. Strongest pure-quantization arm (peak 47.8% at
α=0.1 with the **lowest upload comm** of any arm, ~4.7 GB), validating the adaptive
axis FedMAQ builds on. No open findings. Ready.

## Per-algorithm notes (KD family — see distillation-direction-audit for depth)

**FedMD / FedDistill** — 🟡 implementations were reviewed in the distillation audit
and **look correct** (server-side logit/score averaging is faithful; the "no
server-side temperature/KL" is by-design, not a missing-KD bug). **But zero
MobileNetV2GN runs (F13)** — behavioural status is **unmeasured**; do not read
"looks correct" as a health verdict the runs haven't earned.

**FedKD** — 🟡 F10 **collapse mechanism fixed, residual reclassified**: SVD rank
starvation (candidate 1) confirmed fixed — `min_rank_frac=0.25` floor landed
(PR #8) and re-confirmed on a formal 50-round re-run (2026-07-17, both α,
width-0.5 MobileNetV2GN student) — see
`docs/audits/distillation-direction-audit.md` F10. Accuracy now climbs well
above chance (peak 30.1%/38.3%) with no starvation-induced collapse. But
`mean_rank_retained` sits at the floor throughout rather than climbing past
it — the original audit's candidate 3 (SVD too lossy for depthwise-separable
weights) is still live and is the leading explanation for the remaining
15-27pp gap vs. other baselines. FedKD is unblocked for comparison tables;
the gap is a finding to report, not a closed investigation.

**CFD / FedAvg+KD** — 🟡 reviewed, look sound (CFD dual-distillation is
temperature-consistent; FedAvg+KD shares the vetted `kd_utils.py` engine). Both
**unmeasured (F13)** and both carry **F12** (`num_public_samples` fallback). FedAvg+KD
is **Ablation Config 6**, realized via the FedMAQ/KD path (`fedavg_kd.py`), not a
standalone baseline config.

---

## New findings this pass

Severity: 🔴 high · 🟠 medium · 🟡 low. Counter continues from F14.

### 🟡 F15 — `results.md` mislabels FedProx μ as 1.0; the smoke actually ran μ=0.01 [documentation error]

The MobileNetV2GN smoke `results.md` records FedProx at **μ=1.0** (and also, self-
contradictorily, "Default configuration"). The **surviving hydra artifact settles
it**: `multirun/2026-07-15/mobilenetv2-smoke-50r/fedprox/dirichlet_alpha_0.1/.hydra/
config.yaml` resolves `mu: 0.01`, and `overrides.yaml` contains **no μ override**.
So the run used the **canonical μ=0.01** (Table 4.1); the "μ=1.0" annotation is a
**documentation error**. **This upgrades, not downgrades, F14:** the late-round
collapse happened at the *shipped* config, so it is a genuine stability finding, not
a bad-μ artifact. **Actions:** (a) correct `results.md`'s FedProx μ label to 0.01;
(b) note that [`distillation-direction-audit.md`](distillation-direction-audit.md)
**F14 propagates the same μ=1.0 mislabel** and should be corrected to "μ=0.01,
canonical" — the collapse is real at the shipped config.

### 🟡 F16 — Manuscript Table 4.1 lags the FedKD `min_rank_frac` decision [manuscript-follows-decision]

The F10 remediation (`min_rank_frac=0.25`, `fedkd.yaml`) is a real, load-bearing
hyperparameter that changes FedKD's behaviour, but Table 4.1 pins only `T_SVD:
0.1→0.95` and does not mention it. Since the manuscript follows decisions (not vice
versa), the code/decision is correct and the **manuscript is out of date**.
**Action:** add `min_rank_frac` (+ its F10 justification) to the manuscript
hyperparameter table, or record an explicit decision that it stays an
implementation-level guard undocumented in the table. `align-manuscript` skill.

> **Resolved 2026-07-16.** Added row `FedKD: Minimum Retained-Rank Fraction
> $r_{\min}$ = 0.25` to Table 4.1 in `fedmaq-manuscript/chapter_4.tex`.

### 🟡 F17 — Manuscript §4.1 architecture (ResNet-18 / LeNet student) lags the MobileNetV2GN iso-arch decision [manuscript-follows-decision]

Manuscript `chapter_4.tex` still specifies **ResNet-18 (~11.17M) teacher** and a
**LeNet-5-style (~2.16M) student** for CIFAR, whereas the code and
[DECISIONS.md #1](../DECISIONS.md) switched to **MobileNetV2GN (~2.24M) iso-arch**.
The decision is authoritative; the manuscript is behind. **Action:** update
manuscript §4.1 model prose (and any Table 4.1-adjacent architecture text) to
MobileNetV2GN via `align-manuscript`. Lands in `fedmaq-manuscript`, not this repo.

> **Resolved 2026-07-16 (DECISIONS #22).** Auditing this surfaced that the iso-arch
> switch made FedKD's old SimpleCNN student (~2.16M) no longer meaningfully smaller
> than the full MobileNetV2GN (~2.24M), *and* off its depthwise-separable family —
> a comparison confound. Resolved by switching FedKD's CIFAR student to a
> **width-0.5 MobileNetV2GN** (~0.59M, ~0.26×; code + `tests/test_models.py`) and
> rewriting §4.1 accordingly: full model = MobileNetV2GN; FedMAQ = full-model
> self-distillation (un-bundled from FedKD); FedKD = compact width-0.5 student.
> FedKD numbers are re-run-gated on the new student. Not a pure manuscript-sync
> after all — it drove a code + design decision.

### ✅ F12 (RESOLVED — PR #9)

`num_public_samples` silently fell back to `200` at four sites — `strategy.py:261`,
`cfd.py:298`, `fedavg_kd.py:97`, `fedmaq.py:484` — while Table 4.1 / all shipped
configs supply **3000** (manuscript §4.1 confirms: 3000 scales DynFed's 200-sample
proxy to the larger benchmarks). Latent (config always supplies the value) but a
reproducibility footgun: the public slice advances the numpy RNG before the
Dirichlet partition, so a silent 200 would divergently repartition. **Fixed:**
all four sites now call `require_num_public_samples(self.config)`
(`src/fedmaq/core/config_defaults.py`), which raises on a missing key instead of
masking it. `partitioning.py:159`'s `= 200` default param is a separate
call-site default; leave or align when the fix lands. Bundle with code-audit F8.

---

## What needs revision before the formal grid (prioritized)

1. **F13 — run the 4 unmeasured KD baselines** (FedMD, FedDistill, CFD, FedAvg+KD)
   on MobileNetV2GN. Gates every KD comparison claim. `run-minitest` is the tool.
   **Only remaining run-gate.**
2. ~~F10 re-confirm~~ — **DONE 2026-07-17.** Formal 50R re-run on both α confirms
   the collapse mechanism is fixed; FedKD unblocked for tables. Residual gap
   reclassified to open candidate-3 finding, not re-opened as a run-gate.
3. **F14 — FedProx stability watch at μ=0.01.** The collapse is at the canonical
   config (F15 confirms the smoke ran μ=0.01), so the formal grid needs a
   convergence guard or per-model μ check for FedProx on MobileNetV2GN. Doc-only
   otherwise — no code action.
4. ~~F12~~ — **DONE (PR #9).** Fail-loud fix applied.
5. **F16 / F17 — manuscript sync** (`min_rank_frac`; MobileNetV2GN architecture) —
   `align-manuscript`, lands in `fedmaq-manuscript`.

## Housekeeping flag

- `conf/algorithm/fedmaq_lite.yaml` — **RESOLVED.** Carries an
  "EXPLORATION-APPENDIX ONLY" header so it isn't mistaken for a grid arm
  ([DECISIONS.md #4](../DECISIONS.md)).

## Bottom line

**FedAvg/FedPAQ/DAdaQuant are 🟢** (run, sane, config-consistent). **FedKD is 🟡**:
its F10 collapse is fixed and confirmed, but the residual gap vs. other
baselines is reclassified to an open candidate-3 finding, not a closed
investigation. **FedProx is config-correct but 🟠**: it
ran the canonical μ=0.01 (F15 corrects a `results.md` mislabel) and still
collapsed late-round on MobileNetV2GN (**F14**) — a real stability concern at the
shipped config, needing a formal-grid stability watch. The **remaining KD-family
gap**: four baselines are **unmeasured (F13)** — none are *broken*, but none can
be given a behavioural verdict until they run; this is the only run-gate left.
Two low-severity **manuscript-sync** items (**F16**, **F17**) are resolved, one
**doc-error correction** (**F15**) is resolved, and the **code fix** (**F12**) is
merged — round out the
list. No 🔴 findings — the stack is not broken, it is **under-measured** (and in
FedProx's case, **under-guarded**); the gating work is runs, not rewrites.
