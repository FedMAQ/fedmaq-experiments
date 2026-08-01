# FedMAQ Decisions Log

Single source of truth for resolved project decisions. Append-only, dated. STATUS.md, RUNBOOK.md, and formal-experiment-plan.md link here instead of repeating this content.

---

## 2026-07-16 — Formal Experiment Framing & Grid (13 decisions)

Resolved via grilling session. Full rationale: [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md).

### Architecture & Framing

1. **Iso-architecture**: every algorithm (FedMAQ + 8 baselines) trains **MobileNetV2GN (~2.24M)**. Baselines re-run from scratch on it. Old ResNet18GN standings retired.
2. **Switch rationale = edge realism only.** Compression ratio (~1.7×) is model-independent at iso-arch (ratio = `avg_bits/32`, set by bit-width allocation, not param count). "Improved comm savings" claim dropped.
3. **Contribution = mechanism-primary**: quantization-robust accuracy under heterogeneity. Comm savings reported honestly as secondary (~1.7×). The ablation table is the headline evidence, not the comm number.
4. **FedMAQ-Lite dropped** from the formal thesis. Same size as main FedMAQ now (SimpleCNN 2.16M ≈ MobileNetV2GN 2.24M); its size-contrast story was the confounded cross-arch comparison being retired. Smoke results → exploration appendix.

### Methodology

5. **Single fixed config per dataset**, held across α. β–α regime dependence reported as a _sensitivity study_, never exploited in headline numbers.
6. **Paired seeds + paired test**: all arms share the same 3 seeds with **identical partitions**; report per-seed deltas + CIs (paired t / Wilcoxon). Cancels seed variance so ~3pp ablation deltas are detectable at n=3.
7. **Baseline parity = matched light tuning**: each baseline gets an equal small budget on its key HP, frozen before confirmation.
8. **Hard explore/confirm freeze**: exploration is adaptive (mechanisms up for debate, single-seed, cheap). It ends by **pre-registering** (git tag) a frozen config + fixed mechanism set. Confirmatory grid runs frozen. Surprises during confirmation become documented findings or a new labeled exploration round — never silent edits.

### Grid

9. **Datasets**: CIFAR-10 + CIFAR-100 + FEMNIST, full grid. **3 seeds × 100 rounds.**
10. **α ∈ {0.1, 1.0}** for CIFAR-10/100 (severe + moderate extremes; intermediate values dropped to cut runs). FEMNIST = writer-partition (no α). Confirmatory run count ≈ **135 runs**.
11. **Freeze granularity**: explore + freeze on **CIFAR-10 (primary)**, transfer to CIFAR-100/FEMNIST with a verification spot-check; re-freeze per dataset only if transfer fails (document deviation as a finding).
12. **Ablation**: additive ladder (narrative) + leave-one-out (rigorous attribution). Run on **CIFAR-10 at α ∈ {0.1, 1.0}** only. ~66 ablation runs.

### Infrastructure

13. **Config-as-code registry**: a manifest enumerates every formal run (algo × dataset × α × seed), hashes frozen configs, and drives the process-isolated runners (mandatory per `hydra-config.md` — no Hydra `--multirun`). Two-phase layout enforces the freeze boundary structurally.

---

## 2026-07-16 — Architecture Refactor: Determinism + Hook Decoupling (PRs #6, #7)

Resolved across the autonomous architecture pass. Commits carry the mechanics;
recorded here is the _why_. See [PR #6](https://github.com/FedMAQ/fedmaq-experiments/pull/6)
(merged) and [PR #7](https://github.com/FedMAQ/fedmaq-experiments/pull/7).

18. **Determinism oracle closed in three halves.** (a) per-Ray-worker torch-flag
    re-pinning + seeded DataLoader (training reproducibility); (b)
    `SeededPartitionClientManager` — client sampling keyed by _partition-id_ with
    per-round RNG, so paired arms at a matched seed draw identical clients; (c)
    deterministic partitioning locked by `test_partition_seed_invariant_for_paired_arms`.
    Enables the paired-seed/paired-test methodology (Decision 6) — seed variance
    cancels, so ~3pp ablation deltas are detectable at n=3.
19. **`generate_partition_indices` is a pure function** of `(dataset, num_clients,
alpha, num_public_samples, seed, partition)` with **no algorithm input**. Single
    call site passes the _global_ `cfg.experiment.num_public_samples`, so every arm
    gets byte-identical partitions. **Footgun:** `num_public_samples` is sliced
    _before_ Dirichlet advances the RNG — divergent per-arm values would silently
    diverge partitions. Safe only because it is one global config value; see audit
    F12 for the related dead-fallback.
20. **Hook-decoupled strategy (Phases 2/4/5/6).** Server model-factory dispatch
    centralized (P2); cross-hook fallback defaults centralized in
    `config_defaults.py` (P4); `configure_fit` god-method split into named helpers
    with an extracted `_QuantParams` (P5); DAdaQuant backward-compat property
    proxies removed from `strategy.py` — tests now hit `strategy.hook.*` (P6).
    Rationale: one baseline's state must not leak into the shared strategy surface.

**Open follow-ups** (not decisions — tracked in
[distillation-direction-audit.md](audits/archive/distillation-direction-audit.md)):
FedKD near-chance accuracy (F10 — mechanism confirmed + fix landed 2026-07-16,
see below), FedMAQ α=1.0 framing constraint (F11), `num_public_samples=200`
dead-fallback (F12), 4 KD baselines unmeasured on MobileNetV2GN (F13).

---

## 2026-07-16 — F10 Fix: FedKD SVD Rank Floor (`diagnosing-bugs`)

21. **`min_rank_frac` floor on SVD-compressed FedKD deltas.** Energy→rank was
    non-monotonic on concentrated (depthwise-separable) spectra: retained rank
    could collapse toward 1 even as the round-scheduled energy target rose,
    starving the convergence-critical window (audit F10). Probed both
    candidate fixes against the production `FedKDHook`/`compress_tensor` code
    path (real MobileNetV2GN deltas, 15 simulated rounds): raising `tmin`
    alone still dipped non-monotonically mid-schedule; a minimum-rank floor
    (retained rank ≥ `min_rank_frac * full_rank`) eliminated the dip and rank
    rose monotonically. Landed `min_rank_frac` param on `compress_tensor`,
    threaded through both `FedKDCompressionHook` (client upload) and
    `FedKDHook` (server download/eval); `conf/algorithm/fedkd.yaml` defaults
    `min_rank_frac: 0.25`. Regression test `tests/test_fedkd_compression.py`.
    A first synthetic code-path probe showed rank recovering but never measured
    accuracy — flagged by advisor review as an incomplete gate (the near-chance
    _accuracy_ symptom, not the rank proxy, is what matters). Followed up with a
    real `run-minitest` (CIFAR-10/MobileNetV2GN, preliminary/10R/α=0.1/seed=0):
    peak accuracy 16.9%→26.3%, mean 12.0%→15.1% as the floor holds rank at
    26.25% vs. the old default's 3.7-4.5% collapse. Noisy at minitest scale —
    F13's full MobileNetV2GN smoke is still the gate before FedKD re-enters
    comparison tables, but the fix now has real-run (not just synthetic)
    evidence. **Superseded-evidence note (see Decision 22):** the real-run
    confirmation above trained the _old_ SimpleCNN student; the fix itself is
    architecture-agnostic (it floors rank), but FedKD's accuracy numbers here
    are retired along with the SimpleCNN student and must be re-measured on the
    width-0.5 MobileNetV2GN student. The synthetic probe already used
    MobileNetV2GN deltas, so "depthwise-separable spectra" now describes the
    real FedKD student too, not just the probe.

---

## 2026-07-16 — F17: FedKD Student Architecture (methodology cohesion)

22. **FedKD CIFAR student = width-0.5 MobileNetV2GN** (path B). The iso-arch
    switch (Decision 1) shrank the full model to ~2.24M, leaving FedKD's old
    SimpleCNN student (~2.16M) neither meaningfully smaller nor on the
    depthwise-separable family the rest of the grid trains — a comparison
    confound (is FedKD's result from SVD+distillation, or from a different
    backbone?) and a collapsed "compact-student" communication story. Rejected:
    (A) iso-arch student=teacher MobileNetV2GN (degenerate mutual distillation
    between identical-capacity nets, unfaithful to Wu 2022's mentor-mentee
    asymmetry) and (C) keep SimpleCNN and merely document the confound. Chose a
    **width-0.5 MobileNetV2GN student (~0.59M CIFAR-10, ~0.26× the full model)**:
    genuinely smaller, same depthwise-separable backbone, so FedKD's SVD now
    compresses depthwise-separable deltas and the compact-student story holds.
    Scoped to CIFAR — FEMNIST keeps its TinyCNN student paired with the
    LeNet-scale full model. Implementation: `get_kd_student_model` +
    num_groups-divisible channel rounding in `MobileNetV2GN` (no-op at
    width_mult=1.0, full-model param counts unchanged); regression test
    `tests/test_models.py`. Clarifies Decision 1's blanket "every algorithm
    trains MobileNetV2GN": FedKD is a deliberate compact-student exception.
    The dropped FedMAQ-Lite variant keeps its **own legacy SimpleCNN student**
    (`get_fedmaq_lite_student_model`), deliberately decoupled from FedKD's so
    this switch does not alter FedMAQ-Lite's archived-smoke architecture
    (Decision 4). **Run-gated:** prior FedKD
    (SimpleCNN) numbers are retired; a re-run on the new student rides the GPU
    wave (F10 re-confirm / F13). Manuscript §4.1 (F17) updated to match.

---

## 2026-07-16 — Context-Docs Management Conventions (4 decisions)

Resolved via grilling session on doc-hygiene drift (stale STATUS.md date, broken section numbering, duplicate registries). Enforcement: [.claude/rules/docs-management.md](../.claude/rules/docs-management.md) + `.claude/skills/docs-audit/`.

14. **Single experiment registry**: `docs/experiments/README.md` is canonical. `.claude/project/experiment_registry.md` deleted (was stale, duplicated coverage).
15. **`docs/plans/` is active-only**: a plan doc exists only while it has open questions. On full resolution, content merges into this file (`DECISIONS.md`) and the plan file is deleted — git history is the historical record, not a docs folder.
16. **Archive pattern = per-directory `archive/` subfolder** (e.g. `docs/experiments/archive/`), not a single top-level `docs/archive/`. Archived content stays next to its live counterpart.
17. **Enforcement = rule + skill**: `.claude/rules/docs-management.md` (always-loaded conventions, applied during routine edits) plus `.claude/skills/docs-audit/` (on-demand full sweep; auto-fixes mechanical issues like stale dates/numbering/dead links, flags semantic overlap/staleness for human judgment).

---

## 2026-07-17 — F10 collapse mechanism fixed; residual gap reclassified, not resolved

23. **FedKD SVD rank-starvation fix (Decision 21) is confirmed working — but the
    residual accuracy gap is a separate, still-open finding.** Formal 50-round
    smoke re-run on the width-0.5 MobileNetV2GN student (Decision 22), both
    heterogeneity arms: α=0.1 peak 30.10% (R45, final 26.41%), α=1.0 peak 38.31%
    (R49, final 36.29%) — both well above the 10% chance line, and
    `mean_rank_retained` held at the 0.278 floor throughout (vs. a same-model
    minitest A/B's 3.7% unfixed baseline). **Do not compare these figures
    against the retired pre-fix numbers (20.80%/33.62% peak)** — those were
    measured on the old SimpleCNN student and are architecture-confounded with
    this run; the causal evidence for the fix is the same-model minitest, not
    a pre/post delta across the student swap. **FedKD is unblocked for
    comparison tables** — the near-chance collapse that made it unusable is
    gone. But `mean_rank_retained` sitting _at_ the floor rather than climbing
    past it means the original audit's candidate 3 (SVD too lossy for
    depthwise-separable weights) is still live and is the leading explanation
    for the remaining 15–27pp gap vs. every other baseline (FedAvg, FedProx,
    FedMAQ, DAdaQuant, FedPAQ) — report this as an open finding, not a closed
    investigation, until it's contextualized against the other four KD
    baselines (F13, still run-gated). Uploaded comm rose to 589–740 MB
    (floor-mandated cost of retaining more singular vectors) — still ~12×
    cheaper than FedAvg's uncompressed ~8.5 GB baseline.
    Logs: `outputs/2026-07-16/23-36-50/` (α=0.1), `outputs/2026-07-17/10-20-23/`
    (α=1.0). Full analysis: `docs/audits/archive/distillation-direction-audit.md` F10.

    _(Note for `docs-audit`: decisions 14–17 above are numbered out of the file's
    append order — a pre-existing drift this entry did not introduce or fix.)_

---

## 2026-07-17 — FedMD digest-epoch hyperparameter trimmed for future runs

24. **`conf/algorithm/fedmd.yaml` `public_epochs` (digest phase) reduced from
    5 to 3, effective for future runs only — the in-flight F13 minitest
    (task `b4m2fgbfy`) kept the original value of 5.** FedMD is by design the
    heaviest baseline: a one-time 20-epoch (10 public + 10 private) pretrain
    per client on first contact, plus every round a digest (`public_epochs`,
    alignment to the server's averaged public logits) and a revisit
    (`local_epochs=5`, private cross-entropy) phase. The digest phase recurs
    every round for every sampled client, unlike the one-time pretrain,
    making it the dominant lever on total wall-clock (observed ~300–500s/round
    at the default scale, escalating as more distinct clients hit
    first-contact pretrain — see `outputs/2026-07-17/11-10-46/`). The
    original FedMD paper (Li & Wang, 2019) uses a lightweight 1-epoch digest
    step; going straight to 1 was rejected because earlier informal runs at 1
    epoch stalled at chance-level accuracy (insufficient alignment signal per
    round). 3 is a compromise: closer to the paper's intent than 5, without
    the observed stall-at-guessing failure mode at 1. Pretrain epochs (10+10)
    left unchanged — one-time cost per client, not the recurring bottleneck,
    defensible as "train to convergence" per the paper. Apply the new value
    on any future FedMD rerun (including a formal grid run); do not compare a
    future run's per-round curve directly against the in-flight F13 minitest
    numbers without noting the digest-epoch difference.

---

## 2026-07-17 — FedMD dropped from the formal baseline stack

25. **FedMD removed from the 8-baseline stack (now 7: FedAvg, FedProx, FedPAQ,
    DAdaQuant, FedDistill, FedKD, CFD) + FedMAQ.** Supersedes Decision 24's
    digest-epoch trim, which is now moot. Root cause (session 2026-07-17,
    `outputs/2026-07-17/11-10-46/`): FedMD's one-time 20-epoch (10 public + 10
    private) transfer-learning pretrain per client, combined with
    `client_gpus=1.0` forcing **serial** Ray actor execution (only 1 actor
    constructed regardless of `client_fraction`), meant ~90 distinct clients
    each paid the pretrain cost sequentially over the course of the grid.
    Wall-clock was tracking ~7-8.5 min/round even after the Decision 24 digest
    trim (5→3 epochs), projecting 6+ hours for a single 50-round smoke arm —
    infeasible to reproduce reliably across a 3-seed formal grid. A
    convergence-based pretrain-stop (train-loss plateau, patience=5, Δ=0.5%
    rel, min=5, cap=100 per phase) was scoped as an alternative but rejected:
    a cap of 100 can run _longer_ than the original fixed 10/10 in the worst
    case, directly fighting the feasibility goal, and the added plateau-
    detection code path itself needs implementation + validation — the exact
    overhead the user was trying to escape. Considered and rejected keeping
    FedMD via a fixed higher epoch cap or a FedMD-only reduced `num_clients`
    scope — both still require debugging/defending a heavy, paper-deviating
    baseline. **Decided: drop entirely.** The manuscript is an unfinalized
    proposal, not yet a constraint — FedMD's role as the "public-labeled-
    dataset privacy assumption" comparison point (`chapter_2.tex:317`) and its
    mention in the analytical cost model (`chapter_4.tex:255`) will be revised
    to reflect a 7-baseline scope rather than treated as a fixed requirement.
    FedDistill remains the sole Pure KD baseline (also prediction/logit-based,
    covers the "no full-weight-sharing" comparison axis, at a fraction of
    FedMD's cost — no persisted pretrain, plain `local_epochs=5` per round).
    **Not deleted:** `conf/algorithm/fedmd.yaml` and
    `src/fedmaq/core/client_hooks/fedmd.py` retained (marked dropped in both
    files/registries) in case FedMD is revisited later — same pattern as the
    FedMAQ-Lite retirement (Decision 4). `.claude/project/baseline_registry.md`,
    `.claude/rules/baselines.md`, and `docs/audits/baseline-status-audit.md`
    updated to match. F13 re-scoped to the remaining 3 unmeasured KD baselines
    (FedDistill, CFD, FedAvg+KD × α∈{0.1,1.0}); relaunched via new
    process-isolated runner `scripts/run_kd_baselines_smoke.py` (task
    `bg7fu46cg`), same config profile as the aborted FedMD run
    (`experiment=default`, `total_rounds=50`, `num_clients=100`).
    **Separately parked, not part of this decision:** lowering `client_gpus`
    below 1.0 to allow concurrent client actors (would cut wall-clock for
    _every_ baseline, not just FedMD) — user has hit OOM with this before and
    shares the GPU with non-training workloads; needs a dedicated profiling
    pass before any global default change.

---

## 2026-07-17 — CFD dropped from the formal baseline stack

26. **CFD removed from the baseline stack (7 → 6: FedAvg, FedProx, FedPAQ,
    DAdaQuant, FedDistill, FedKD) + FedMAQ.** Closes distillation-audit F15
    (CFD collapses to chance both α — see `docs/audits/archive/distillation-direction-audit.md`
    F15) as **structural, not fixed**. This corrects the F15 write-up's original
    mechanism guess: the collapse is real (confirmed at both toy and
    production scale this session) but the root cause is **not** the
    client-side aggregation codec/ordering the audit suspected — that code
    (`constrained_quantize`, `dequantize`, the shuffle-before-SGD fix in
    `_train_server_model`) was probed directly this session and is correct.
    Empirical diagnosis (three isolated repro runs, `scripts/run.py
algorithm=cfd` with instrumented per-round vote/loss/prediction-histogram
    logging, since reverted): 1. **Server-side dual distillation is exonerated.** A discriminator run
    (5 clients @ full participation → healthy 36–45% client-vote consensus,
    `server_distill_epochs` raised 1→20) showed the server model tracks its
    targets correctly (`server_on_public_acc` climbed 18%→38%→36% alongside
    `targets_acc` 45%→46%→38%) once given enough gradient steps. An
    earlier same-session reading that called this a _server_-side mode
    collapse was itself a toy artifact (100 public samples ÷ batch 64 = 2
    gradient steps/round — undertraining, not a bug) and is retracted. 2. **The real defect is upstream, at production client-count scale.**
    Rerun at 50 clients / `client_fraction=0.1` (matching the smoke's
    ~100-client, low-participation regime) reproduced the audit's
    production symptom directly: `targets_acc` pinned near chance
    (14/10/16/16% across 4 rounds), with individual clients one-hot-voting
    the _same single class_ for all 100 public samples from round 1
    onward. Root cause: each client's private partition is tiny at this
    scale (~470 samples at 100 clients on CIFAR-10's 50k-image train set
    minus the 3000-sample public reserve) and 5 local CE epochs from a
    fresh/reset init is enough to overfit to 1–2 dominant local classes —
    healthy _local_ train accuracy (50–65%, matching the smoke's
    `client/avg_train_acc`) coexists with near-random generalization to
    the disjoint, class-balanced public set. CFD's 1-bit (`b_up=1`)
    constrained quantization then forces each client's vote to **full
    commitment to that one class** with zero soft/hedged signal, unlike
    the other KD baselines' temperature-scaled soft-probability averaging
    — so a few overfit voters dominate the round's consensus outright. 3. **Raising the vote bit-width does not rescue it.** Tested `b_up=b_down=4`
    (16 quantization levels, i.e. much less forced-one-hot) at the same
    production-scale regime: `targets_acc` barely moved (14→21% across 4
    rounds, still chance-adjacent) because the underlying client
    _prediction_ is wrong, not merely imprecisely encoded — more
    quantization levels just transmit the same bad prediction more
    precisely. Rules out bit-width as a config-only fix.
    **Considered and rejected:** raising `client_fraction` to dilute
    degenerate votes (untested, and clients still individually overfit
    regardless of how many are sampled — no strong reason to expect it fixes
    the per-client generalization failure); code-level mitigations (fewer
    local epochs for CFD only, regularization, skipping round 1's contribution
    to persistent server distillation) — all touch shared hyperparameters or
    need their own validation pass, and none address the core mismatch
    between CIFAR-10's per-client data budget at 100 clients and a 1-bit
    hard-vote protocol. **Decided: drop CFD from the formal grid, same
    disposition as FedMD (Decision 25).** Config/hook code retained
    (`conf/algorithm/cfd.yaml`, `src/fedmaq/core/{strategy_hooks,client_hooks}/cfd.py`,
    `src/fedmaq/core/softlabel_codec.py`) for reproducibility; CFD moves to an
    exploration-appendix note (collapse is real and diagnosed, not a
    baseline result) rather than a formal comparison-table entry.
    `.claude/project/baseline_registry.md`, `.claude/rules/baselines.md`, and
    `docs/audits/{baseline-status-audit.md,distillation-direction-audit.md}`
    updated to match. Hybrid Q+KD group is now FedKD-only in the formal stack.

---

## 2026-07-18 — Priority 1 Exploration Campaign Scoped (grilling session)

Resolves the deferred process questions in [formal-experiment-plan.md](plans/formal-experiment-plan.md) §2–§3 (mechanism sweep order, decision rule, baseline-tuning budget). Does not resolve the mechanisms themselves (capacity-EMA on/off, Formulation 3, etc.) — those await Pass 1–3 results.

27. **Explore-α = 0.3** for all Priority 1 exploration-phase runs. Distinct from the confirmatory report grid {0.1, 1.0} (Decision 10) so the single frozen config isn't selected on the exact α values it will later be reported on. `conf/heterogeneity/dirichlet_alpha_0.3.yaml` added.
28. **Exploration run budget**: 50 rounds, single seed (seed=0), per sweep run — matches prior smoke-test convention, cheap enough for repeated passes. Confirmatory grid (multi-seed, 100+ rounds, Decision 9) is unaffected; exploration only picks direction.
29. **Mechanism sweep structure**: grouped into passes rather than one full joint factorial (cost) or pure sequential coordinate-descent (misses interactions). Every mechanism setting includes its control/off arm.
    - Pass 1: soft-voting weights (`entropy_weight` × `precision_weight`), joint — coupled by design.
    - Pass 2: capacity-EMA on/off, grad-norm-smoothing (β=0.7) isolation, client-KD-reg+proximal (μ) — grouped, largely orthogonal.
    - Pass 3: Formulation 3 (dual-tier precision scaling) — still optimal at this capacity?
    - FedMAQ mechanisms fully resolved (all 3 passes) before baseline matched-tuning starts — baseline HPs don't depend on FedMAQ's mechanism choices, so sequential ordering costs nothing and avoids redoing baseline tuning if a later pass changes the FedMAQ config.
30. **Decision rule per mechanism pass**: keep/drop/revise only if the delta clears a noise margin, not just whichever setting scores highest in a single-seed run. Single-seed exploration has no variance estimate, so picking the literal best risks chasing noise.
31. **Baseline matched-tuning budget**: one key HP each (FedProx μ, FedPAQ bit-width, DAdaQuant schedule, FedDistill/FedKD distillation temp — FedMD/CFD already dropped, Decisions 25/26), grid capped at 5 values per baseline, matched to FedMAQ's own per-mechanism sweep run count (equal-budget parity, Decision 7).
32. **F9 (code-audit, client-KD-reg deepcopy) closed WONTFIX, not fixed.** Re-examined ahead of Pass 2 since it gated `client_kd_reg=true`. A cross-round teacher-shell cache (keyed by client partition-id, to survive `client_fn` re-instantiation) was prototyped and reverted: it keeps a GPU-resident model copy alive per client per Ray worker for the whole run, which is exactly the Ray/PyTorch VRAM-accumulation class the process-isolated runners (`hydra-config.md`, `flower-patterns.md`) exist to prevent, and cache hit-rate isn't even guaranteed (flwr sim doesn't pin partitions to actors). The cost being optimized is ~9MB copied once per client per round — negligible against per-round GPU training time. `client_kd_reg=true` runs as-is in Pass 2; no code change needed. `docs/audits/fedmaq-code-audit.md` F9 entry updated to WONTFIX.

---

## 2026-07-18 — Priority 1 Pass 1 Results (soft-voting sweep, MobileNetV2GN)

All 18 runs (2 ablation + 16-cell `entropy_weight` × `precision_weight` grid, explore-α=0.3, 50R single-seed) completed in `multirun/2026-07-18/03-30-59-soft-voting-explore-mobilenetv2/`. Run 17 (ew=4.0, pw=4.0) needed 5 relaunches to finish due to recurring Windows Ray crashes (SIGSEGV/ActorDiedError, per `flower-patterns.md`'s documented instability class, not algorithm-related); its `experiment_log.csv` had duplicate rows from the retries (round-0 x5) and run 14 had 3 duplicate rows (rounds 0-2) — both hand-deduped in place (kept first occurrence per round; not committed to git, no backup taken — WandB should hold the authoritative copy per `evaluation-metrics.md`).

Round-50 top-1 accuracy: sweep-grid (16 cells) ranges 0.4601-0.5179 (median 0.4907); 14/16 cells cluster tightly in 0.4838-0.4996 (~1.6pp band). Ablation: sv_on 0.4920 vs sv_off 0.4796 (+1.24pp, sv_on favored but inside the cluster band).

33. **No numeric noise margin was ever sourced for Decision 30's rule** (single-seed runs give no variance estimate, and no prior repeated-seed characterization on MobileNetV2GN exists). Resolved pragmatically: the empirical cluster band (~1.6pp) observed across the 14 non-outlier sweep cells stands in as the noise floor for this pass only.
34. **Pass 1 tentative pick: `entropy_weight=2.0`, `precision_weight=0.5`, `soft_voting=true`** (run index 10, 0.5179 top-1, +1.6-3.4pp over the cluster band — clears the empirical margin). **Flagged provisional**, not frozen: single-seed result, no repeated-seed confirmation yet. User has agreed to a future multi-seed re-verification pass (repeat idx10 plus 1-2 other cells with 2-3 more seeds) before this is treated as load-bearing for the confirmatory grid.
35. Run 17 (ew=4.0, pw=4.0, 0.4601, the sweep floor) is flagged lower-confidence than the other 17 runs given its rocky completion history (5 crash-retries, post-hoc CSV dedup) — not excluded, but not to be over-interpreted as a clean "high ew+pw hurts" signal without a rerun.

---

## 2026-07-22 — Hardware & Telemetry Grounding (Late-2023 Coherent Era)

Resolves telemetry calibration for physical execution time and communication energy modeling (§4.1, §4.3). Replaces former placeholder constants (`bandwidth_mbps: 10.0`, `compute_samples_per_sec: 200.0`, `server_compute_speed: 2000.0`) with a mathematically grounded, temporally aligned Late-2023 hardware ecosystem (see [docs/adr/0002-hardware-telemetry-grounding.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/adr/0002-hardware-telemetry-grounding.md)).

36. **Late-2023 Hardware Standardization**:
    - **Edge Clients**: **Raspberry Pi 5 Series** (launched Oct 2023, Broadcom BCM2712 Quad Cortex-A76 @ 2.4 GHz). Memory sampling $c_k \sim \mathcal{U}(2048, 16384)$ MB with $c_{\text{unit}}=512$ MB maps to Pi 5 RAM variants (2GB $\rightarrow$ 4-bit, 4GB $\rightarrow$ 8-bit, 8GB $\rightarrow$ 16-bit, 16GB $\rightarrow$ FP32).
    - **Wireless Link**: `bandwidth_mbps: 10.0` based on integrated Dual-Band 802.11ac Wi-Fi® under edge channel contention and distance path loss.
    - **Client Compute Speed ($v_{\text{client}}$)**: Grounded in Quad Cortex-A76 @ 2.4 GHz **sustained** FP32 throughput (~18.0 GFLOPS, ~57% of 31.5 GFLOPS peak — see Decision 37):
      - **CIFAR-10/100 (`MobileNetV2GN`)**: `compute_samples_per_sec: 20.0` (based on $0.90\text{ GFLOPs/sample}$ fwd+bwd).
      - **FEMNIST (`SimpleCNN`)**: `compute_samples_per_sec: 600.0` (based on $0.015\text{ GFLOPs/sample}$ fwd+bwd, capped for DataLoader overhead).
    - **Central FL Server**: **High-Density Enterprise Server Node** (launched Late 2023: 24-Core Intel Xeon 5th Gen Emerald Rapids CPU, 1× NVIDIA L40S 48GB Universal GPU, 64 GB DDR5-4800 ECC System RAM).
    - **Server Compute Speed ($v_{\text{server}}$)**: Per-dataset (see Decision 37): `5000.0` s/s (CIFAR MobileNetV2GN), `10000.0` s/s (FEMNIST SimpleCNN).

---

## 2026-07-22 — Sustained-Throughput Telemetry Revision & Per-Dataset Server Speed

Supersedes the peak-GFLOPS-based values in the original Decision 36. Full derivation chains in [ADR-0002](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/adr/0002-hardware-telemetry-grounding.md).

37. **Sustained-throughput revision (client + server)**:
    - **Client 35.0 → 20.0 s/s** (CIFAR MobileNetV2GN). Peak FP32 GEMM on Pi 5 is ~31.5 GFLOPS; sustained efficiency is ~57% ($P_{\text{sustained}} = 18.0$ GFLOPS) due to PyTorch eager-mode dispatch overhead, Python GIL contention, non-optimized ARM NEON codegen, and LPDDR4X memory bandwidth sharing. $v_{\text{client}} = 18.0 / 0.90 = 20.0$ s/s. FEMNIST SimpleCNN stays at 600 s/s (DataLoader-capped: theoretical 1,200 s/s > 600 cap, so compute speed is not the binding constraint).
    - **Server 4500 → 5000 s/s** (CIFAR) / **10,000 s/s** (FEMNIST). Derived from L40S memory-bandwidth roofline ($B_{\text{mem}} \times I_{\text{arith}} = 864 \times 22 = 19{,}008$ GFLOPS effective) × 8% framework efficiency (kernel-launch-limited regime for small 2.24M-param model on 48GB GPU) / $F_{\text{fwd}}$. SimpleCNN has ~25 CUDA kernels per forward pass (vs MobileNetV2GN's ~120), yielding ~2× faster per-batch processing in the overhead-limited regime.
    - **Per-dataset server speed**: `server_compute_speed` is now resolved via `resolve_server_compute_speed()` (experiment config → algorithm config → module fallback). FEMNIST experiment config overrides the algorithm default (5000 → 10000) so a single `experiment=femnist` is sufficient.
    - **Impact**: Client training time increases from 67s to 118s per round (CIFAR), making computation even more dominant relative to communication (~8:1 ratio). This is **conservative for thesis claims** — the real-world advantage of FedMAQ's communication compression is at least as large as simulated. Server time has <5% sensitivity on round time.
    - **Manuscript note**: §4.1 and §4.3 will be updated to reflect these values once the formal experiment configuration is frozen and pre-registered.

---

## 2026-07-22 — Quantization Precision Bounds Documented as Intentional

38. **`q_max=16` and `bit_widths=[1,2,3,4,5,6,7,8,16,32]` formally documented**:
    - **`q_max=16`**: The Tier-2 soft quality target interpolation range is $[1, 16]$. The quality formulation never assigns FP32 precision — FP16→FP32 gains are marginal for FL accuracy (local SGD gradient noise exceeds FP16 quantization noise) while doubling communication cost. Consequence: 8GB and 16GB Pi 5 clients are functionally identical in achievable precision (both cap at 16-bit). Intentional, not an oversight.
    - **`bit_widths` set**: The gap between 8 and 16 (no 9–15 bit options) is **hardware-aligned** — real quantization formats with silicon support are power-of-2 (INT4, INT8, FP16, FP32). Fine granularity at 1–8 bits captures the most impactful precision range for resource-constrained clients; the 8→16→32 jumps reflect the physical hardware landscape. Standard practice in mixed-precision quantization literature (HAWQ, HAQ, MBQ). Adding 9–15 would introduce non-standard bit-widths for negligible accuracy gain while complicating the precision-allocation narrative.

---

## 2026-07-22 — Canonical Experiment Taxonomy & Declarative Matrix Runner Infrastructure

39. **Standardization of Output Paths & Declarative Matrix Sweeps**:
    - **Single Canonical Output Root (`outputs/`)**: All future experiment artifacts land strictly under `outputs/<phase>/<dataset>_<model>/<exp_group>/<algorithm>/<heterogeneity>/seed_<seed>/` (eliminating path ambiguity between `experiments/`, `multirun/`, and `outputs/`).
    - **Phases & Standardized Round Counts**:
      - `ci`: Fast integration checks ($R = 2$).
      - `smoke`: Short validation sweeps ($R = 50$, single seed).
      - `explore`: Mechanism & hyperparameter exploration ($R = 50$, single seed).
      - `formal`: Confirmatory thesis runs ($R = 100$, 3 seeds: 0, 42, 123).
    - **Declarative YAML Matrices (`conf/matrix/*.yaml`)**: Replaced 17 single-purpose ad-hoc Python runner scripts with declarative YAML manifests (`ci_test.yaml`, `mobilenetv2_smoke_50r.yaml`, `pass2_explore.yaml`, `benchmark_grid.yaml`).
    - **Unified Execution Driver (`scripts/run_matrix.py` + `scripts/common.py`)**: Centralized cross-platform Ray cleanup (`kill_ray_processes()`), process-isolated execution, dry-run support (`--dry_run`), and start-index resumption (`--start_at N`).
    - **Script Cleanup**: Obsolete ad-hoc `run_*.py` scripts deleted from `scripts/`.

---

## 2026-07-23 — Step 2 Golden-Diff Gate Confirmed Bit-Exact (Architecture Deepening)

40. **Architecture Deepening Step 2's golden-diff gate is literal bit-exact — no tolerance-based fallback needed**:
    - During Step 1 (telemetry seam) validation, re-running an identical seeded `ci_test` config appeared to produce varying `test/loss`/`test/accuracy` between runs, suggesting GPU/cuDNN non-determinism would block Step 2's planned "bit-exact golden diff per baseline" gate. Three fallback designs (force determinism, CPU-only validation, tolerance-based gate) were drafted as a result.
    - Directly re-tested before committing to any of those: ran `experiment=ci` for both `fedavg` and `fedmaq` (the exact pair implicated) twice each, as independent processes, same seed, on GPU. Diffed the resulting `experiment_log.csv` files. Result: `test/loss`, `test/accuracy`, precision/recall/f1, communication bytes, simulated `system/*_time` columns, and FedMAQ-specific metrics were **byte-identical** across both reruns for both algorithms. Only `wall_time_sec` (real wall-clock, never a golden-diff candidate) differed.
    - **Why the earlier observation was wrong**: not reconstructed in detail, but the existing `strict_determinism` infra (`fedmaq.simulation.set_seed`/`configure_torch_determinism` — seeded Python/NumPy/torch/CUDA, `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=False)`, seeded `DataLoader` workers, per-Ray-worker reseed in `client_fn`) was already in place at Step 1 and is sufficient on its own; the Step 1 validation comparison itself was the artifact, not the platform.
    - **Consequence**: Step 2's golden-output harness should implement literal bit-exact comparison (old code vs. new code, same seed) over all CSV columns except `wall_time_sec`. No determinism-forcing work, no CPU-only fallback, and no tolerance thresholds are needed — the infrastructure already delivers this.
    - **How to apply**: if a genuine non-determinism reappears during Step 2 (e.g. a baseline the ci_test probe didn't cover), re-run the same two-independent-runs diff first to confirm it's real before reaching for a tolerance-based design — don't assume GPU training is inherently non-reproducible in this codebase.

---

## 2026-07-23 — Architecture Deepening Complete: Client Training Skeleton (Step 2 / Candidate A)

Closes `docs/plans/architecture-deepening.md` — all steps (B, A, C; D explicitly deferred) are now resolved, so the plan file is deleted per `docs-management.md` (plans are active-only; this entry is the historical record).

41. **Client training skeleton narrowed to two pieces, not one broad class — `run_epochs` + `compress_and_reconstruct` — per ADR-0003** (`docs/adr/0003-training-skeleton-seam.md`). Grilling session (2026-07-23) found the plan's original "one `TrainingSkeleton` shared by all five baselines" framing only actually fit `standard`+`fedkd` (the only two with a delta→compress→reconstruct tail); `feddistill`/`cfd`/`fedmd` have 1/2/4 single-model batch loops respectively with no compression tail at all. Final seam: `run_epochs(model, loader, optimizer, epochs, step_fn, device, on_after_backward=None) -> AggregatedMetrics` (single-model batch-loop atom) and `compress_and_reconstruct(original_params, updated_params, compressor_hook) -> (reconstructed_params, byte_size)` (delta tail, shared only by `standard`+`fedkd`). `fedkd`'s joint student+teacher optimizer keeps its own hand-rolled loop — forcing it onto `run_epochs` would have required the atom to support multi-model joint optimization for a single caller, failing the deletion test.
42. **S2a landed**: `standard.py` (all 6 branches — FedAvg, FedProx, FedPAQ, FedAvgKD, DAdaQuant, FedMAQ) migrated onto both `run_epochs`+`compress_and_reconstruct`; `fedkd.py` migrated onto `compress_and_reconstruct` only. All 7 configs verified bit-exact via `scripts/golden_diff.py` against pre-refactor golden output (Decision 40's gate).
    - **Harness bug found and fixed during this step, not a code regression**: `fedkd`'s golden-diff initially failed (21 mismatched columns) despite `fedkd.py` being untouched by the `standard.py` migration. Root cause: `fedkd`/`fedmd` persist client/teacher state to disk (`.data_partitions/fedmd_models/`, per `baselines.md`) keyed only by client-id, not by run — the harness's capture and compare runs were silently inheriting each other's trained weights instead of each starting cold. Fixed by wiping `PERSISTENCE_DIR` at the top of every `scripts/golden_diff.py` `_run()` call, both capture and compare. Re-ran with the fix: `fedkd` passes bit-exact, confirming the failure was harness contamination, not a defect.
43. **S2b landed**: `feddistill.py`, `cfd.py`, `fedmd.py` migrated onto `run_epochs` only (called 1/2/4 times respectively); each hook's own upload-assembly code (raw weights+logits / quantized soft-label codes / public-set predictions) left untouched. All 3 verified bit-exact.
    - **Non-obvious constraint surfaced during this step**: several of these hooks accumulate a single running loss/accuracy average across *multiple* sequential loop phases sharing one accumulator (e.g. FedMD's `priv-pretrain` + `revisit` phases both feed the same `loss_sum`/`batches`; CFD's/FedMD's per-epoch-reset batch offsets into a server-provided target array). Reconstructing a combined average by multiplying-then-adding two independently-computed `run_epochs()` averages would reassociate the underlying floating-point sum and risk breaking the bit-exact gate (floating-point addition is not associative). Resolved by keeping the raw accumulator variables external to `run_epochs` in those cases (closures mutate them directly, batch-by-batch, in the exact same order as the original single-accumulator code) and using `run_epochs` purely as the batch-loop mechanic, ignoring its own internal averaging for these specific merges. Phases with no cross-call combination (CFD's two independently-reported phase averages, FedDistill's single loop) safely use `run_epochs`' own `AggregatedMetrics` directly.
44. **Gate outcome**: all 10 golden-set configs (fedavg, fedprox, fedpaq, fedavg_kd, dadaquant, fedmaq, fedkd, feddistill, cfd, fedmd) pass bit-exact against pre-refactor golden output; full test suite green (111 passed). Step 2 — and with it the whole architecture-deepening plan (Step 1 done 2026-07-23 morning, Step 3/Candidate C closed via ADR-0001, Candidate D explicitly deferred past thesis) — is complete.

---

## 2026-07-24 — FedMD Excluded from Default Golden-Diff / Smoke Sweeps

45. **`fedmd` dropped from `scripts/golden_diff.py`'s default `GOLDEN_SET`** and from smoke-test matrices going forward. It remained in Step 2's golden set (Decision 44) purely for regression coverage of that migration; it was already dropped from the formal experiment grid (Decision 25) and, per that decision, is unlikely to reappear in future experiments. It is also by far the slowest config in the set — its disk-persisted multi-phase training (pub/priv pretrain + digest + revisit, up to 4x `run_epochs` per round) took ~9 minutes for a 2-round `experiment=ci` config alone during Candidate A's golden-diff run, versus under a minute for most other baselines. **Rationale**: paying that cost on every golden-diff/smoke run going forward has no payoff for a baseline not being iterated on. **How to apply**: `fedmd.py`'s code and its `run_epochs` migration (Decision 43) are untouched and remain correct; only re-add `fedmd` to `GOLDEN_SET` (or a smoke matrix) for a change that actually touches its code path (`fedmd.py`, or the FedMD branch of `kd_utils.py`/persistence helpers). See `.claude/rules/baselines.md`.

---

## 2026-07-24 — Architecture Deepening, Candidate A Complete: QuantizationPlanner

Closes the architecture-deepening candidate this session named "A" (Decision 20's P4/P5 — pulling FedMAQ's quantization policy out of the hook into its own seam).

46. **`QuantizationPlanner` + `QuantPlan` + `RunContext` + shared `inject_client_q` landed and verified bit-exact.** New module `src/fedmaq/core/quantization_planner.py` owns the quantization policy previously embedded in `FedMAQHook`: `QuantizationPlanner` (holds `_grad_norm_model`/`_grad_norm_ema` state, exposes `plan_round(...)`), `QuantPlan` (frozen dataclass: `client_q` + `grad_norms`), `_QuantParams`, and `compute_fedmaq_q_k_t` (moved verbatim). `inject_client_q` is a shared FitIns-rewrite helper now used by both `fedmaq.py` and `dadaquant.py` (folding in the candidate previously called "G" — the two hooks no longer duplicate FitIns-rewriting logic). `RunContext` (frozen dataclass: `dataset_name`/`num_classes`/`batch_size`/`device`/`alg_cfg`) + `resolve_run_context(config)` added to `config_defaults.py` so the planner doesn't need to reach into the raw Hydra config. `FedMAQHook` is now orchestration only: `configure_fit` resolves a `RunContext` → calls `planner.plan_round(...)` → `inject_client_q`; `aggregate_fit`/`get_eval_metrics` read a single `self._current_plan` field, replacing the old three-field spread (`_round_client_q`/`_last_grad_norms`/`_last_assigned_q`).
    - **Circular import found and fixed during implementation**: `quantization_planner.py`'s original module-level import of `strategy_hooks._partition` triggered `strategy_hooks/__init__.py`, which imports `fedmaq.py`/`dadaquant.py`, which import back into `quantization_planner.py` mid-initialization. Fixed by making that import lazy (moved inside `QuantizationPlanner._probe_grad_norms`, its only caller) — not a design change, just an import-order fix.
    - **Gate outcome**: all 9 configs in `scripts/golden_diff.py`'s post-Decision-45 `GOLDEN_SET` (fedavg, fedprox, fedpaq, fedavg_kd, dadaquant, fedmaq, fedkd, feddistill, cfd) pass bit-exact against the pre-Candidate-A golden output, including the two configs the change actually touches (`fedmaq`, `dadaquant`). Full test suite green (111 passed) after updating `tests/test_environment.py`'s `DEFAULT_BIT_WIDTHS` import path and `tests/test_refinement_features.py`'s hook-field assertions (`hook._current_plan.client_q` / `hook._planner._grad_norm_ema` / `hook._planner._smooth_grad_norms`) to match the new seam.
    - **How to apply**: any future change to FedMAQ's quantization policy (bit-width assignment, gradient-norm smoothing, cost/comm modeling) belongs in `quantization_planner.py`, not in `strategy_hooks/fedmaq.py`. `strategy.py`'s re-export of `compute_fedmaq_q_k_t` now points at `quantization_planner`, not `fedmaq.py` — keep that redirect if `fedmaq.py` is touched again.

---

## 2026-07-24 — Architecture Deepening, Candidate B Complete: RunContext Adoption

47. **`RunContext`/`resolve_run_context` adopted in the two remaining sites Candidate A didn't touch: `fedavg_kd.py:aggregate_fit` and `cfd.py.__init__`.** `fedavg_kd.py` replaced its inline `dataset_name`/`num_classes`/`batch_size`/`device`/`alg_cfg` pulls (previously duplicated from `config_defaults` constants + raw `config.get(...)`) with a single `ctx = resolve_run_context(self._config)` call, mirroring `fedmaq.py`'s existing pattern; the now-unused `torch` import was removed. `cfd.py.__init__` sources `self.dataset_name`/`self.num_classes`/`self.device` from the same resolved `ctx`, leaving CFD-specific `alg_cfg` fields (`b_up`, `b_down`, `temperature`, `distill_epochs`, etc.) and the separate `batch_size` resolution inside `_get_public_loader` untouched — those aren't part of the RunContext quintuple's scope and widening it there would be scope creep, not DRY.
    - **Pre-existing, out-of-scope lint finding surfaced, not introduced or fixed**: `cfd.py`'s `server_sim_time` has an unused `alg_cfg` local (`F841`); confirmed via `git stash` that this predates Candidate B. Left as-is — fixing it isn't part of this candidate's scope.
    - **Gate outcome**: all 9 configs in `GOLDEN_SET` pass bit-exact, including the two this change touches (`fedavg_kd`, `cfd`). Full test suite green (111 passed).
    - **How to apply**: `config_defaults.py`'s `RunContext` quintuple (`dataset_name`/`num_classes`/`batch_size`/`device`/`alg_cfg`) is now the single resolution point across `fedmaq.py`, `dadaquant.py` (via Candidate A), `fedavg_kd.py`, and `cfd.py`. Any new KD-ish hook needing these should call `resolve_run_context(config)` rather than re-deriving them inline.

---

## 2026-07-24 — Architecture Deepening, Candidate C Complete: PhysicalCostModel

48. **`NetworkSimulator` renamed and deepened into `PhysicalCostModel`, absorbing the array-build previously scattered across `TelemetryFedAvg.__init__`.** `strategy.py`'s old `NetworkSimulator` (interface ≈ implementation — 4 ndarray fields, 3 divisions, a bare 3-tuple return) is now `PhysicalCostModel`: a new `from_config(config, num_clients)` classmethod owns bandwidth/compute/memory array construction (previously ~50 lines inline in `TelemetryFedAvg.__init__`, lines 100-148 pre-refactor), and `simulate_client_delay` is renamed `client_round_delay`, returning a `ClientDelay(NamedTuple)` instead of a bare tuple (still unpacks as `t_download, t_train, t_upload = ...` for existing call sites, so `telemetry.py`'s `record_fit_round` needed no destructuring changes beyond the method-name rename). `TelemetryFedAvg.__init__` now holds a single `self.cost_model = PhysicalCostModel.from_config(config, self.num_clients)`; `self.client_memory` (read externally by `fedmaq.py`'s hook) becomes `self.cost_model.client_memory`, and `self.network_simulator.simulate_client_delay(...)` (called from `telemetry.py`) becomes `self.cost_model.client_round_delay(...)`. Per-hook time-model contributions (`base.py:125-129` — `download_size_bytes`, `server_sim_time`, `local_train_sample_count`, `compute_speed_scale`) are **untouched**, per the design's explicit "don't touch the per-hook time methods" constraint — this candidate deepens the physics-simulation seam, it doesn't touch the per-algorithm dispatch seam ADR-0002 already settled.
    - **Test surface updated, not just source**: `tests/test_environment.py::test_network_simulator` and `tests/test_timing_golden.py` construct the class directly with the plain 4-arg constructor (`upload_bw, download_bw, comp_speed, num_clients=1`) for physics-only golden numbers, bypassing `from_config` — kept that constructor shape unchanged (all new fields default to `None`/`""`) specifically so these tests only needed a name rename (`NetworkSimulator`→`PhysicalCostModel`, `simulate_client_delay`→`client_round_delay`), not a rewrite. `test_environment.py`'s FedMAQ memory-cap test patches `strategy.cost_model.client_memory` instead of `strategy.client_memory`.
    - **Pre-existing, out-of-scope lint finding surfaced, not introduced or fixed**: `telemetry.py` has an unsorted import block (`I001`, confirmed via `git stash` to predate this candidate). Left as-is.
    - **Gate outcome**: all 9 `GOLDEN_SET` configs pass bit-exact (this candidate's seam is on every algorithm's hot path, not just one or two configs). Full test suite green (111 passed).
    - **How to apply**: any future change to the physical time/bandwidth/compute/memory simulation belongs in `PhysicalCostModel` (`strategy.py`), not back in `TelemetryFedAvg.__init__` or inline in `telemetry.py`. Per-algorithm timing (compute penalties, epoch counts, server-side KD time) stays in `StrategyHook` subclasses — do not fold that dispatch back into `PhysicalCostModel`.

---

## 2026-07-24 — Baseline Validation Scope: Anchor + Directional, Not Full Paper Replication

49. **Baselines are not validated by full paper-exact replication across the board.** Reproducing every baseline's (FedAvg, FedProx, FedPAQ, DAdaQuant, FedDistill, FedKD) original paper setup exactly (architecture, hyperparameters, partitioning, compute budget) is expensive, and many of the source papers don't publish enough detail for bit-for-bit reproduction anyway. Doing this for all six baselines would be disproportionate to what a thesis needs and risks becoming its own side-project (three sessions were already spent on architecture refactoring without producing new experiment data).
    - **Chosen approach**: full-fidelity replication against published numbers for one or two well-documented anchor baselines only (FedAvg on CIFAR-10 is the natural candidate — best-documented, easiest to compare against public results). For the remaining baselines, validate *directionally* against each paper's key claims rather than absolute numbers — e.g. confirm FedPAQ/DAdaQuant reduce communication bytes with roughly the reported order-of-magnitude accuracy tradeoff, confirm FedKD (hybrid Q+KD) outperforms both pure-quantization and pure-KD baselines, etc.
    - **Why**: the goal is confidence that each baseline implementation is *correct enough* to serve as a fair comparison point for FedMAQ — not to produce a standalone reproducibility study. Directional/relative validation is standard practice in FL reproducibility work and matches that goal at a fraction of the cost.
    - **How to apply**: when validating a baseline (or when a bug is suspected in one), check the anchor baseline's absolute numbers only if it's FedAvg/CIFAR-10; for every other baseline, check that the qualitative claim from its source paper holds in our results (comm-vs-accuracy tradeoff direction, relative ordering vs. other baselines) rather than chasing exact number matches. Revisit this scope only if a specific baseline's correctness becomes genuinely contested (e.g. results contradict its paper's qualitative claim, not just its absolute numbers).

---

## 2026-07-24 — Architecture Deepening, Candidate D-live Complete: kd_distill_step Atom

50. **`kd_distill_step` extracted as a standalone atom from `kd_utils.py::run_server_side_kd`'s per-batch loop body.** New function `kd_distill_step(student_model, optimizer, kl_criterion, images, teacher_soft_preds, temperature) -> float` performs one SGD step distilling `teacher_soft_preds` into `student_model` via KL divergence (zero_grad → forward → scaled KL loss → backward → step → return `loss.item()`). `run_server_side_kd`'s loop now computes teacher soft-target weighting (the entropy/precision-weighted ensemble average, unchanged) and calls `kd_distill_step(...)` per batch instead of inlining the optimizer step. Signature, docstring, and all other logic of `run_server_side_kd` are untouched; `distill_ensemble_into_global` (the shared FedMAQ/FedAvgKD caller) is untouched.
    - **CFD half explicitly skipped**: CFD is a dropped baseline (Decision 26); `strategy_hooks/cfd.py::_train_server_model`'s own parked KD loop was read for design-consistency comparison only and was not modified — touching it would be parked-code churn with zero live benefit, per ADR-0003's "don't widen a seam for its least-similar caller" lesson.
    - **Gate outcome**: all 9 `GOLDEN_SET` configs pass bit-exact, including the two this change's call path touches (`fedmaq`, `fedavg_kd`). Full test suite green (111 passed).
    - **How to apply**: any future change to the server-side KD per-batch update rule (loss function, optimizer step shape) belongs in `kd_distill_step`, not inlined back into `run_server_side_kd`'s loop. CFD retains its own separate, unrefactored KD loop — do not unify the two without a fresh ask.

---

## 2026-07-24 — Architecture Deepening, Candidate E Complete: RoundSnapshot

51. **`telemetry.py`'s 7-field `last_*` attribute bag replaced with a typed `RoundSnapshot` frozen dataclass**, deepening (not undoing) ADR-0002's telemetry-grounding interface. `RoundSnapshot` (`round_bytes`, `round_time`, `client_time`, `server_time`, `wall_time`, `client_bytes_stats`, `round_client_metrics`, all zero/empty-defaulted) replaces `TelemetryManager`'s 7 individual `self.last_*` init attributes with a single `self._last_snapshot = RoundSnapshot()`. `record_fit_round` builds a fresh `RoundSnapshot` at the end of each round from local vars instead of assigning 7 separate attributes; on the empty-`results` early-return path, `dataclasses.replace(self._last_snapshot, round_client_metrics=round_client_metrics)` preserves the original behavior where only `round_client_metrics` resets to `{}` while the other 6 fields retain their prior round's stale values. New method `snapshot_for_round(server_round)` centralizes the round-0 zeroing previously repeated as 5 separate `... if server_round > 0 else <default>` ternaries at the `strategy.py` call site: for `server_round > 0` it returns `self._last_snapshot` as-is; for round 0 it returns a fresh `RoundSnapshot(round_client_metrics=self._last_snapshot.round_client_metrics)` — preserving the original asymmetry where `round_client_metrics` was read unguarded (no `server_round > 0` check) while the other 6 fields were round-0-zeroed. `strategy.py::TelemetryFedAvg.evaluate` now calls `tm.snapshot_for_round(server_round)` once and reads fields off the returned snapshot, replacing the 5 ternaries plus 2 separate truthiness/round guards (`client_bytes_stats` update, `round_client_metrics` update).
    - **Behavior-preservation verified, not just gated**: confirmed via repo-wide grep that no other `src/` or `tests/` code referenced the old `last_round_bytes`/`last_round_time`/`last_client_time`/`last_server_time`/`last_wall_time`/`last_client_bytes_stats`/`last_round_client_metrics` attribute names, so removing them is safe. `client_bytes_stats` defaulting to `{}` at round 0 makes `log_metrics.update(snapshot.client_bytes_stats)` a no-op, exactly matching the original explicit `if server_round > 0:` guard it replaces.
    - **Import block reordered by `ruff --fix`** in `telemetry.py` as a side effect of adding the `dataclasses` import — this touches the same block Decision 48 flagged as a pre-existing unsorted-import finding; since this candidate added a new import line into that exact block, fixing the resulting sort was in-scope (not the "leave pre-existing findings alone" case).
    - **Gate outcome**: all 9 `GOLDEN_SET` configs pass bit-exact (this seam is on `TelemetryFedAvg.evaluate`'s hot path for every algorithm). Full test suite green (111 passed).
    - **How to apply**: any future telemetry field added to the per-round fit snapshot belongs in `RoundSnapshot`, with round-0 zeroing behavior decided once in `snapshot_for_round`, not re-derived per-field at call sites. This is additive to ADR-0002, not a reopening of it — the per-algorithm dispatch seam ADR-0002 settled is untouched.

---

## 2026-07-26 — Post-Processing Regime, and What Supersedes Decisions 33-35

52. **`post_process` is a per-grid setting, not a per-algorithm one.** §4.3 gives
    the error-feedback/difference-coding/zlib pipeline to the primary
    benchmarking grid and to nothing else. It shipped `false` in every
    `conf/algorithm/*.yaml`, so the grid would have reported communication
    numbers without the pipeline it claims to run. The obvious repair — flipping
    it in `fedmaq.yaml` — is worse than the bug: that file is also the
    formulation study's config and Ablation Configuration 7, and the ablation
    inherited Configuration 7 from the primary grid. A pipeline there but not in
    the arms makes every arm two removals from its reference rather than one,
    with the damage falling on the communication axis where the ablation's claims
    live. The arms matched only because the flag was uniformly false.
    - **Resolution**: the flag is overridden per matrix file. Configuration 7 is
      dispatched by `conf/matrix/ablation.yaml` pipeline-free as the study's own
      anchor (grid 177 → 183, 7 net-new arms). `conf/matrix/benchmark_grid.yaml`
      and `conf/matrix/uniform_memory_control.yaml` override it on.
    - **The governing rule**, now stated in §4.3 so future runs inherit an answer:
      a FedMAQ run carries the pipeline if and only if the runs it is compared
      against carry it. The control arm's partner is the primary grid, so it
      does; the formulation study and every ablation arm are contrasted within
      their own sets, so they do not.
    - **How to apply**: never set `post_process` in an algorithm config. Decide it
      where the comparison is defined. Both directions are enforced in
      `tests/test_simulation.py`.

53. **Decisions 33-35 are superseded, not pending.** They recorded a
    single-seed soft-voting pass whose noise floor was an empirical cluster band
    (~1.6pp) standing in for a variance estimate nobody had, with #34 explicitly
    flagged provisional pending multi-seed confirmation. The rewritten §4.3.1
    replaces that whole procedure: σ is now measured from the unrefined cell at
    three seeds and the keep-or-drop threshold is √2σ, implemented in
    `scripts/analysis.py:exploration_noise_margin` and dispatched by
    `conf/matrix/pass2_factorial.yaml`.
    - The three refinement flags' shipped values are therefore an **exploration
      outcome not yet measured**, not a decision awaiting a flip. Exploration has
      not run. Read `fedmaq.yaml`'s current flag values as defaults, not as
      findings, and do not cite Decision 34's `entropy_weight=2.0` /
      `precision_weight=0.5` pick as settled.

## 2026-07-26 — Run Identity, and the Formulation Study Gets a Matrix

54. **A run is identified by the config file it composed and the matrix that
    dispatched it, never by `algorithm.name`.** Every §4.3.7 ablation arm declares
    `name: fedmaq` so they all dispatch the same strategy hook, which means
    `algorithm.name` names the *hook* and collapses six of the seven net-new arms
    onto one value. `scripts/analysis.py` keyed on it, so §4.3.7's table could not
    be built at all — and, worse, the arms sit on the winning formulation at the
    reported skews and seeds, so `select_winner` counted each of them as a
    formulation-study candidate and `compare_to_baselines` let one silently stand
    in for FedMAQ (its `{seed: run}` maps are dict comprehensions; last write
    wins, ordered by directory glob). `RunRecord` now carries `algorithm_config`
    (Hydra's `runtime.choices.algorithm`, read from `.hydra/hydra.yaml`),
    `experiment_group` and `phase` (segments of the canonical output path), and
    `post_process`. Cross-study analyses filter on the group, not the algorithm:
    Configuration 6 is `fedavg_kd` and Configuration 7 is `fedmaq`, both names the
    benchmark grid also uses, so an algorithm-level filter cannot separate them.
    `manifest.py`'s `_algorithm_config_name` field, never populated by anything
    since it was added, now carries the same value — §5.4 asks the reader to
    confirm arm parity from the run manifests.

55. **`build_ablation_table()` assembles §4.3.7's table, and refuses to report it
    when the design does not hold.** Eight configurations at both skews, accuracy
    and cumulative MB as mean ± seed SD, the per-arm formulation column so
    Configuration 4's fallback is visible in the table rather than only in the
    prose, and the pipeline regime as a note. `parity.attributable` goes false on
    a pipelined arm, a refinement-layer deviation not recorded as inapplicable, a
    missing seed, or a fallback arm whose formulation-study anchor does not exist
    yet. That last check is what §4.5's calendar ordering exists to guarantee, now
    verified rather than assumed.

56. **The formulation study is dispatched by `conf/matrix/formulation_study.yaml`,
    not a hand-typed `--multirun`.** It had been a comment in `conf/config.yaml`,
    which put its 30 runs in a date-keyed `multirun/` tree with no
    `experiment_group` — in tension with Decision 13, which makes the declarative
    matrix runner mandatory. The Formulation 1 cell is Ablation Configuration 4's
    parity anchor, and an anchor analysis cannot address is not an anchor.

57. **`get_canonical_output_dir` takes an optional `variant`, appended as
    `<algorithm>__<variant>`.** The path keys on the algorithm config rather than
    the run label, so any matrix listing one `alg` twice under different overrides
    writes every such run into one directory and keeps only the last, silently.
    This had already bitten the uniform-memory control arm, worked around there by
    splitting the heterogeneity config per alpha; the formulation study's five
    formulations of `fedmaq` would have been the second occurrence. A
    parametrized guard now asserts output-directory uniqueness across every
    confirmatory matrix, so the next occurrence fails in CI rather than at
    analysis time.

---

## 2026-07-31 — Freeze Ergonomics and the Pre-Registered Empty Branch

58. **The §4.3.7 ablation arms inherit `conf/algorithm/fedmaq.yaml` through their
    Hydra defaults list instead of restating it.** Each of `fedmaq_no_resource`,
    `fedmaq_no_data`, `fedmaq_no_state`, `fedmaq_no_kd`, and
    `fedmaq_no_refinements` is now `defaults: [fedmaq, _self_]` plus its own
    removal and nothing else. Previously all five restated `soft_voting` /
    `ema_student` / `grad_norm_ema` as literals, which made the freeze a five-file
    hand edit at the single highest-stakes moment in the runbook; a partial or
    empty surviving set desynced three of the arms. §4.3.7's parity requirement
    stops being a property the tests check after the fact and becomes one the
    configs cannot violate: a second difference is no longer expressible.
    `fedmaq_no_refinements` still names all three flags, because "every refinement
    off" is its definition rather than a copy of the freeze — whatever subset
    `fedmaq.yaml` freezes on, that arm removes exactly that subset.

59. **The cost of Decision 58 is paid by `scripts/dump_frozen_configs.py`, which
    generates `docs/freeze/resolved_configs.yaml`.** Inheritance destroys the arms'
    self-describing property, which chapter 6 §6.2's "recoverable frozen
    configuration" promise leans on: reading an arm no longer tells you what it
    runs. The script composes each `fedmaq*` config exactly as `scripts/run.py`
    would and commits the resolved result with a per-arm `config_sha256` (over the
    algorithm block alone, so it is stable across the dataset/skew/seed a matrix
    sweeps and does *not* equal any run manifest's whole-config digest). Being
    generated, it cannot drift the way five hand-maintained copies could.
    `--check` exits non-zero on a stale snapshot, and
    `test_frozen_config_snapshot_is_current` runs that check in the suite, so a
    freeze commit that edits `fedmaq.yaml` without regenerating fails rather than
    ships. Regenerating it is step 5 of the freeze runbook in `docs/RUNBOOK.md`,
    and an instruction in a runbook is not a guard — the test is. A stale snapshot
    is worse than none, since it describes a configuration the tag does not
    contain.

60. **The empty-freeze branch is pre-registered in the manuscript body, not only
    in YAML comments.** `conf/matrix/pass3_freeze_confirm.yaml` and
    `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` already
    pre-registered that an empty surviving set drops Ablation Configuration 8 and
    the chapter 6 contribution bullet resting on its contrast. The manuscript said
    flatly that the study "consists of eight test configurations", so the
    pre-registration was one no reader could hold the thesis to — and writing it
    in after seeing an empty result is the exact thing pre-registration prevents.
    §4.3.1 now carries the failure branch (empty set, FedMAQ ships unrefined, no
    subset retries, no tuning rescue) and §4.3.7 carries Configuration 8's
    existence condition together with the six runs the empty branch removes from
    §4.4's totals. §4.3.7's claim that Configuration 8 "supplies the parity
    reference that makes the remaining arms legible" was corrected in the same
    edit: Configuration 7 is the parity anchor for arms 2–5; Configuration 8
    prices the shared layer.

61. **§4.3.1 states the noise-margin *rule* and names no seed counts.** Three
    sites said "three seeds" while §4.4 deepened the reference cell to five, so
    the two sections contradicted each other. The deeper defect a number-swap
    would have hidden: §4.3.1 read as though there were one σ measured once, while
    `scripts/analysis.py:exploration_noise_margin` takes an `experiment_group` and
    *refuses* to pool stages — the factorial and the R=100 confirmation each
    compute their own. That methodological choice was visible only in a Python
    default argument. §4.3.1 now states that σ comes from the unrefined reference
    cell of the stage making the call, at the held-out α = 0.3, with margin √2σ,
    never pooled across stages; §4.4 is the single site owning the seed
    arithmetic. Exploration is described as three stages throughout, matching
    `pass3_freeze_confirm.yaml`'s "Stage 3 of 3".

62. **The pre-existing `ruff` drift was cleared before the freeze, and the five
    `B905` sites took `strict=True` rather than `strict=False`.** 21 errors and 11
    unformatted files at HEAD; §6.2 offers the tagged commit as the reproducible
    artifact, so an artifact that fails its own linter was worth not shipping.
    `strict=` was the only judgement call in the batch: ruff's own unsafe fix
    writes `strict=False`, which silences the rule while preserving the silent
    truncation it warns about. All five sites zip sequences whose lengths are
    invariant-equal — `REFINEMENT_NAMES` against a 3-tuple cell key in
    `exploration_noise_margin`, and seed/accuracy literals in four
    `test_analysis.py` fixtures — where a length mismatch means the refinement
    roster changed under the analysis. Truncating there would label a keep-or-drop
    verdict with the wrong mechanism set, so crashing is the better failure.
    `scripts/analysis.py:790` already used `strict=True` for the same pattern; the
    batch makes that consistent. Everything else was mechanical: `ruff format`,
    import sorting, two unused `flwr.common` imports, `datetime.UTC`, a dead
    `alg_cfg` left over from the F12 fix in `cfd.py`, and six re-wrapped long
    lines. 168 tests pass, unchanged.

---

## 2026-07-31 — HANDOFF.md Retired

63. **`HANDOFF.md` is deleted and no committed handoff file replaces it.**
    Session-to-session orientation is now a temporary artifact produced by the
    `handoff` skill and never tracked in this repo. The committed file failed in
    the way a tracked handoff always does: it accreted durable operational content
    that then went stale and was cited from three other docs, while the transient
    action items it actually existed for were rewritten wholesale every session.
    The audit that prompted this found the failure mode directly —
    `.claude/rules/flower-patterns.md` pointed at `HANDOFF.md § "Execution Model"`,
    a section that did not exist, so the single most important operational fact in
    the project (runs execute on a JupyterHub-gated allocation that no agent can
    drive) was one dead link from being lost.

    Its content was split by lifetime rather than discarded. Durable operational
    reference — execution model, dispatch order, operational controls — moved to
    the new `docs/RUNBOOK.md`. Current state moved into `docs/STATUS.md`, which
    was stale on two counts the move surfaced: it still said the exploration phase
    had not started (Stage 1 ran 2026-07-31) and still carried the pre-deepening
    exploration run counts (24/4 rather than 26/8). The transient "Next Session"
    section was dropped, which is the whole point. `docs-management.md` now
    forbids the pattern, and the `docs-audit` skill treats a tracked handoff file
    as a finding rather than as something to audit.

---

## 2026-08-01 — Experiment-Defensibility Audit, Pass 3

Third audit pass over the frozen-configuration design, run against the standard of
the previous two (find defects that would make the frozen configuration not mean
what the manuscript says it means) with an explicit feasibility constraint: a fix
that costs weeks of runs is not a fix. Three of the prior handoff's six leads
cleared on inspection and are recorded here as non-findings so they are not
re-audited — FEMNIST's `num_clients` and throughput constants agree across Table 4.1
note (a), `conf/experiment/femnist.yaml` and §4.3.4 (the `L40S` in config comments is
the *modeled* server, correctly distinguished from the A100 execution host at
`chapter_4.tex:286`); Configuration 4's parity-anchor claim holds, because
`formulation_study.yaml` sets no refinement overrides and therefore inherits the
freeze through `fedmaq.yaml`; and `entropy_weight`/`precision_weight` ship at the
neutral `1.0`/`1.0` that §4.3.1 requires.

64. **A formulation winner that splits across skews now has a rule: the α = 0.1
    winner freezes.** `scripts/analysis.py:select_winner` is per-`(dataset, alpha)`
    and the formulation study runs CIFAR-10 at both skews, so two verdicts are
    structural, not exceptional. The freeze takes one scalar (`fedmaq.yaml`'s
    `formulation:`), the 105-run grid runs one FedMAQ, and every §4.3.7 arm inherits
    one formulation — and §4.3.6 saw the possibility ("should the winning formulation
    shift with skew, that is itself a headline finding") without resolving it, leaving
    three downstream branches undefined: Configuration 4's expressibility, which is
    conditioned on whether Formulation 3 won; the reserved recheck, conditioned on the
    same; and the freeze itself. The rule is now agreement-or-severe-skew: if both
    skews pick the same formulation it freezes, and if they diverge the α = 0.1 winner
    freezes and the split is reported as the finding §4.3.6 already promises. Severe
    skew breaks the tie because it is the regime this thesis's claims are staked on,
    so selecting where the problem is hardest is the conservative direction rather
    than the flattering one. A symmetric aggregate was considered and rejected on
    technical grounds, not rhetorical ones: `compute_target_floor` is called *inside*
    the per-skew loop, so each skew's cumulative-MB is measured against its own floor
    at its own convergence speed, and summing incommensurable quantities hands the
    decision to whichever skew has the larger absolute MB scale. Normalizing to fix
    that introduces exactly the second free parameter §4.3.6 forbids.
    Implemented as `scripts/analysis.py:resolve_frozen_formulation`.

65. **Total disqualification is pre-registered, and per-seed disqualification is
    stated in the manuscript rather than inferred from a docstring.** `select_winner`
    already returned `winner = None` when the accuracy-floor guard caught all five
    formulations; §4.3.6 had no branch for it. This is not hypothetical — the floor is
    90% of *uncompressed* FedAvg, and the formulation study deliberately runs FedMAQ
    in its least communication-efficient state (quantized under a
    U(2048, 16384) MB ceiling with the post-processing pipeline withheld so the
    formulas are judged on mathematical merit). Clearing that at α = 0.1 is genuinely
    uncertain. The rule is also stricter than the prose read: `if round_num is None`
    sits inside the per-seed loop, so **one** failing seed of three disqualifies the
    formulation, where §4.3.6's "any formulation that fails to reach the target
    accuracy" reads naturally as a claim about its mean. The strict rule is the right
    one — it refuses to average away a seed that never converged — and is now written
    as such. Branches: a wipeout at one skew defers to the other outright (Decision
    64's divergence rule does not apply, there being only one valid verdict); a
    wipeout at both falls back to highest mean top-1 at R=100 at α = 0.1, the
    criterion already pre-registered as the near-tie break, **and withdraws §4.3.6's
    framing of formulation selection as the thesis's primary methodological
    contribution**, with §5.7 recording that no formulation reached the
    FedAvg-relative target. This deliberately does *not* mirror Decision 60's
    empty-freeze branch, which withdraws a claim rather than manufacturing a winner.
    The disanalogy is that "ship unrefined" is a coherent configuration and "ship no
    formulation" is not: `fedmaq.yaml:18` gets a number either way, so the only real
    choice is whether it is picked empirically or left at whatever the implementation
    already contained — and freezing the incumbent by default would mean the thesis's
    self-described primary contribution was settled by a default value.

66. **The near-tie tie-break compares against within-candidate spread, never pooled
    spread.** The implementation concatenated both candidates' `crossing_mbs` and took
    the sample standard deviation of the combined six, which folds the *between*-
    candidate separation into the threshold. With within-candidate sd `s` and
    separation `d`, the combined sample variance is `(4s² + 1.5d²)/5`, so the rule
    fired at `d < 1.069s` and the threshold grew with the very margin it was judging.
    A pre-registered rule cannot be self-referential, the behaviour breaks
    asymmetrically when the two candidates have unequal spread, and it is not what
    §4.3.6 published ("smaller than either candidate's seed-to-seed variability").
    Now `max` of the two candidates' own standard deviations, with §4.3.6 corrected to
    say exactly that and to state that the rule applies to the **top two** candidates
    only — which is what the code has always done and the prose never said. The
    sharper statistic, the standard error of the difference `sqrt(s1²/3 + s2²/3)`, was
    considered and rejected: §4.4 already argues that at n = 3 the σ estimate itself
    carries a roughly twelvefold interval, so a finer instrument built on top of it is
    false precision. A difference smaller than one candidate's own run-to-run spread
    should not decide the thesis's primary methodological contribution.

67. **Baseline matched-tuning exists as a real stage: `conf/matrix/baseline_tuning.yaml`,
    55 runs, Stage 1b.** Decision 31 pre-registered the budget, Decision 29 sequenced
    it, §4.3.1 promises "the baseline hyperparameter table (Table 4.1)" is locked and
    tagged at the freeze, and `STATUS.md` carried a standing warning that a
    MobileNetV2GN sweep was required — and nothing produced it. No matrix file, no
    stage in `RUNBOOK.md`, and the word "tuning" appeared in `chapter_4.tex` only in
    the phrase "rather than a single unstructured tuning pass". **The gap is transfer,
    not provenance** — §4.3.2 already sources every baseline constant carefully (FedProx's
    μ = 1 cited to the image benchmarks of its own paper's sweep, DAdaQuant's φ = 10
    derived from a published rule instantiated at R = 100, FedPAQ's 8-bit adopted from
    the precision another paper benchmarks it at since FedPAQ publishes levels rather
    than bit-widths, FedKD's SVD endpoints declared as ours rather than inherited). That
    paragraph was found *after* this decision was first drafted and corrects it: the
    baselines are not unprovenanced. What no published value can establish is that it
    still holds on MobileNetV2GN at α ∈ {0.1, 1.0}, which is a different architecture
    and a different skew from any of the sources. FedMAQ meanwhile received a 38-run
    exploration phase and a 30-run formulation study on exactly this configuration.
    *You verified yours on this grid and theirs on someone else's* is the surviving form
    of the attack, and it had no answer.

    Design: five baselines (FedAvg is the uncompressed control and has no knob), one
    key hyperparameter each, a five-seed reference cell at the shipped Table 4.1 value
    plus two three-seed challengers = 5 × 11 = 55 runs, at the held-out α = 0.3, under
    FedMAQ's own √2σ rule, uncounted among the 183. **R=100, not the R=50 every other
    exploration stage screens at**: baselines get no confirmation stage, so a
    truncated-horizon pick would ship into the reported grid uncorrected. That is most
    of the ~26 GPU-hour cost and it is the one place worth spending rather than
    economizing. Reference cells are deepened to five seeds by Decision 61's argument
    applied per baseline; standardizing to three would save ten runs and leave the
    baselines' margins estimated more coarsely than FedMAQ's own freeze gate, which is
    an indefensible ordering.

    **The null result is the product.** The likely outcome is that no challenger
    clears and every baseline freezes at the value it would have had with no sweep —
    which converts an appeal to authority into a measurement, and answers the question
    the zero-run alternative cannot: how do you know a constant tuned on a different
    architecture transfers to MobileNetV2GN under Dirichlet skew? FedProx's μ is the
    sharp case, being the proximal-strength knob and skew-sensitive by construction.
    A cheaper selective sweep (tune only the knobs that "look sensitive") was rejected
    for its shape rather than its cost: the person whose algorithm benefits would be
    deciding which of his competitors' knobs deserved tuning, which invites the exact
    suspicion the sweep exists to dispel. Uniform treatment has no soft spot.

    **This supersedes Decision 31's knob list for FedKD.** Decision 31 named
    "FedDistill/FedKD distillation temp"; FedDistill has no temperature (its knob is
    `reg_alpha`) and FedKD's `temperature` governs client-side mutual KD while `tmax`,
    the SVD energy cutoff, governs the accuracy–communication trade-off this grid
    actually compares on. Each baseline is tuned on the knob governing its own
    accuracy–communication trade-off, because that is the axis every claim rests on.
    Recorded rather than silently substituted.

68. **The reserved 6-run recheck is promoted from a YAML comment into §4.3.6, and the
    freeze is tagged once, at the end of exploration.** `formulation_study.yaml`
    pre-registers that a winner other than Formulation 3 triggers a 6-run recheck of
    the frozen refinement layer under the winning formulation, and that a failed
    recheck removes `soft_voting` from the frozen set, rewrites `fedmaq.yaml` and every
    §4.3.7 arm, and shrinks or drops Configuration 8. A procedure that spends uncounted
    runs and mutates the frozen configuration was visible nowhere in the manuscript,
    while §4.3.1 said flatly that exploration ends once the third stage's confirmation
    has held. That is the failure Decision 60 exists to prevent: a branch only in code
    comments cannot function as pre-registration. Kept rather than deleted, because
    this is one named mechanistic coupling and not the generic sequential-selection
    caveat already in §5.7 — `soft_voting` reweights teacher logits *by quantization
    level*, so it is specifically the mechanism whose value depends on which formula
    assigns those levels, and the expected cost is 6 runs × P(F3 loses) on an
    incumbent default. §4.3.6 now states it as a **veto, not a search**: it can only
    remove from the frozen set, never add; the factorial is never re-opened; no subset
    is retried. Auditing this surfaced a second defect — `RUNBOOK.md` git-tagged at
    the refinement freeze, but §4.3.1 locks and tags three artifacts together (fixed
    mechanism set, selected formulation, baseline hyperparameter table) and two of them
    do not exist at that point. The tag moved to a new Stage 2b after the formulation
    study resolves.

69. **`docs/plans/formal-experiment-plan.md` is deleted.** Per
    `.claude/rules/docs-management.md`, plans are active-only. Its resolved content had
    gone beyond stale to actively wrong: it asserted "Pass 1 Complete (provisional
    pick: `ew=2.0`, `pw=0.5`)" while `fedmaq.yaml:59–60` ships `1.0`/`1.0`, because
    §4.3.1 now requires neutral-point hyperparameters throughout exploration — a reader
    following the plan would reintroduce the tuned values and silently violate the
    protocol. Its Pass 1/2/3 vocabulary also collided with §4.3.1's three stages
    (its "Pass 3" is the formulation study, now a 30-run first-class experiment, not a
    stage), and its "Execution Structure (tentative)" duplicated `RUNBOOK.md`.
    `STATUS.md` cited it twice as authoritative; both links now point at `RUNBOOK.md`
    and the `conf/matrix/*.yaml` headers, which are authoritative over prose describing
    them. One item existed nowhere else and is carried forward here: **FedMAQ must be
    compared against the pure-quantization baselines (FedPAQ, DAdaQuant) at matched bit
    budgets**, since an accuracy-vs-compression frontier plotted across unmatched
    budgets is apples-to-oranges. `baseline_tuning.yaml` brackets FedPAQ's `q` at
    {4, 8, 16} partly to make that comparison constructible; the plot itself remains
    unbuilt and is tracked in `STATUS.md`, not in a revived plan file.

70. **`client_kd_reg` and `kd_prox_mu` are retired from the exploration roster, and the
    retirement is recorded because a standing decision promised the opposite.**
    Decision 29 put "client-KD-reg + proximal (μ)" on the Pass 2 roster and Decision 32
    closed with "`client_kd_reg=true` runs as-is in Pass 2; no code change needed". It
    does not run: `pass2_explore.yaml` screens `soft_voting`, `ema_student` and
    `grad_norm_ema`, `pass2_factorial` is a 2³ over those same three, and
    `fedmaq.yaml:71,74` ship `client_kd_reg: false`, `kd_prox_mu: 0.0`. So a mechanism
    left the roster with a decision on record saying it would be evaluated and none
    retiring it. §4.3.1 is unaffected — its refinement list is explicitly "illustrative
    rather than closed" — but the repo record contradicted itself. Retired rather than
    reinstated: adding a fourth factor makes Stage 2 a 2⁴ factorial, doubling it from
    26 runs to 50 for a mechanism whose only prior evaluation (Decision 32) concluded
    its implementation cost was not worth paying. The code path remains for
    reproducibility, as with FedMD.

## Pass 4 — 2026-08-01

Terminal audit pass before dispatch. One finding, with three code sites and a
dispatch-order change; it is the last structural defect the freeze machinery had.

71. **The formulation study's accuracy floor was circularly dependent on the grid it
    configures, and its candidate set admitted three populations of impostor runs.**
    Both halves are the same defect seen from either end: `select_winner` identified
    formulation-study runs by algorithm rather than by matrix, and identified the
    FedAvg reference by algorithm rather than by matrix.

    *The circularity.* §4.3.6 defines the target accuracy as 90% of the uncompressed
    FedAvg reference "reusing the FedAvg runs already present in the benchmark grid",
    and `compute_target_floor` implements exactly that. But FedAvg appears only in
    `benchmark_grid.yaml`, dispatched at Stage 4 — after the Stage 2b tag that freezes
    the formulation those very runs are needed to select. Run `analysis.py` at the
    point `RUNBOOK.md` says to resolve the verdict and it raised `ValueError: No
    FedAvg reference runs found` before writing anything. **Resolved by pulling the
    grid's six CIFAR-10 `fedavg` rows forward into a new Stage 1c** (`run_matrix.py
    --matrix benchmark_grid --only fedavg`, a new label filter on the runner). They
    keep the grid's `experiment_group`, overrides and output directories, so Stage 4's
    `--skip_completed` passes over them: six of 42 rows moved, not six added, and the
    183-run total is unchanged.

    Considered and rejected: a dedicated 6-run FedAvg reference cell inside
    `formulation_study.yaml` at `phase: explore`. It keeps every confirmatory run
    behind the tag, which is the honest attraction, but it creates two FedAvg
    populations at the same config, skew and seeds — the floor computed from one, the
    grid reported from the other. A reader recomputing the floor from the published
    table would then be entitled to a different disqualification set than the one that
    actually fired, which is a worse property for a pre-registered rule than an
    ordering exception. The exception is admissible and is disclosed in §4.3.6:
    FedAvg is invariant to all three artifacts §4.3.1's tag locks — no refinement
    layer, no formulation, and no tunable constant, which is why Stage 1b excludes it
    as the uncompressed control. It extends to nothing else.

    *The impostors.* `select_winner` selected candidates as `algorithm_config ==
    "fedmaq" and formulation is not None`, excluding only the ablation group. Three
    populations satisfy that and should not: (a) `pass2_factorial` and
    `pass3_freeze_confirm`, which dispatch `algorithm=fedmaq` at the held-out
    α = 0.3 while `fedmaq.yaml` carries `formulation: 3` — these manufacture a verdict
    at a skew where no FedAvg reference exists by design, so the observable failure is
    the whole analysis dying rather than a subtly wrong winner; (b) the grid's own
    FedMAQ rows, same dataset, skews and seeds as the study, differing only by
    `post_process=true` — pooled into Formulation 3's cell they credit the *incumbent*
    formula with the pipeline savings §4.3.6 exists to withhold, so the contamination
    flatters the status quo rather than perturbing it randomly; (c) in
    `compare_to_baselines`, the same collision resolved by dict-iteration order, which
    could report pipeline-free FedMAQ against pipeline-era baselines and break
    §4.3.6's own regime rule in the one direction that understates FedMAQ.
    All three are closed by scoping on `experiment_group`: candidates from
    `formulation_study`, floor and headline comparison from `benchmark_grid`,
    Configuration 4's parity anchor from `formulation_study` (tightened from "not the
    ablation group"). `compare_to_baselines` no longer takes its targets from the
    winner verdict at all — it enumerates the grid and uses the verdict only to assert
    the grid ran the frozen formulation, reporting `formulation_matches_freeze` and
    printing a `FREEZE DRIFT` line when it did not. Four regression tests; the
    fixtures now carry an `experiment_group`, since a fixture without one describes a
    run no matrix could produce.

**Non-findings this pass** (verified, do not re-audit): `build_ablation_table`'s
Configuration 4 guard does not assume a single unambiguous formulation winner — it
reads the arm's own `formulation` field, so `resolve_frozen_formulation` cannot
desynchronize it. The `not r.post_process` condition in that guard already excluded
the grid's rows before the group filter was added; the filter makes the intent
explicit rather than repairing a live bug.

### Decision 72 — Every confirmatory run postdates the pre-registration tag; the calendar moves to fit

**Finding.** Manuscript §4.5 read *"the six baselines complete their full dataset ×
skew × seed grid during the August--October window"*, which is why *"the February
benchmarking block carries only FedMAQ's own main-grid runs, the uniform-memory
control arm, and the 42 net-new ablation runs rather than the entire 183-run grid."*
That places all 90 reported baseline runs (6 baselines × 5 dataset-skew cells × 3
seeds) — 75 of them belonging to the five *tuned* baselines — before the January tag that §4.3.1 uses to lock the baseline hyperparameter
table those very runs are configured by — and §4.3.2's matched-tuning stage
(Decision 67) is entitled to rewrite five of those constants right up until the tag
fires. Five of six baselines would therefore have reported accuracies measured under
values the freeze had not yet fixed. This is the exact tuning asymmetry §4.3.2 was
added to remove, reintroduced through the schedule.

It also made the §4.3.6 disclosure added under Decision 71 false as written: FedAvg's
six CIFAR-10 rows could not be *"the only confirmatory runs that execute before the
pre-registration tag"* while 90 baseline rows preceded it too.

**Resolved: the confirmatory block runs whole, after the freeze.** August--October is
now stated as baseline *reproduction* — implementation, correctness checks against
published behaviour, and short validation runs — with no confirmatory measurement in
it. The 147 confirmatory runs remaining after Stage 1c's six FedAvg rows are allotted
the second half of January together with February, rather than February alone. Two
Gantt rows change and one is added (`Baseline Matched-Tuning`, November, previously
absent from the chart although §4.3.2 requires it).

*Rejected: a pre-registered conditional re-run* — baselines keep the early window,
and any baseline whose matched-tuning stage changes its constant has its 15 grid rows
re-run after the tag. Expected cost is near zero, since §4.3.2 expects no challenger
to clear, and it preserves the calendar's feasibility argument untouched. It was
rejected because the common case still reports rows measured before the table locking
them was tagged, and one only knows no re-run was needed *after* seeing the tuning
result — a pre-registration that is partly retrospective. The calendar was tentative
and the compute is allocation-bound rather than month-bound, so there was no reason
to buy schedule with defensibility.

*Rejected: dropping Table 4.1 from the tagged artifacts* — cheapest, and it unwinds
Decision 67 and §4.3.2 one day after they were added.

**Stage 1b's placement is unchanged and is not an ordering constraint.** Baseline
matched-tuning shares no configuration with FedMAQ, so it is unordered with respect
to the refinement search; it sits at Stage 1b because Decision 29 sequenced it there.
The only hard ordering the tag imposes on it is that it must finish before the tag.

### Decision 73 — DAdaQuant was capped at 5 bits by a constant of ours that read as 8

**Finding.** `conf/algorithm/dadaquant.yaml` carried `q_max: 8` beside
`conf/algorithm/fedpaq.yaml`'s `q: 8` and `conf/algorithm/fedmaq.yaml`'s
`q_max: 16`, and the three do not denominate the same quantity. FedPAQ's `q` and
FedMAQ's bounds are bit-widths; DAdaQuant's `q` counts quantization levels *per
sign* — codes in $[-q, q]$, so $2q+1$ levels and $\lceil\log_2(2q+1)\rceil$ bits
per element. `DAdaQuantCompressionHook`'s docstring states this plainly; nothing
outside that docstring did. Eight levels per sign is five bits, so the baseline a
reader would take to be matched to FedPAQ's 8-bit budget was running at roughly
five-eighths of it.

Two consequences, neither disclosed anywhere. First, the precision ceiling of a
competitor on the exact accuracy-versus-communication frontier every claim in the
thesis is drawn on was set by a constant we chose, absent from Table 4.1, absent
from §4.3.2's list of values marked as ours, and not among the five the
matched-tuning stage tests. Second, and worse for the baseline's own integrity:
`q_t` starts at `q_min = 1` and doubles, so a ceiling of 8 is reached after three
doublings — around round 30 of 100 at $\phi = 10$ — leaving DAdaQuant effectively
static for the remaining seventy rounds. Time-adaptive escalation is what
DAdaQuant *is*; the cap suppressed most of it.

**Resolved: `q_max = 127`, and the table says why.** 127 levels per sign is
$2 \times 127 + 1 = 255$ codes, exactly eight bits, so DAdaQuant's ceiling now
equals FedPAQ's and the two pure-quantization baselines are separated by
adaptivity rather than by budget. `q_min = 1` is left alone — it is published, the
$\max(1, \cdot)$ floor of DAdaQuant's own client-adaptive formula (Hönig et al.
2022, §3.5). Some upper bound is still required, the source clamping none on the
per-client $q_i$ and stating no ceiling on the doubling of $q_t$, so a
large-share client would otherwise be assignable outside any budget; the change
is which bound, not whether.

Table 4.1 gains the row, footnote (d) states the levels-versus-bit-widths trap
outright, and §4.3.2 now marks $q_{\max}$ as ours — five values warranting
comment rather than four. `dadaquant.yaml` carries the units in the file, since
the key name is what made the mismatch invisible.

**Zero run cost:** nothing has been dispatched. Had the grid already run, this
would have been a re-run of DAdaQuant's fifteen rows rather than an edit, and the
right call would still have been to make it — a baseline held below the budget its
neighbour gets is the asymmetry §4.3.2 exists to remove, in a place §4.3.2 does
not look.

**Direction of the correction.** It strengthens a competitor. That is deliberate:
of the two available fixes, disclosing the 5-bit cap and leaving it, or matching
the budget and disclosing that, only the second cannot be read as the author of
FedMAQ choosing how much precision his competitors are allowed.

**Also corrected:** `baseline_tuning.yaml` described $\psi$ as "the escalation
factor". It is the weight of the loss moving average (`dadaquant.py:190`); the
escalation is the doubling of `q_t`. The same header's `phi` rationale now notes
that $\phi$ decides more under a 127 ceiling — seven doublings to climb rather
than three — which makes its $\{5, 10, 20\}$ bracket more consequential than when
it was written, not less.

### Decision 74 — Chapter 6 claimed two contributions whose withdrawal Chapter 4 had already pre-registered

**Finding.** §4.3.1 pre-registers the empty-freeze branch — no mechanism clears
its noise margin at R = 100, FedMAQ freezes unrefined, Ablation Configuration 8
drops for want of anything to remove — and §4.3.6 pre-registers total
disqualification, where every formulation falls below the accuracy floor at both
skews and the formulation contribution is withdrawn. `docs/RUNBOOK.md` step 5
even names the consequence: *"the `chapter_6.tex` contribution bullet resting on
its contrast goes with it."* Chapter 6 stated both contributions with no
condition attached. Its bullets are `[PLACEHOLDER]`, so nothing false was
asserted yet — but a branch that exists only in the chapter that will not be
written under it is not pre-registered, it is remembered, and the RUNBOOK line
was the only thing holding the coupling.

**Resolved.** The formulation bullet now carries both §4.3.6 branches (total
disqualification withdraws the characterization and reports the null; skew
divergence freezes the α = 0.1 winner and reports the split as a finding rather
than smoothing it). The Configuration 8 bullet states that it drops entirely
under the empty freeze, leaving the allocation structure as the contribution —
"the honest reading of that outcome rather than a diminished one."

**Also corrected: "183-run confirmatory grid" in three chapters.** §4.3.4 is
careful that the 183 reported runs are 153 confirmatory *plus* the 30-run
formulation study, which is exploration-phase and "reported here but not counted
among the confirmatory runs". Chapters 1, 5 and 6 each collapsed that into a
"183-run confirmatory grid", and §4.5 called the exploration runs uncounted among
"the 183 confirmatory runs" — the one phrase §4.3.4 exists to prevent. All four
now say what they mean. The distinction is load-bearing, not pedantic: the
formulation study selects the configuration the confirmatory grid then runs, and
a reader who believes it is *inside* that grid has been told the freeze was
decided by runs it was supposed to precede.
