# FedMAQ Decisions Log

Single source of truth for resolved project decisions. Append-only, dated. STATUS.md, HANDOFF.md, and formal-experiment-plan.md link here instead of repeating this content.

---

## 2026-07-16 — Formal Experiment Framing & Grid (13 decisions)

Resolved via grilling session. Full rationale: [docs/plans/formal-experiment-plan.md](plans/formal-experiment-plan.md).

### Architecture & Framing

1. **Iso-architecture**: every algorithm (FedMAQ + 8 baselines) trains **MobileNetV2GN (~2.24M)**. Baselines re-run from scratch on it. Old ResNet18GN standings retired.
2. **Switch rationale = edge realism only.** Compression ratio (~1.7×) is model-independent at iso-arch (ratio = `avg_bits/32`, set by bit-width allocation, not param count). "Improved comm savings" claim dropped.
3. **Contribution = mechanism-primary**: quantization-robust accuracy under heterogeneity. Comm savings reported honestly as secondary (~1.7×). The ablation table is the headline evidence, not the comm number.
4. **FedMAQ-Lite dropped** from the formal thesis. Same size as main FedMAQ now (SimpleCNN 2.16M ≈ MobileNetV2GN 2.24M); its size-contrast story was the confounded cross-arch comparison being retired. Smoke results → exploration appendix.

### Methodology

5. **Single fixed config per dataset**, held across α. β–α regime dependence reported as a *sensitivity study*, never exploited in headline numbers.
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
recorded here is the *why*. See [PR #6](https://github.com/FedMAQ/fedmaq-experiments/pull/6)
(merged) and [PR #7](https://github.com/FedMAQ/fedmaq-experiments/pull/7).

18. **Determinism oracle closed in three halves.** (a) per-Ray-worker torch-flag
    re-pinning + seeded DataLoader (training reproducibility); (b)
    `SeededPartitionClientManager` — client sampling keyed by *partition-id* with
    per-round RNG, so paired arms at a matched seed draw identical clients; (c)
    deterministic partitioning locked by `test_partition_seed_invariant_for_paired_arms`.
    Enables the paired-seed/paired-test methodology (Decision 6) — seed variance
    cancels, so ~3pp ablation deltas are detectable at n=3.
19. **`generate_partition_indices` is a pure function** of `(dataset, num_clients,
    alpha, num_public_samples, seed, partition)` with **no algorithm input**. Single
    call site passes the *global* `cfg.experiment.num_public_samples`, so every arm
    gets byte-identical partitions. **Footgun:** `num_public_samples` is sliced
    *before* Dirichlet advances the RNG — divergent per-arm values would silently
    diverge partitions. Safe only because it is one global config value; see audit
    F12 for the related dead-fallback.
20. **Hook-decoupled strategy (Phases 2/4/5/6).** Server model-factory dispatch
    centralized (P2); cross-hook fallback defaults centralized in
    `config_defaults.py` (P4); `configure_fit` god-method split into named helpers
    with an extracted `_QuantParams` (P5); DAdaQuant backward-compat property
    proxies removed from `strategy.py` — tests now hit `strategy.hook.*` (P6).
    Rationale: one baseline's state must not leak into the shared strategy surface.

**Open follow-ups** (not decisions — tracked in
[distillation-direction-audit.md](audits/distillation-direction-audit.md)):
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
    *accuracy* symptom, not the rank proxy, is what matters). Followed up with a
    real `run-minitest` (CIFAR-10/MobileNetV2GN, preliminary/10R/α=0.1/seed=0):
    peak accuracy 16.9%→26.3%, mean 12.0%→15.1% as the floor holds rank at
    26.25% vs. the old default's 3.7-4.5% collapse. Noisy at minitest scale —
    F13's full MobileNetV2GN smoke is still the gate before FedKD re-enters
    comparison tables, but the fix now has real-run (not just synthetic)
    evidence. **Superseded-evidence note (see Decision 22):** the real-run
    confirmation above trained the *old* SimpleCNN student; the fix itself is
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
    gone. But `mean_rank_retained` sitting *at* the floor rather than climbing
    past it means the original audit's candidate 3 (SVD too lossy for
    depthwise-separable weights) is still live and is the leading explanation
    for the remaining 15–27pp gap vs. every other baseline (FedAvg, FedProx,
    FedMAQ, DAdaQuant, FedPAQ) — report this as an open finding, not a closed
    investigation, until it's contextualized against the other four KD
    baselines (F13, still run-gated). Uploaded comm rose to 589–740 MB
    (floor-mandated cost of retaining more singular vectors) — still ~12×
    cheaper than FedAvg's uncompressed ~8.5 GB baseline.
    Logs: `outputs/2026-07-16/23-36-50/` (α=0.1), `outputs/2026-07-17/10-20-23/`
    (α=1.0). Full analysis: `docs/audits/distillation-direction-audit.md` F10.

    *(Note for `docs-audit`: decisions 14–17 above are numbered out of the file's
    append order — a pre-existing drift this entry did not introduce or fix.)*

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
    a cap of 100 can run *longer* than the original fixed 10/10 in the worst
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
    *every* baseline, not just FedMD) — user has hit OOM with this before and
    shares the GPU with non-training workloads; needs a dedicated profiling
    pass before any global default change.

---

## 2026-07-17 — CFD dropped from the formal baseline stack

26. **CFD removed from the baseline stack (7 → 6: FedAvg, FedProx, FedPAQ,
    DAdaQuant, FedDistill, FedKD) + FedMAQ.** Closes distillation-audit F15
    (CFD collapses to chance both α — see `docs/audits/distillation-direction-audit.md`
    F15) as **structural, not fixed**. This corrects the F15 write-up's original
    mechanism guess: the collapse is real (confirmed at both toy and
    production scale this session) but the root cause is **not** the
    client-side aggregation codec/ordering the audit suspected — that code
    (`constrained_quantize`, `dequantize`, the shuffle-before-SGD fix in
    `_train_server_model`) was probed directly this session and is correct.
    Empirical diagnosis (three isolated repro runs, `scripts/run.py
    algorithm=cfd` with instrumented per-round vote/loss/prediction-histogram
    logging, since reverted):
    1. **Server-side dual distillation is exonerated.** A discriminator run
       (5 clients @ full participation → healthy 36–45% client-vote consensus,
       `server_distill_epochs` raised 1→20) showed the server model tracks its
       targets correctly (`server_on_public_acc` climbed 18%→38%→36% alongside
       `targets_acc` 45%→46%→38%) once given enough gradient steps. An
       earlier same-session reading that called this a *server*-side mode
       collapse was itself a toy artifact (100 public samples ÷ batch 64 = 2
       gradient steps/round — undertraining, not a bug) and is retracted.
    2. **The real defect is upstream, at production client-count scale.**
       Rerun at 50 clients / `client_fraction=0.1` (matching the smoke's
       ~100-client, low-participation regime) reproduced the audit's
       production symptom directly: `targets_acc` pinned near chance
       (14/10/16/16% across 4 rounds), with individual clients one-hot-voting
       the *same single class* for all 100 public samples from round 1
       onward. Root cause: each client's private partition is tiny at this
       scale (~470 samples at 100 clients on CIFAR-10's 50k-image train set
       minus the 3000-sample public reserve) and 5 local CE epochs from a
       fresh/reset init is enough to overfit to 1–2 dominant local classes —
       healthy *local* train accuracy (50–65%, matching the smoke's
       `client/avg_train_acc`) coexists with near-random generalization to
       the disjoint, class-balanced public set. CFD's 1-bit (`b_up=1`)
       constrained quantization then forces each client's vote to **full
       commitment to that one class** with zero soft/hedged signal, unlike
       the other KD baselines' temperature-scaled soft-probability averaging
       — so a few overfit voters dominate the round's consensus outright.
    3. **Raising the vote bit-width does not rescue it.** Tested `b_up=b_down=4`
       (16 quantization levels, i.e. much less forced-one-hot) at the same
       production-scale regime: `targets_acc` barely moved (14→21% across 4
       rounds, still chance-adjacent) because the underlying client
       *prediction* is wrong, not merely imprecisely encoded — more
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
