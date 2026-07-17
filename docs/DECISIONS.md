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
