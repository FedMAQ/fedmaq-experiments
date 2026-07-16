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
    evidence.

---

## 2026-07-16 — Context-Docs Management Conventions (4 decisions)

Resolved via grilling session on doc-hygiene drift (stale STATUS.md date, broken section numbering, duplicate registries). Enforcement: [.claude/rules/docs-management.md](../.claude/rules/docs-management.md) + `.claude/skills/docs-audit/`.

14. **Single experiment registry**: `docs/experiments/README.md` is canonical. `.claude/project/experiment_registry.md` deleted (was stale, duplicated coverage).
15. **`docs/plans/` is active-only**: a plan doc exists only while it has open questions. On full resolution, content merges into this file (`DECISIONS.md`) and the plan file is deleted — git history is the historical record, not a docs folder.
16. **Archive pattern = per-directory `archive/` subfolder** (e.g. `docs/experiments/archive/`), not a single top-level `docs/archive/`. Archived content stays next to its live counterpart.
17. **Enforcement = rule + skill**: `.claude/rules/docs-management.md` (always-loaded conventions, applied during routine edits) plus `.claude/skills/docs-audit/` (on-demand full sweep; auto-fixes mechanical issues like stale dates/numbering/dead links, flags semantic overlap/staleness for human judgment).
