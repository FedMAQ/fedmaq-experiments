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

## 2026-07-16 — Context-Docs Management Conventions (4 decisions)

Resolved via grilling session on doc-hygiene drift (stale STATUS.md date, broken section numbering, duplicate registries). Enforcement: [.claude/rules/docs-management.md](../.claude/rules/docs-management.md) + `.claude/skills/docs-audit/`.

14. **Single experiment registry**: `docs/experiments/README.md` is canonical. `.claude/project/experiment_registry.md` deleted (was stale, duplicated coverage).
15. **`docs/plans/` is active-only**: a plan doc exists only while it has open questions. On full resolution, content merges into this file (`DECISIONS.md`) and the plan file is deleted — git history is the historical record, not a docs folder.
16. **Archive pattern = per-directory `archive/` subfolder** (e.g. `docs/experiments/archive/`), not a single top-level `docs/archive/`. Archived content stays next to its live counterpart.
17. **Enforcement = rule + skill**: `.claude/rules/docs-management.md` (always-loaded conventions, applied during routine edits) plus `.claude/skills/docs-audit/` (on-demand full sweep; auto-fixes mechanical issues like stale dates/numbering/dead links, flags semantic overlap/staleness for human judgment).
