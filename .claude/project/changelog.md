# FedMAQ Project Milestone Changelog

Major project milestones, architecture shifts, and PR releases for the FedMAQ thesis codebase.

**Policy (as of 2026-07-11):** Only append entries for major milestones (merged PRs, architecture shifts, phase completions).

---

## Major Milestones

### 2026-07-18 — Architecture & Exploration Pass 1 Complete

- **Pass 1 Exploration**: Completed soft-voting sweep (18/18 runs) on MobileNetV2GN (`ew=2.0`, `pw=0.5` provisional pick; Decisions 33–35).
- **Architecture Branch**: Merged PR #6 (determinism, centralized model factory, config defaults) and PR #7 (Phase 6 decouple DAdaQuant proxies). Landed seeded client manager (`SeededPartitionClientManager`).

### 2026-07-16 — Baseline Stack Cleanup & Methodology Decisions

- Resolved 13 framing/methodology decisions (Decisions 1–26 in `docs/DECISIONS.md`).
- Formally dropped **FedMD** (Decision 25, infeasible pretrain cost) and **CFD** (Decision 26, structural collapse under client budget scale). Active baselines set to 6.

### 2026-07-15 — MobileNetV2GN Switch & Dual-Variant Support

- Switched default CIFAR model from ResNet18GN (~11.17M params) to **MobileNetV2GN** (~2.24M params) for edge realism (Decision 1). Deprecated ResNet18GN smoke tests.
- Partitioned codebase into `fedmaq` (MobileNetV2GN) and `fedmaq_lite` (SimpleCNN).

### 2026-07-14 — Soft-Voting, EMA, and Gradient Smoothing

- Added quantization-aware soft-voting, student EMA, and per-client gradient norm EMA (`grad_norm_beta=0.7`).

### 2026-07-11 — Baseline Ports & Initial Test Infrastructure

- Completed baseline ports (FedPAQ, DAdaQuant, FedKD, FedDistill, CFD).
