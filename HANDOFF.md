# Handoff Context: FedMAQ Experiments

**Purpose**: Operational orientation and immediate action items for the next agent. Historical details, audit findings, and resolved methodology decisions are maintained in `docs/`.

**Last updated**: 2026-07-22

---

## 1. Quick Pointers & Primary Resources

- **Current State & Standings**: [docs/STATUS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/STATUS.md)
- **Resolved Methodology & Framing Decisions**: [docs/DECISIONS.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/DECISIONS.md)
- **Experiment Registry & Historical Runs**: [docs/experiments/README.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/docs/experiments/README.md)
- **Terminology & Glossary**: [CONTEXT.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/CONTEXT.md)

---

## 2. Current Project State Summary

FedMAQ trains **MobileNetV2GN** (~2.24M params) on CIFAR-10 as its primary thesis model.

- **Exploration Phase**: Adaptive Pass 1 (soft-voting weights) completed (`ew=2.0, pw=0.5, soft_voting=true` provisional pick; Decisions 33-35 in `DECISIONS.md`).
- **Baseline Comparators**: 6 formal baselines (FedAvg, FedProx, FedPAQ, DAdaQuant, FedDistill, FedKD). FedMD & CFD are dropped (Decisions 25/26).

---

## 3. Immediate Next Actions

### Priority 1: MobileNetV2GN Exploration & Baseline Tuning

1. **Pass 2 Exploration**: Run Pass 2 sweeps for capacity-EMA (on/off), grad-norm-smoothing isolation, and client-KD-reg.
2. **Pass 3 Exploration**: Evaluate Formulation 3 (dual-tier precision scaling).
3. **Matched Baseline Tuning**: Conduct matched light hyperparameter tuning (≤5 HP values each) for the 6 active baselines.
4. **Pre-registration**: Freeze and git-tag the final FedMAQ configuration and baseline HP table.

### Priority 2: Confirmation Infrastructure

5. **Config-as-Code Registry**: Implement the manifest driver for formal multi-seed confirmatory runs.
6. **Seed Determinism**: Verify partition generation and client sampling seed invariants (`pytest tests/test_environment.py`).

---

## 4. Key Operational Controls

- **Subprocess Runner Mandate**: Hydra `--multirun` causes CUDA VRAM leaks. Always launch sweeps using process-isolated runners in `scripts/` (e.g. `python scripts/run_soft_voting_explore.py`).
- **RAM Headroom & Crash Recovery**: Check system RAM headroom before Flower simulations. Resume crashed chains using `--start_at N`.
