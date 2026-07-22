# FedMAQ Formal Experiment Plan

**Status**: Active. Settled decisions in `docs/DECISIONS.md`. Currently executing adaptive exploration phase on MobileNetV2GN.
**Last updated**: 2026-07-22

---

## Settled Decisions & Scope

See **[docs/DECISIONS.md](../DECISIONS.md)** for full decision history (Decisions 1–35), including framing, architecture, baseline drops (FedMD, CFD), and exploration scoping.

---

## Exploration Phase Mechanisms (MobileNetV2GN)

| Mechanism                                           | Status / Disposition                                                                            | Pass |
| :-------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :--- |
| Soft-voting (`entropy_weight` × `precision_weight`) | **Pass 1 Complete** (provisional pick: `ew=2.0`, `pw=0.5`, `soft_voting=true`; Decisions 33-35) | 1    |
| Capacity-EMA                                        | **Pass 2 (Next)**: Evaluate on/off for MobileNetV2GN                                            | 2    |
| Grad-norm smoothing ($\beta=0.7$)                   | **Pass 2 (Next)**: Isolation ablation                                                           | 2    |
| Client KD reg + proximal ($\mu=0.1$)                | **Pass 2 (Next)**: Evaluate necessity at lower capacity                                         | 2    |
| Dual-tier precision scaling (Formulation 3)         | **Pass 3**: Evaluate dual-tier precision scaling                                                | 3    |

---

## Deferred Sub-Details

Process questions (selection rule, sweep structure, decision rule, baseline-tuning budget) are resolved — see [DECISIONS.md](../DECISIONS.md) 2026-07-18 entry (27–32). Still open, pending Pass 1–3 results:

- **Capacity-EMA resolution**: Pass 2 answers whether EMA is in the frozen config.
- **Pareto plot**: compare FedMAQ vs pure-quant baselines (FedPAQ, DAdaQuant) at **matched bit budgets**, else accuracy-vs-compression frontier is apples-to-oranges. Not yet built.

---

## Execution Structure (tentative)

Two-phase, freeze-enforced — design summary only; concrete build steps (config-as-code registry, seed-determinism check) are tracked in [HANDOFF.md](../../HANDOFF.md) ("Immediate Next Actions").

- **Exploration phase** (now): adaptive, single-seed, mechanisms in flux. Output = one pre-registered frozen config (CIFAR-10) + baseline HP table + fixed mechanism set. Git-tag the pre-registration.
- **Confirmation phase**: config-as-code manifest → process-isolated runners → WandB (fixed project/group/tag scheme). Read-only configs after launch (hashed in manifest).

Open for restructuring: directory layout (`exploration/` vs `formal/`), manifest schema, WandB namespacing. To be planned before confirmation launch.

---

## Metrics (per `evaluation-metrics.md`)

Top-1 accuracy; CE + distillation loss; macro P/R/F1; cumulative comm (MB/GB per-client + aggregate); wall-clock; convergence curves (acc vs rounds, acc vs transmitted bytes).
