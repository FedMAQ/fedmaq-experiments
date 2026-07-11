---
name: align-manuscript
description: >-
  Synchronizes experiment code & config variables to LaTeX thesis definitions.
  Use when the manuscript changes hyperparameters, formulas, or system
  definitions and the codebase needs to be checked/updated to match.
---

# Align Manuscript

1. Check the LaTeX manuscript files in [fedmaq-manuscript/](../../../../fedmaq-manuscript/) (e.g., `chapter_*.tex`) to identify changes in hyperparameters, formulas, or system definitions.
2. Read the canonical rules in [.claude/rules/manuscript-alignment.md](../../rules/manuscript-alignment.md) to check active constraints.
3. Validate and synchronize parameters in [default.yaml](../../../conf/experiment/default.yaml) and [preliminary.yaml](../../../conf/experiment/preliminary.yaml) (such as client counts $K$, batch sizes $B$, epochs $E$, learning rate decay $\gamma$, or proxy size $\|D_{pub}\|$).
4. Verify that the client delay simulation and server distillation delay match the mathematical formulations defined in the manuscript. Dispatch is hook-based — check `core/client_hooks/` and `core/strategy_hooks/` per algorithm, not a monolithic `client.py`/`strategy.py`.
5. Check that precision-scaling allocations inside the strategy hooks match the soft quality targets (Formulations 0 to 4).
6. Run `uv run pytest` to verify that all mathematical formulations and telemetries pass their test assertions.
