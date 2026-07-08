# FedMAQ Experiments — Agent Instructions

FedMAQ thesis experiments: a uv monorepo (Flower, Hydra, PyTorch, WandB) implementing communication-efficient federated learning via multi-adaptive quantization and knowledge distillation. This is the domain "hub" repo in a 5-repo thesis workspace — see `@HANDOFF.md` for the cross-repo picture.

Read `@AGENTS.md` for the resource index (registries, skills, commands).

## Domain rules

> [!NOTE]
> Claude Code's `@import` is unconditional — every file below loads on every session, unlike the old Cursor setup where several of these were scoped by `globs`/`alwaysApply: false` and only loaded when touching matching files (e.g. `flower-patterns` was scoped to `src/**/*.py`). In practice most globs already covered nearly all of `src/**`/`conf/**`, so the effective difference is small, but keep these files tight since they're now always in context.

@rules/project-overview.md
@rules/repo-preferences.md
@rules/manuscript-alignment.md
@rules/agent-delegation.md
@rules/ablation-phases.md
@rules/baselines.md
@rules/datasets-simulation.md
@rules/evaluation-metrics.md
@rules/flower-patterns.md
@rules/hydra-config.md
@rules/tech-stack.md

## Sibling repos

`fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations` — see `@HANDOFF.md` §2 for the workspace map.
