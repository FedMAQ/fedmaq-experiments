# FedMAQ Experiments — Agent Instructions

FedMAQ thesis experiments: a uv monorepo (Flower, Hydra, PyTorch, WandB) implementing communication-efficient federated learning via multi-adaptive quantization and knowledge distillation. This is the domain "hub" repo in a 5-repo thesis workspace.

## Constraints

- Don't add top-level `reproductions/` packages — use `src/fedmaq/baselines/`.
- Don't duplicate thesis domain content outside `fedmaq-experiments/.claude/rules/`.
- No per-algorithm branching in `strategy.py`/`client.py` — dispatch is hook-based (`core/client_hooks/`, `core/strategy_hooks/`); don't reintroduce `alg_name` string-dispatch or if/elif chains keyed on algorithm name.
- Don't hand-edit `.claude/project/*.md` registries out of band — update them in the same commit/session as the change they describe.
- Don't recreate a second "session narrative" file — `changelog.md` (trimmed, milestones only) + claude-mem cover that; a new HANDOFF-shaped file would reintroduce the exact duplication just removed.

## Always-active rules

@.claude/rules/project-overview.md
@.claude/rules/repo-preferences.md
@.claude/rules/manuscript-alignment.md
@.claude/rules/agent-delegation.md

## Task-specific rules

Not imported — read the relevant file when the task matches:

| Rule                                   | Read when...                                                |
| -------------------------------------- | ----------------------------------------------------------- |
| `.claude/rules/ablation-phases.md`     | working across the 4 research phases                        |
| `.claude/rules/baselines.md`           | adding or porting a SOTA baseline                           |
| `.claude/rules/datasets-simulation.md` | touching dataset or heterogeneity config                    |
| `.claude/rules/evaluation-metrics.md`  | wiring up WandB logging                                     |
| `.claude/rules/flower-patterns.md`     | writing Flower client/server/strategy code                  |
| `.claude/rules/hydra-config.md`        | editing `conf/`                                             |
| `.claude/rules/tech-stack.md`          | choosing or adding a dependency                             |
| `.claude/rules/agent-memory.md`        | deciding whether to trust remembered vs. current repo state |

Skills (`.claude/skills/`) are auto-discovered by Claude Code — no manual index needed here.

## Project registries

| File                                     | Contents                                       |
| ---------------------------------------- | ---------------------------------------------- |
| `.claude/project/workspace_map.md`       | Sibling-repo table and tooling migration state |
| `.claude/project/env_vars.md`            | Environment variables and secrets              |
| `.claude/project/baseline_registry.md`   | Baseline algorithm implementation status       |
| `.claude/project/experiment_registry.md` | Completed experiment runs                      |
| `.claude/project/changelog.md`           | Milestone-level session history                |

## Sibling repos

`fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations` — see [.claude/project/workspace_map.md](.claude/project/workspace_map.md) for the workspace map.
