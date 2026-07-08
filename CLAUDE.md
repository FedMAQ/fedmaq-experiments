# FedMAQ Experiments — Agent Instructions

FedMAQ thesis experiments: a uv monorepo (Flower, Hydra, PyTorch, WandB) implementing communication-efficient federated learning via multi-adaptive quantization and knowledge distillation. This is the domain "hub" repo in a 5-repo thesis workspace.

@HANDOFF.md

## Always-active rules

@.claude/rules/project-overview.md
@.claude/rules/repo-preferences.md
@.claude/rules/manuscript-alignment.md
@.claude/rules/agent-delegation.md

## Task-specific rules

Not imported — read the relevant file when the task matches:

| Rule                                     | Read when...                                    |
| ----------------------------------------- | ------------------------------------------------ |
| `.claude/rules/ablation-phases.md`       | working across the 4 research phases            |
| `.claude/rules/baselines.md`             | adding or porting a SOTA baseline               |
| `.claude/rules/datasets-simulation.md`   | touching dataset or heterogeneity config        |
| `.claude/rules/evaluation-metrics.md`    | wiring up WandB logging                         |
| `.claude/rules/flower-patterns.md`       | writing Flower client/server/strategy code      |
| `.claude/rules/hydra-config.md`          | editing `conf/`                                 |
| `.claude/rules/tech-stack.md`            | choosing or adding a dependency                 |

Skills (`.claude/skills/`) and slash commands (`.claude/commands/`) are auto-discovered by Claude Code — no manual index needed here.

## Sibling repos

`fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations` — see `@HANDOFF.md` §2 for the workspace map.
