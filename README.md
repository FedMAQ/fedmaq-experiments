# fedmaq-experiments

FedMAQ thesis experiments: phased uv monorepo (Flower, Hydra, PyTorch, WandB).

## Structure

```text
conf/                 # Hydra config groups
src/fedmaq/
  core/               # shared Flower simulation & telemetry utilities
  baselines/          # SOTA baseline implementations
.claude/rules/         # canonical thesis domain context for workspace
.claude/project/      # experiment + baseline registries
```

## Setup

```bash
uv sync
uv run pytest
```

## Agent onboarding

1. Read [CLAUDE.md](CLAUDE.md), the agent entry point. Domain rules live in `.claude/rules/`.
2. Read [.claude/project/workspace_map.md](.claude/project/workspace_map.md) for the sibling-repo map.

**Sibling repos:** `fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations`.
