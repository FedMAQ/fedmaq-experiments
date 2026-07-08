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

1. Read [HANDOFF.md](HANDOFF.md) (workspace handoff; update via `agent-handoff` skill each session).
2. Read [AGENTS.md](AGENTS.md). Domain rules live in `.claude/rules/`.

**Sibling repos:** `fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations`.
