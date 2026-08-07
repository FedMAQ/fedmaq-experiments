# fedmaq-experiments

FedMAQ thesis experiments: phased uv monorepo (Flower, Hydra, PyTorch, WandB).

## Structure

```text
conf/                 # Hydra config groups
src/fedmaq/
  core/               # shared Flower simulation & telemetry utilities
  baselines/          # SOTA baseline implementations
.claude/rules/        # always-loaded domain rules (canonical for the workspace)
docs/adr/             # every decision, one file per decision
docs/agents/          # consult-on-demand reference (execution model, issue tracker)
```

## Setup

```bash
uv sync
uv run pytest
```

## Agent onboarding

1. Read [CLAUDE.md](CLAUDE.md), the agent entry point.
2. Read [CONTEXT.md](CONTEXT.md) for the glossary, the sibling-repo authority map, and
   the precedence order when two sources disagree.

Current project state lives in pinned GitHub Issues, never in a tracked file.

**Sibling repos:** `fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations`.
