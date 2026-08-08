# fedmaq-experiments

FedMAQ thesis experiments: phased uv monorepo (Flower, Hydra, PyTorch, WandB).

## Structure

```text
conf/                 # Hydra config groups
src/fedmaq/
  core/               # shared Flower simulation & telemetry utilities
  baselines/          # SOTA baseline implementations
.agent/rules/         # stable tool-neutral domain rules
docs/adr/             # every decision, one file per decision
docs/agents/          # consult-on-demand reference (execution model, issue tracker)
```

## Setup

```bash
uv sync
uv run pytest
```

## Agent onboarding

1. Read [AGENTS.md](AGENTS.md), the canonical agent instruction index.
   [CLAUDE.md](CLAUDE.md) imports the same instructions for Claude Code.
2. Read [CONTEXT.md](CONTEXT.md) for the glossary, the sibling-repo authority map, and
   the precedence order when two sources disagree.

Current project state lives in pinned GitHub Issues, never in a tracked file.

**Sibling repos:** `fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations`.
