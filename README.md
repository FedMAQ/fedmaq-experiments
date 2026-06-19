# fedmaq-experiments

FedMAQ thesis experiments: phased uv monorepo (Flower, Hydra, PyTorch, WandB).

## Structure

```text
conf/                 # Hydra config groups
src/fedmaq/
  core/               # shared utilities
  phase1_env/         # baseline FL environment
  phase2_quant/       # quantization ablations
  phase3_kd/          # server-side KD
  phase4_benchmark/   # unified benchmarking
  baselines/          # SOTA baseline implementations
.cursor/rules/        # canonical thesis domain context for workspace
.cursor/project/      # experiment + baseline registries
```

## Setup

```bash
uv sync
uv run pytest
```

## Agent onboarding

1. Read [HANDOFF.md](HANDOFF.md) (workspace handoff; update via `agent-handoff` skill each session).
2. Read [AGENTS.md](AGENTS.md). Domain rules live in `.cursor/rules/`.

**Sibling repos:** `fedmaq-literature`, `fedmaq-analyses`, `fedmaq-manuscript`, `fedmaq-presentations`.
