# FedMAQ Experiments — Agent Instructions

Multi-adaptive quantization and knowledge distillation for memory-constrained
federated learning under non-IID data. A uv monorepo (Flower, Hydra, PyTorch,
WandB). One of six repos in the FedMAQ workspace, and the domain **hub** — siblings
index this repo's rules rather than duplicating them.

**Read `CONTEXT.md` first.** It holds the canonical glossary, the authority map
naming which repo owns what, and the precedence order for when two sources disagree.

## Rules

@.claude/rules/experiment-design.md
@.claude/rules/engineering.md

## Two constraints that govern everything

- **An agent cannot run experiments.** Every reported run executes on a Linux
  allocation reached through JupyterHub. Emit paste-ready commands; the user runs
  them and pastes results back. Never write a plan step where an agent dispatches a
  sweep or polls for completion.
- **Nothing downstream of the `pre-registration` tag may edit a frozen config.**
  Thirteen files under `conf/` are frozen. Treat them as read-only.

## Where things live

| | |
| :-- | :-- |
| Decisions, and why | `docs/adr/` — the sole decision record |
| Dispatch order, operational controls | `docs/agents/execution-model.md` (consult on demand) |
| What is true *right now* | pinned GitHub Issues — never a tracked file |
| Frozen configuration snapshot | `docs/freeze/resolved_configs.yaml` (generated; do not hand-edit) |

Run counts, dispatch state and sync status live only in Issues. **A number in a
tracked file is stale by construction** — that is the failure this layout exists to
prevent ([ADR-0014](docs/adr/0014-agentic-context-layout.md)).

## Agent skills

### Issue tracker

GitHub Issues for [FedMAQ/fedmaq-experiments](https://github.com/FedMAQ/fedmaq-experiments), via `gh`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
