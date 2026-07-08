# Agent Delegation

Guidance for when to delegate work rather than doing it all inline in the main conversation:

- **Large or exploratory navigation** (e.g. surveying `src/fedmaq/` phases, baseline layout, or an unfamiliar part of the codebase) — consider using a subagent (Claude Code's Task/Agent tool) to search and report back, keeping the main context focused.
- **Long shell-heavy sequences** (`uv run`, Docker, Hydra multirun sweeps, WandB CLI) — fine to run inline, but if a sweep is exploratory/multi-step, consider a subagent to isolate the noise from the main conversation.
- **Before merging large baseline ports or phase refactors** — consider an independent review pass (e.g. the `code-review` skill/command) before finalizing, rather than self-certifying.
- **Library API lookups** (Flower, Hydra, PyTorch) — if a documentation MCP server (e.g. `context7`) is configured at the user/global Claude Code level, use it; otherwise fall back to web search or reading installed package source directly.

Before ending a session, run the **agent-handoff** skill to update [HANDOFF.md](../../HANDOFF.md) and indicate whether to hand off to another agent for clean context.
