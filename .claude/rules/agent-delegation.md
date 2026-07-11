# Agent Delegation

Guidance for when to delegate work rather than doing it all inline in the main conversation:

- **Large or exploratory navigation** (e.g. surveying `src/fedmaq/` phases, baseline layout, or an unfamiliar part of the codebase): delegate to a subagent (Claude Code's Task/Agent tool) to search and report back, keeping the main context focused.
- **Exploratory or multi-step shell sequences** (Hydra multirun sweeps, long Docker/WandB CLI runs): delegate to a subagent to isolate the noise from the main conversation. Short one-off commands run inline.
- **Before merging large baseline ports or phase refactors:** run an independent review pass (e.g. the `code-review` skill/command) before finalizing — do not self-certify.
- **Library API lookups** (Flower, Hydra, PyTorch): use the configured documentation MCP server (e.g. `context7`) if available; otherwise fall back to web search or reading installed package source directly.

Before ending a session, run the **agent-handoff** skill to log a changelog milestone (if warranted) and confirm touched registries are up to date.
