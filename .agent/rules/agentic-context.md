# Agentic Context

- Keep shared instructions in `AGENTS.md` and `.agent/rules/`; `CLAUDE.md` imports `AGENTS.md`.
- Create each shared skill only at `.agents/skills/<name>/SKILL.md`.
- Reserve `.claude/` for genuinely Claude-specific extensions; never copy a shared skill there.
- After changing skill locations, run `pytest tests/test_agent_context.py`.
