# Agentic Context

- Keep the universal core in `AGENTS.md`; route activity-specific guidance through plain pointers to on-demand rules and skills.
- Keep `AGENTS.md` free of `@` imports; `CLAUDE.md` remains the thin `@AGENTS.md` wrapper.
- Give each shared skill one owner at `.agents/skills/<name>/SKILL.md`; reserve `.claude/` for Claude-specific extensions.
- After changing skill locations or names, run `pytest tests/test_agent_context.py` and update active references.
