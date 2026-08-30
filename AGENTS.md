# FedMAQ Experiments

- Read `CONTEXT.md` before naming shared domain concepts or resolving cross-repository authority.
- Preserve unrelated work; inspect status before editing and stage only task-owned paths.
- Treat GitHub Issues as the live record for specifications and task state; read `docs/agents/issue-tracker.md` before issue work.
- Keep one canonical triage state on active request issues; read `docs/agents/triage-labels.md` when triaging.
- Direct changes to `main` follow ADR-0017; run `just check` before staging or committing.
- Agents prepare commands for the user; read `.agents/skills/user-run-matrix/SKILL.md` for explicit matrix dispatch and wait for returned evidence.
- Read `.agents/skills/jupyterhub-golden-gate/SKILL.md` for exact-commit golden assurance work.
- Read `.agents/skills/sweep-recovery/SKILL.md` when a matrix run dies or status needs recovery.
- Read `.agents/rules/experiment-design.md` for method, baseline, metric, or frozen-configuration work.
- Read `.agents/rules/engineering.md` for source, configuration, test, or runner changes.
- Read `.agents/rules/comment-hygiene.md` when changing source or test comments; use `.agents/skills/comment-hygiene/SKILL.md` for a repository-wide audit or pruning request.
- Read `.agents/rules/agentic-context.md` and `.agents/skills/docs-audit/SKILL.md` for agent-context work; use the validator and keep AGENTS free of imports.
- Read the relevant `docs/adr/` record for durable decisions and `docs/agents/` reference for repository boundaries or execution details.
- Keep frozen configurations, generated freeze snapshots, evidence, and author-owned scientific claims unchanged unless explicitly authorized.
