# FedMAQ Experiments

- Read `CONTEXT.md` before naming shared terms or authority.
- Preserve unrelated work: inspect status and stage only task-owned paths.
- GitHub Issues own live state. Read `docs/agents/issue-tracker.md` before issue work and `triage-labels.md` only when triaging.
- Direct-to-`main` follows ADR-0017; run `just check` before staging or committing.
- For explicit matrix dispatch, read `.agents/skills/user-run-matrix/SKILL.md`; prepare commands and wait for user evidence.
- Read `.agents/skills/jupyterhub-golden-gate/SKILL.md` for exact-commit golden assurance work.
- Read `.agents/skills/sweep-recovery/SKILL.md` for failed runs or status recovery.
- Read `.agents/rules/experiment-design.md` for method, baseline, metric, or frozen-config work; read `engineering.md` for code, config, tests, or runners.
- Read `.agents/rules/comment-hygiene.md` for source/test comments; use its skill for repository-wide pruning.
- For agent-context work, read `.agents/rules/agentic-context.md` and `.agents/skills/docs-audit/SKILL.md`; keep AGENTS import-free.
- Use `docs/adr/` for durable rationale and `docs/agents/` for reference.
- Do not change frozen config, freeze artifacts, evidence, or scientific claims without explicit authorization.
- When a Windows Python launcher fails, read `docs/agents/windows-python-tooling.md` for the module-execution and targeted-repair procedure.
