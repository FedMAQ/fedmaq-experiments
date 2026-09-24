# FedMAQ Experiments

- Take terminology and authority from this repository's docs, ADRs, configuration,
  and GitHub Issues. Read relevant ADRs and current issues before methodology,
  protocol, or result claims.
- Preserve unrelated work: inspect status and stage only task-owned paths.
- GitHub Issues own live state. Read `docs/agents/issue-tracker.md` before issue work and `triage-labels.md` only when triaging.
- Direct-to-`main` follows ADR-0017; run `just check` before staging or committing. Push authorization is standing: push finished work without asking.
- For a long foreground verification command, attach once and wait up to ten minutes before one status sample; retain the same process and do not start a duplicate while it runs.
- For explicit matrix dispatch, read `.agents/skills/user-run-matrix/SKILL.md`; prepare commands and wait for user evidence.
- Read `.agents/skills/pre-dispatch-assurance/SKILL.md` to guide the author through current-candidate local smoke evidence and JupyterHub golden repeatability.
- Read `.agents/skills/jupyterhub-golden-gate/SKILL.md` for exact-commit golden assurance work.
- Read `.agents/skills/sweep-recovery/SKILL.md` for failed runs or status recovery.
- Read `.agents/rules/experiment-design.md` for method, baseline, metric, or frozen-config work; read `engineering.md` for code, config, tests, or runners.
- Read `.agents/rules/comment-hygiene.md` for source/test comments; use its skill for repository-wide pruning.
- Keep `AGENTS.md` import-free. For instruction edits, consult
  `.agents/rules/agentic-context.md`.
- Use `docs/adr/` for durable rationale and `docs/agents/` for reference.
- Do not change frozen config, freeze artifacts, evidence, or scientific claims without explicit authorization.
- When a Windows Python launcher fails, read `docs/agents/windows-python-tooling.md` for the module-execution and targeted-repair procedure.
