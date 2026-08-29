# FedMAQ Experiments

- Read `CONTEXT.md` before naming shared domain concepts or resolving cross-repository authority.
@.agents/rules/experiment-design.md
@.agents/rules/engineering.md
@.agents/rules/comment-hygiene.md
@.agents/rules/agentic-context.md
- Agents do not run experiments; emit paste-ready JupyterHub commands and await user-supplied results.
- Do not edit configurations frozen downstream of the `pre-registration` tag.
- Read `docs/adr/` for durable decisions and `docs/agents/` for task-specific reference.
- GitHub Issues are the sole live-state record; do not create tracked handoffs, changelogs, or status files.
- When the user explicitly requests sync, wrap-up, push, or GitHub issue reconciliation, treat the named GitHub actions as authorized and proceed without redundant confirmation, subject to platform/tool permission gates.
- Direct-to-`main`, no PRs (ADR-0017). Commit clean before pushing; run `just check` first.
- Run `just check` before staging or committing changes.
- Treat generated freeze snapshots as read-only.
