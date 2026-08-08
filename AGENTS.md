# FedMAQ Experiments

- Read `CONTEXT.md` before naming shared domain concepts or resolving cross-repository authority.
@.agent/rules/experiment-design.md
@.agent/rules/engineering.md
- Agents do not run experiments; emit paste-ready JupyterHub commands and await user-supplied results.
- Do not edit configurations frozen downstream of the `pre-registration` tag.
- Read `docs/adr/` for durable decisions and `docs/agents/` for task-specific reference.
- GitHub Issues are the sole live-state record; do not create tracked handoffs, changelogs, or status files.
- Treat generated freeze snapshots as read-only.
