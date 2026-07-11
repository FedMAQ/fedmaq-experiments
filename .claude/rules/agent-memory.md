# Agent Memory

Durable facts (hyperparameters, architecture decisions, registry status) live
in git-tracked repo files — `.claude/rules/`, `.claude/project/*.md` — and are
authoritative. claude-mem is for session recall ("did we try X already",
cross-session narrative continuity), not a substitute for reading current
registry or rule state.

Before acting on a claude-mem recollection that names a specific file,
function, or config value, verify it still exists/matches current repo state
(same caveat as the global memory instructions, restated here for repo-local
emphasis).
