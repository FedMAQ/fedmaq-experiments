---
name: agent-handoff
description: >-
  Logs a changelog.md milestone entry (only if warranted) and reminds the
  agent to update touched registries before ending a session. Use when the
  user says "handoff" or "update handoff", or before closing a long
  implementation session.
---

# Agent Handoff

End-of-session checklist. There is no more standalone HANDOFF.md — session
continuity is covered by `.claude/project/changelog.md` (milestones only) and
claude-mem (session-level recall).

## When to run

- User asks to hand off, prepare handoff, or end session with continuity
- Before closing a long implementation session
- After completing a queue item or switching primary repo

## Procedure

### 1. Assess whether this session hit a milestone

Per the trim policy in `changelog.md` — merged PR, architecture shift, phase
completion, baseline port, etc. Routine incremental edits do **not** qualify;
claude-mem already covers that granularity.

If it qualifies, prepend a `### YYYY-MM-DD — <title>` entry to
[changelog.md](../../project/changelog.md) with 3-8 factual bullets (no
emojis). Do not edit existing historical entries.

### 2. Update touched registries

Check whether this session touched any of:

- `.claude/project/baseline_registry.md`
- `.claude/project/experiment_registry.md`
- `.claude/project/workspace_map.md`
- `.claude/project/env_vars.md`

If so, confirm they reflect the current state before ending — don't leave a
registry stale relative to the code/config change it describes.

### 3. Confirm to user

Tell the user:

1. Whether a changelog entry was added (and why/why not).
2. Which registries, if any, were updated.

## Related files

- [changelog.md](../../project/changelog.md)
- [workspace_map.md](../../project/workspace_map.md)
- Domain rules: [`.claude/rules/`](../../rules/)
