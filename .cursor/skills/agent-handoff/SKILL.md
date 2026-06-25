---
name: agent-handoff
description: >-
  Updates fedmaq-experiments/HANDOFF.md and indicates whether to hand off to
  another agent for clean context. Use when ending a work session, preparing
  handoff, handing off to a new agent, or when the user says "handoff" or
  "update handoff".
---

# Agent Handoff

End-of-session workflow for the FedMAQ multi-repo workspace. Keeps [HANDOFF.md](../../HANDOFF.md) current and indicates if a new agent session is recommended for clean context.

## When to run

- User asks to hand off, prepare handoff, or end session with continuity
- Before closing a long implementation session
- After completing a queue item or switching primary repo

## Procedure

### 1. Gather session facts

From the current conversation and git status, determine:

- **Date** (YYYY-MM-DD)
- **Last session focus** (one line)
- **Active repo** (primary repo worked in)
- **Completed** (bullets: what changed, files/areas)
- **Next task** (single concrete item from HANDOFF.md Section 6, or new item)
- **Blockers** (or "None")
- **Constraints** for next agent (optional: env, GPU, do-nots)

### 2. Update HANDOFF.md

Edit [HANDOFF.md](../../HANDOFF.md):

| Section                     | Update                                                          |
| --------------------------- | --------------------------------------------------------------- |
| Top table                   | `Last updated`, `Last session focus`, `Active repo`, `Blockers` |
| Section 4 (per-repo status) | Move items from Pending → Done where applicable                 |
| Section 6 (queue)           | Mark completed `[x]`; set **Current focus** to next task        |
| Section 7                   | Add any new env vars                                            |
| Section 10 (changelog)      | Prepend new `### YYYY-MM-DD — <title>` with 3–8 bullets         |

**Rules:**

- Keep changelog entries factual; no emojis.
- Do not remove locked decisions (Section 3) unless user explicitly changed them.
- If scope changed, update Section 3 or 5 and note in changelog.

### 3. Recommend Handoff

Indicate clearly in your final response whether you recommend handing off to another agent session to obtain clean context. 
- **Recommend Handoff** if:
  - You have completed a major task/implementation phase.
  - The conversation context has grown long/complex (which might cause slower processing or context limits).
  - The active repo or focus is changing significantly.
- **Do Not Recommend Handoff** if:
  - Only small incremental edits or minor follow-ups remain.
  - The current context is still clean and relevant.

### 4. Confirm to user

After updating the file, tell the user:

1. `HANDOFF.md` was updated (mention sections touched).
2. Your recommendation on whether they should hand off to a new agent session for clean context.
3. Optional: commit HANDOFF.md with their other changes.

## Related files

- Canonical handoff: [HANDOFF.md](../../HANDOFF.md)
- Domain rules: [`.cursor/rules/`](../rules/)
- Sibling AGENTS.md files reference HANDOFF.md from other repos
