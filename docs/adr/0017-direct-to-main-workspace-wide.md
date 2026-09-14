# ADR-0017: Direct-to-main across the workspace; the PR workflow is retired

**Status**: Accepted
**Date**: 2026-08-25

## Context

All six repos' `AGENTS.md` documented the same standard PR workflow (`git push
-u origin HEAD && gh pr create --fill`, squash-merge, delete branch). Practice
diverged from that document in half the workspace: `fedmaq-analyses`,
`fedmaq-presentations`, and `fedmaq-journal-article` have zero merged PRs each
across their full history — every commit went straight to `main`. The other
three (`fedmaq-experiments`, `fedmaq-manuscript`, `fedmaq-literature`) do have
PR history, but `fedmaq-manuscript`'s own wayfinder issue #6 called it a
"direct-to-`main` policy" in writing, contradicting its own `AGENTS.md` and
its 11 merged PRs.

Checked before deciding, not assumed: no repo's CI runs on `pull_request` or
`push` — `fedmaq-experiments/.github/workflows/ci.yml`, the only CI in the
workspace, is `workflow_dispatch`-only. No repo has branch protection on
`main` (the private repos can't enable it on the current GitHub plan;
`fedmaq-presentations`, the one public repo, has it off). **The PR workflow
gated nothing automated.** It was ceremony layered over a solo-author
workspace where the real review already happens at the diff and the
chapter/session boundary, independent of GitHub's PR UI — `fedmaq-manuscript`'s
own `AGENTS.md` already says "stop for user review before proceeding to the
next chapter," which is the actual checkpoint.

## Decision

**Direct-to-`main` is the standard workflow, workspace-wide, replacing the PR
workflow line in all six `AGENTS.md` files.** Push straight to `main`. No
branch, no PR, no squash-merge ceremony.

What the PR/squash step was actually doing — collapsing agent work-in-progress
into one clean commit before it landed on `main` — is replaced by the same
discipline stated directly: **commit clean before pushing** (amend/rebase
locally rather than pushing a string of WIP commits), and run whatever
per-repo check already exists first (`just check` in
`fedmaq-experiments`/`fedmaq-analyses`/`fedmaq-literature`; compile-clean in
`fedmaq-manuscript`/`fedmaq-journal-article` per their writing-style rules).
None of those were ever PR-enforced — they were always a local step — which is
exactly why they survive this change unchanged.

`fedmaq-manuscript`'s wayfinder #6 parenthetical, written before this ADR
existed, called manuscript direct-to-main while its `AGENTS.md` and history
said otherwise. That statement is now correct by adoption rather than stale;
it does not need editing.

## Consequences

- All six `AGENTS.md` files: the PR-workflow line is replaced by a
  direct-to-main line plus the local pre-push check for that repo.
- Citing work in ADRs and wayfinder issues continues to use commit SHAs, not
  PR numbers — the existing convention in every repo's "done" markers already
  does this (e.g. `fedmaq-manuscript`'s port-ticket table cites raw SHAs).
- If a repo later gets meaningful automated CI (tests that actually gate
  merges) or a second contributor, that repo should revisit this ADR rather
  than silently drift back to PRs.

## Amendment (2026-09-14): the push authorization is standing

This ADR said to push straight to `main` but never said whether an agent may
do so on its own. Agents default to confirming an outward-facing action each
time, so in practice every push stopped for approval — approval the author
granted every time, having already granted it in this ADR.

**The author's push authorization is standing.** An agent pushes finished work
to `main` without asking, in every repo in the workspace. Withdrawing it for a
particular change is the author's call, stated at the time.

The pre-push discipline above is unchanged and is what makes this safe: commit
clean, and run the repo's own check first. The gate was always the check and
the diff, never the pause to ask.

Each `AGENTS.md` carries this authorization inline rather than by reference,
because an agent decides whether to push from what is already in its context
and will not read this ADR first.
