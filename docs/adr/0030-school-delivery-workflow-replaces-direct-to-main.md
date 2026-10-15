# ADR-0030 — Thesis repositories deliver through the School workflow; direct-to-main is retired

**Status**: Accepted · 2026-10-15
**Supersedes**: [ADR-0017](0017-direct-to-main-workspace-wide.md), including its 2026-09-14 standing-push amendment.
**Decided by**: the thesis author, 2026-10-15.

## Context

ADR-0017 made direct-to-`main` the workflow in the six thesis repositories and gave agents standing authority to push to `main`. The School workspace has since adopted one delivery workflow for every Git repository beneath it, `School/docs/agent-delivery.md`. The School `AGENTS.md` sends agents to that workflow before any change.

The two rules conflict. The School workflow requires a task branch, an isolated worktree and, for a remote-backed repository, a pull request to `main`. It lands a change only when the author asks. The six repository `AGENTS.md` files still stated ADR-0017's rule. An agent therefore loaded two delivery rules for the same change, and the most recent session had to stop and ask which one applied.

## Decision

**The School delivery workflow governs delivery in every FedMAQ thesis repository.** Each repository's `AGENTS.md` points to it and keeps only that repository's own pre-commit check.

ADR-0017's direct-to-`main` rule and its standing push authorization are withdrawn. An agent pushes its task branch to open the pull request, and it lands the change only on the author's explicit request.

## Consequences

- The six `AGENTS.md` files (experiments, analyses, literature, manuscript, journal-article, presentations) replace the ADR-0017 line with a pointer to the School workflow plus the repository's check.
- The pre-commit checks ADR-0017 named (`just check`, compile-clean builds) remain. They now run before each commit on the task branch.
- ADR-0017 still holds that no repository has CI on `pull_request` and no `main` has branch protection. A pull request here is a review point, not an automated gate.
- Commit SHAs remain the citation convention in ADRs and issues. A PR number may be cited alongside them.
