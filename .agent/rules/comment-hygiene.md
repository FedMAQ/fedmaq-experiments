# Comment Hygiene

- Treat comments and docstrings as evidence: retain only non-inferable rationale, invariants, external constraints, or public contracts.
- Express behaviour through names, types, tests, and structure; omit narration, step labels, and restatements of nearby code.
- Before finishing a changed source or test file, prune comments whose removal leaves the behaviour and intent clear.
- Preserve comments that protect determinism, experiment validity, compatibility workarounds, failure recovery, or counterintuitive correctness.
- Preserve frozen configuration, ADR, and historical-record comments unless the task explicitly includes them.
- Read `.agents/skills/comment-hygiene/SKILL.md` when asked to audit or prune comments repository-wide.
