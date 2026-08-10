---
name: comment-hygiene
description: Audit and prune comment or docstring bloat.
disable-model-invocation: true
---

# Comment Hygiene

Use this skill when the user asks to audit, prune, debloat, or review comments or docstrings.

1. **Scope.** Read repository instructions and inspect `git status`. Audit tracked source and test files by default; include documentation or configuration only when the request names it. Respect frozen and historical material.
   - **Done when:** every included file class and exclusion is explicit.

2. **Inventory.** Find comment- and docstring-heavy files, then inspect each candidate in context. Prioritize generated-looking narration, numbered steps, repeated explanations, and stale implementation history.
   - **Done when:** every candidate is classified rather than judged from a line count alone.

3. **Apply the evidence test.** Keep a comment only when its removal would hide a non-inferable invariant, rationale, external constraint, public contract, or workaround. Delete narration that code, names, types, tests, or a nearby canonical document already make clear. Collapse a comment only when a shorter statement preserves unique information.
   - **Done when:** every retained comment answers what misunderstanding or regression it prevents.

4. **Adversarial check.** Challenge each keep/delete decision: could a competent reader infer it from the code? Does removal endanger determinism, experiment validity, compatibility, failure recovery, or counterintuitive correctness? Preserve the smallest comment that carries a real risk.
   - **Done when:** all risky removals have been reconsidered and every exception has a concrete reason.

5. **Validate.** Review the diff for comment-only scope, run the repository's relevant static or test checks, and report removed material, preserved exceptions, and blockers.
   - **Done when:** the diff is clean and every audited candidate is removed, retained, or explicitly deferred.
