---
name: jupyterhub-golden-gate
description: Guide a user through exact-commit GPU golden capture/compare runs on this FedMAQ project's remote JupyterHub checkout, including clean-state handling, evidence collection, and failure triage.
---

# JupyterHub Golden Gate

Use this skill for this FedMAQ repository when a user must run the user-run
golden harness on JupyterHub. The agent prepares paste-ready commands; the
user runs them and returns the evidence.

## Required decision before running

Identify two exact commits:

- `BASELINE`: the parent or otherwise documented pre-change commit.
- `CANDIDATE`: the exact pushed commit containing the change under test.

A later ticket with its own fresh-gate requirement needs a new capture at the
commit immediately before that ticket and a new compare at the ticket's final
commit; a later gate does not retroactively validate an earlier change.

## Prepare the JupyterHub checkout

Give the user commands that:

1. Enter `~/fedmaq-experiments` and run `git fetch origin --prune`.
2. Check `git status --short --branch` before switching commits. Stop if there
   are modified tracked files or unknown source artifacts.
3. Preserve existing generated directories outside the repository. Common paths
   are `logs/`, `outputs/`, and `scripts/analysis_output/`; never use `git
   clean` for this workflow.
4. Keep command logs outside the repository. The FedMAQ manifest recognizes
   `outputs/` and `scripts/analysis_output/` as generated artifacts, but a
   `logs/` directory is a source-tree change for provenance purposes.

Preparation is complete only when the intended commits are available remotely,
the checkout has no modified tracked files or unknown source artifacts, and the
evidence directory is outside the repository.

The detached state from `git switch --detach <commit>` is appropriate for
experiments. The candidate must be available from the remote before starting.

## Capture then compare

Use an evidence directory outside the checkout and record the actual hashes:

```bash
cd ~/fedmaq-experiments
git fetch origin --prune
mkdir -p ~/fedmaq-golden-EVIDENCE

git switch --detach BASELINE_SHA
git rev-parse HEAD | tee ~/fedmaq-golden-EVIDENCE/00-baseline-commit.txt
uv run python scripts/golden_diff.py capture \
  2>&1 | tee ~/fedmaq-golden-EVIDENCE/01-capture.log

git switch --detach CANDIDATE_SHA
git rev-parse HEAD | tee ~/fedmaq-golden-EVIDENCE/02-candidate-commit.txt
uv run python scripts/golden_diff.py compare \
  2>&1 | tee ~/fedmaq-golden-EVIDENCE/03-compare.log
```

Do not remove the capture outputs between `capture` and `compare`. The current
`scripts/golden_diff.py` clears each target and the known persistent model state
before each algorithm; inspect it before adding manual cleanup commands.

Execution is complete only when capture has finished at `BASELINE`, compare has
finished at `CANDIDATE`, and both recorded hashes match the intended commits.

## Evidence and stopping rules

Ask the user to return the four evidence files, or their complete contents:

- baseline hash file;
- complete capture log ending in the capture-complete message;
- candidate hash file;
- complete compare log.

Treat the gate as passed only when the recorded hashes are the intended pair,
each expected algorithm reports an exact match, and the compare log ends with
`All golden diffs passed.` A failure, missing output, wrong commit, or
unexpected dirty-working-tree warning is a stop condition. Preserve and inspect
the original evidence; do not recapture merely to erase a failure.

Routine upstream deprecation or framework future warnings do not by themselves
fail a bit-exact gate, but distinguish them from errors, dirty-tree provenance
warnings, and golden-diff failures.

Generated outputs and logs are evidence, not source changes. Do not commit them
unless this repository explicitly requires a tracked artifact. A formal
verification commit is not normally needed; record the result in the issue or
handoff after local tests and the issue's other acceptance criteria pass.

The verification handoff is complete only when the original evidence is
preserved, the result is classified as passed or failed, and warnings are
separated from actual gate failures.
