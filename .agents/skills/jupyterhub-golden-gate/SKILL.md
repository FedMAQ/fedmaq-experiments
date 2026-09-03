---
name: jupyterhub-golden-gate
description: Guide a user through the GPU golden assurance runs on this FedMAQ project's remote JupyterHub checkout — the repeatability release gate and the old-to-new transition diagnostic — including clean-state handling, evidence collection, and failure triage.
---

# JupyterHub Golden Gate

Use this skill for this FedMAQ repository when a user must run the user-run
golden harness on JupyterHub. The agent prepares paste-ready commands; the
user runs them and returns the evidence.

## The two operations are not interchangeable

`scripts/golden_diff.py` exposes two comparisons with different standing. Choose
deliberately; running the wrong one is the most common failure of this workflow.

- **`repeatability` is the release gate.** It runs each golden algorithm twice at
  the *same* candidate commit, into `capture_a/` and `capture_b/`, and requires
  matching provenance plus bit-exact output. It is the only operation that emits
  PASS and the only one that exits non-zero on failure. It needs one commit.
- **`transition` is a diagnostic and can never pass.** It compares a `capture`
  taken at an older commit against a fresh run at the candidate, classifies what
  changed, and records it. Intentional RNG, packing, and FedKD lifecycle repairs
  are *supposed* to differ, so a match is not evidence of correctness and a
  difference is not a failure. Its reports carry `"pass": false` and
  `"pass_eligible": false` by construction. It needs two commits.

`compare` is a backward-compatible alias for `transition`. It does not perform
the release gate. If a ticket or older handoff says "capture at BASELINE, compare
at CANDIDATE, expect all diffs to pass," that instruction predates the split —
run `repeatability` for the gate and treat the compare as diagnostic only.

## Required decision before running

For the **release gate**, identify one commit:

- `CANDIDATE`: the exact pushed commit whose freeze is being certified.

For a **transition diagnostic**, identify two:

- `BASELINE`: the parent or otherwise documented pre-change commit.
- `CANDIDATE`: the exact pushed commit containing the change under test.

A later ticket with its own fresh-gate requirement needs a new gate run at that
ticket's final commit; a later gate does not retroactively validate an earlier
change.

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

## Run the release gate

Both captures must come from the same commit and the same clean tree. Switching
commits, editing files, or changing GPU allocation between them invalidates the
gate rather than failing it.

```bash
cd ~/fedmaq-experiments
git fetch origin --prune
mkdir -p ~/fedmaq-golden-EVIDENCE

git switch --detach CANDIDATE_SHA
git rev-parse HEAD | tee ~/fedmaq-golden-EVIDENCE/00-candidate-commit.txt
git status --short --branch | tee ~/fedmaq-golden-EVIDENCE/01-tree-state.txt
uv run python scripts/golden_diff.py repeatability \
  2>&1 | tee ~/fedmaq-golden-EVIDENCE/02-repeatability.log
```

This runs every algorithm in `GOLDEN_SET` twice, so budget roughly double a
single capture. Results land under `outputs/golden/step2_repeatability/`, one
`<algorithm>.json` report per algorithm beside the two capture directories.

## Run a transition diagnostic (optional, not a gate)

Only when a ticket asks for old-to-new change classification. Do not remove the
capture outputs between `capture` and `transition`.

```bash
git switch --detach BASELINE_SHA
git rev-parse HEAD | tee ~/fedmaq-golden-EVIDENCE/10-baseline-commit.txt
uv run python scripts/golden_diff.py capture \
  2>&1 | tee ~/fedmaq-golden-EVIDENCE/11-capture.log

git switch --detach CANDIDATE_SHA
git rev-parse HEAD | tee ~/fedmaq-golden-EVIDENCE/12-candidate-commit.txt
uv run python scripts/golden_diff.py transition \
  2>&1 | tee ~/fedmaq-golden-EVIDENCE/13-transition.log
```

`scripts/golden_diff.py` clears each target and the known persistent model state
before each algorithm; inspect it before adding manual cleanup commands.

## Evidence and stopping rules

For the **release gate**, ask the user to return:

- the candidate hash file and the recorded tree state;
- the complete repeatability log;
- the per-algorithm `outputs/golden/step2_repeatability/<algorithm>.json` reports.

Treat the gate as passed only when the recorded hash is the intended candidate,
the tree was clean, and every expected algorithm printed
`PASS — independent captures are bit-exact`. The harness exits non-zero on the
first failure, so a log that ends early is a failure, not a partial pass. A
failure, missing output, wrong commit, or dirty-working-tree warning is a stop
condition. Preserve and inspect the original evidence; do not recapture merely
to erase a failure.

For a **transition diagnostic**, return both hash files, both logs, and the
`outputs/golden/step2_transition/<algorithm>.json` reports. Report it as
recorded, with the classified differential summarized. Never describe it as
passed, and never let it stand in for the release gate — its reports assert
`"pass_eligible": false` precisely so that it cannot.

Routine upstream deprecation or framework future warnings do not by themselves
fail a bit-exact gate, but distinguish them from errors, dirty-tree provenance
warnings, and golden-diff failures.

Generated outputs and logs are evidence, not source changes. Do not commit them
unless this repository explicitly requires a tracked artifact. A formal
verification commit is not normally needed; record the result in the issue or
handoff after local tests and the issue's other acceptance criteria pass.

The verification handoff is complete only when the original evidence is
preserved, the result is classified — passed or failed for the gate, recorded
for a diagnostic — and warnings are separated from actual gate failures.
