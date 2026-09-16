---
name: pre-dispatch-assurance
description: Guide the thesis author through FedMAQ's current-candidate local smoke evidence and JupyterHub GPU golden repeatability before a pipeline-freeze declaration.
---

# Pre-Dispatch Assurance

Use for the final author-operated assurance path before a FedMAQ Stage A or other scientific dispatch. The author runs every smoke and GPU command; the agent prepares, verifies, and records evidence.

## Establish the candidate

Read `docs/freeze/pre_dispatch_smoke_gate.md` and the relevant GitHub issue before proposing a command. Confirm the experiments candidate is clean, pushed, and matches the intended manuscript and literature revisions. A source/configuration change after either capture requires that evidence to be recaptured at the new candidate.

## Select the assurance path

Read the execution Issue and the dated envelope it names before preparing smoke or
golden commands.

- **Sealed-candidate path:** when the exact proposed commit already has complete
  same-candidate smoke and repeatability evidence in its envelope, verify the
  revision vector and preserved evidence. Use that detached commit for the matrix
  handoff; do not repeat the assurance sequence merely because `main` later gained
  non-candidate documentation.
- **New-or-invalidated path:** when the Issue names a new candidate, missing
  evidence, or an in-scope source/configuration change, complete the smoke,
  repeatability, and non-overwriting declaration steps below for that exact commit.

## Local smoke evidence

Guide the author through the five CPU smoke cells and their verifier exactly as specified in `docs/freeze/pre_dispatch_smoke_gate.md`. These run in the author's local checkout, not on JupyterHub. Request the verifier output and preserve the returned evidence; classify a failed or incomplete cell as blocked rather than repairing artifacts or launching a substitute run.

## JupyterHub GPU repeatability

After local smoke evidence is valid for the same candidate, read and apply `../jupyterhub-golden-gate/SKILL.md`. Prepare the `repeatability` release-gate commands for the author on JupyterHub. Do not replace it with a transition diagnostic, and do not launch or poll the GPU process yourself.

## Declare only from complete evidence

For the new-or-invalidated path, treat the pipeline freeze as eligible only when
the local smoke verifier and same-candidate GPU repeatability both pass with
preserved provenance. Then guide the author through the non-overwriting envelope
re-declaration procedure in `docs/freeze/pre_dispatch_smoke_gate.md`. Keep
pipeline, evidence, and results freezes distinct.

## Long verification waits

For an attached local verification process, use one wait of up to ten minutes before sampling status. Keep the original process attached until it exits; do not re-run the gate merely because it is quiet. Report a meaningful state change, final exit code, or a real timeout/blocker.
