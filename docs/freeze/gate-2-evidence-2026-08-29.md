# Gate 2 — Repair and bounded re-audit: evidence

Envelope: `docs/freeze/assurance-envelope-2026-08-29.json` (#75). Prior gate:
gate 1, PASS, `docs/freeze/gate-1-evidence-2026-08-29.md`, candidate pin at
the time `6783a30`.

## 1. Disposition of every gate-1 finding

Gate 1's evidence (§1 and §4) names exactly two findings. Both already carry
a disposition recorded inline at gate 1; this gate restates and confirms each
rather than leaving the restatement implicit:

- **Simulation-lifecycle `finally`-block semantic change** (`5617bf0`,
  `d01650b`, architecture pass 3). Gate 1 labeled this "**Finding, not a
  defect**": `SimulationRun.execute()` now preserves the primary exception
  instead of letting a secondary finalization error mask it, on a path the
  golden capture/compare cannot exercise. Gate-1 disposition: "no repair
  needed — intentional, tested, and correctly favors surfacing the primary
  failure." Confirmed at gate 2: no repair action taken; disposition stands
  unchanged.
- **FedDistill download-leg 9.4% over-charge** (`d031cc8`). A substantive
  defect, but one found and fixed *within* the delta gate 1 audited, not a
  residual defect gate 2 inherits. Gate 1 confirmed the fix present and
  tested at the candidate pin (`strategy_hooks/feddistill.py::
  download_size_bytes` delegates the weight leg and measures the logit leg
  through the shared `measure_bytes` seam). Gate-2 disposition: already
  repaired prior to and confirmed by gate 1; no further repair action
  required.

Gate 1 §4 states no other unresolved static mismatch was found inside the
scope boundary, and no escalation to the author was triggered by any check
in that gate. There is no third finding to dispose of.

## 2. Confirmation that repairs altered no unapproved method contract

Zero repairs were made at gate 2, because gate 1 left zero open findings
requiring one (§1 above). This mandatory-evidence item is satisfied
vacuously — a repair that did not occur cannot have altered a method
contract — and that is recorded here explicitly rather than inferred from
the absence of a diff. No method change, source-fidelity trade-off,
frozen-configuration change, or result interpretation occurred, so nothing
in this gate meets #74's escalation criteria.

## 3. Final candidate re-pin

Per #74 §"The candidate is re-pinned, not pinned once": on every re-pin the
responsible verifier classifies the delta against
`docs/freeze/source_manifest.json`'s `scope.include`
(`.github/workflows/**/*.yaml`, `.github/workflows/**/*.yml`,
`.python-version`, `conf/**/*.yaml`, `conf/*.yaml`, `justfile`,
`pyproject.toml`, `scripts/**/*.py`, `src/**/*.py`, `tests/**/*.py`,
`uv.lock`).

- **Delta**: `6783a30..d804b7f` (the previous candidate pin to the tip of
  `main` after gate 1 landed).
- **Commits**:
  - `667e5f5853547a139283057e59c49b8bad04f2f1` (`667e5f5`) — paths:
    `docs/freeze/assurance-envelope-2026-08-29.json` (created, gate 0).
    `inside_scope_include`: false.
  - `d804b7f2223fa92a8d2bcde803bec1501454faf5` (`d804b7f`) — paths:
    `docs/freeze/assurance-envelope-2026-08-29.json`,
    `docs/freeze/gate-1-evidence-2026-08-29.md` (gate 1 PASS disposition and
    evidence). `inside_scope_include`: false.
- **Clause**: #74 section 2 mechanical-repair clause.
- **Attested**: true.
- **Attestation**: No path in the `6783a30..d804b7f` delta falls inside
  `scope.include`. Both commits are envelope/gate-evidence artifacts only
  (gate 0's envelope creation, gate 1's PASS disposition and evidence file)
  — no protocol, configuration, source code, or claims-bearing content
  changed. `behavioral_content_commit` is therefore unchanged from the prior
  pin (`5617bf0`, architecture pass 3 — the last commit that changed actual
  behavior). Recorded, not inferred from the diff being small.
- **Consequence for already-passed gates**: because the delta touches no
  `scope.include` path and no claims-bearing content, gates 0 and 1 remain
  valid at the new pin under the mechanical-repair clause — they are not
  invalidated and are not re-run. This is the intended effect of the clause,
  not an exception to it: the clause exists precisely so that the gates'
  own recording of their own results does not retroactively invalidate
  themselves.
- **Not licensed by this attestation**: reuse of any evidence from gates 3
  onward as if already produced. Every gate from here still runs fresh
  against the new pin; only gates 0 and 1 are carried forward, and only
  because this attestation says so explicitly.

**This is gate 2's final re-pin.** Per #74 §"Candidate-freeze window", the
window now opens: from this point until gate 7 completes, any commit
touching `scope.include`, or touching claims-bearing content in
fedmaq-literature or fedmaq-manuscript, invalidates every gate already
passed (gates 0 and 1) and the mechanical-repair clause becomes unavailable
for the remainder of the pipeline-freeze process. A commit whose entire
content is envelope/gate-evidence artifacts remains non-invalidating.

## 4. Internal-consistency pass following the re-pin

The re-pin in §3 moves `candidate.revision`. Three other fields name a
revision or a verification event and were checked for drift rather than left
to accumulate it silently:

- **`revision_vector.fedmaq-experiments.revision`** updated to `d804b7f` in
  step with `candidate.revision`, with a `binding_note` added recording that
  this field is a binding that moves on re-pin, not an event log — in
  contrast to `runtime_provenance.executed_revision` and
  `golden_harness.harness_revision`, which record gate-0's execution and
  capture events and are correctly left at `6783a30`.
- **`freeze_certificate.verified_by`** amended to name both the original
  user-run `just freeze` at `6783a30` and the gate-2 `just check` run (this
  gate, 2026-08-29) that confirms `docs/freeze/source_manifest.json` remains
  current at the `d804b7f` tree — licensed because §3's attestation already
  establishes the delta touches no `scope.include` path.
- **Gate 4's `notes`** amended in place to flag that its `just check at
  6783a30` text is authoring-time only and that `candidate.revision` is
  authoritative where the two disagree, so a future gate-4 run does not
  silently execute against a stale pin.

No other `6783a30` reference remains uncovered: `revision_vector`'s
`fedmaq-literature` and `fedmaq-manuscript` entries, and gate 1's `notes`
(delta `530f8ae..6783a30`), are historical statements about gate 1's own
audited range and are correctly left unchanged.

## Disposition

Gate 2: **PASS**, evidence per §1-4 above. Candidate re-pinned to `d804b7f`.
