# Gate 7 — Blocker record, freeze availability, and invalidation demonstration

- Envelope: `fedmaq-experiments:docs/freeze/assurance-envelope-2026-08-29.json`
- Gate: 7
- Candidate: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`
- Bound revision vector: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`,
  `fedmaq-literature@1be87e08d232b8d47ec9c71e80922d7c8235bb5a`,
  `fedmaq-manuscript@5cc82393d639321c240897dbd7eeb728dfc82a02`
- Producer: assurance orchestrator
- Created: 2026-08-30 (Gate 6 evidence ingestion)
- Content digest: recorded in the envelope's `evidence_sha256` entry.

## Current blockers and availability

- Gates 0, 1, 2, and 4: PASS within the declared assurance boundary.
- Gates 3 and 5: PASS. Gate 3 uses the thesis-author-amended Luna High
  independent-review standard; Gate 5 has a retained, self-testing verifier.
- Gate 6: **PASS**. The thesis author supplied the complete external evidence
  package listed below. The baseline and candidate records match the envelope;
  all nine expected algorithms compare bit-exactly after excluding only the
  predeclared wall-clock columns, and the compare ends with
  `All golden diffs passed.`
- Gate 7: **PASS as an assurance record**. The invalidation matrix is
  demonstrated and the remaining blockers and freeze availability are
  recorded below. This does not itself declare a freeze.
- Pipeline freeze: **available for author declaration**, limited to #22 step 1.
- Evidence freeze: **not available**; campaign artifacts and provenance are
  not complete.
- Results freeze: **not available**; claim-support review is not complete.

The availability statements are against the current envelope boundary,
including the declared #42 theoretical residual and the exclusion of
`fedmaq-analyses` except for Gate 5's narrow compatibility attestation.

## Gate 6 evidence package

The files remain outside the checkout at the paths supplied by the thesis
author. Their SHA-256 digests are recorded here so the package is identifiable
even though it is not copied into the repository.

| Artifact | Producer / created | Exact reference | SHA-256 |
| --- | --- | --- | --- |
| baseline commit record | thesis author / 2026-08-30 | `C:\Users\Quirora\Downloads\00-baseline-commit.txt` — `2f3a3c1b102c745efad97b38f305544ffe1aba62` | `bd02b08a76038950877d59124fed171dc5052c371feb3c2a977ae1467f09e0f0` |
| capture log | thesis author / 2026-08-30 | `C:\Users\Quirora\Downloads\01-capture.log` | `7330f62286543429aa4412771ed6474a4d8f00afe54accb034a6c9f6ace2535b` |
| candidate commit record | thesis author / 2026-08-30 | `C:\Users\Quirora\Downloads\02-candidate-commit.txt` — `d804b7f2223fa92a8d2bcde803bec1501454faf5` | `308a20b85ac239e87fd75e95363f677704e1f69dd6f706a98abc1186ad801b46` |
| compare log | thesis author / 2026-08-30 | `C:\Users\Quirora\Downloads\03-compare.log` | `65d938938bab87100d0d5f7115b9092319d08ee6e97b40f63d9ca9d72c859ad5` |

The capture and compare were user-run at the exact baseline/candidate pair
`2f3a3c1b102c745efad97b38f305544ffe1aba62` →
`d804b7f2223fa92a8d2bcde803bec1501454faf5`. The expected algorithm set is
`fedavg`, `fedprox`, `fedpaq`, `fedavg_kd`, `dadaquant`, `fedmaq`, `fedkd`,
`feddistill`, and `cfd`; only wall-clock columns are excluded. Flower/Ray and
missing aggregation-hook messages are retained non-fatal warnings, not
compare failures or provenance failures.

## Invalidation-matrix demonstration

The retained verifier `docs/freeze/verify_gate_7.py` makes disposable copies,
appends a marker only to each copy, verifies that its content hash changed, and
deletes the temporary tree. No repository file is modified. The listed gates
are the exact gates made stale by each representative delta under the #74
matrix and the current envelope dependencies.

| Representative delta | Disposable source | Exact invalidated gates | Reason |
| --- | --- | --- | --- |
| behavior | `fedmaq-experiments/src/fedmaq/core/telemetry.py` | 1, 3, 4, 6 | behavior can change audit, independent review, local verification, and golden evidence |
| configuration | `fedmaq-experiments/conf/algorithm/fedmaq.yaml` | 1, 3, 4, 6 | configuration affects audit, review, local verification, and golden identity |
| protocol claims | `fedmaq-experiments/docs/adr/0021-power-mean-formulation-family.md` | 1, 3 | protocol-claim changes stale audit and independent review |
| manuscript wording | `fedmaq-manuscript/chapter_4.tex` | 1, 3 | claim-bearing wording changes stale audit and independent review |
| harness rules | `fedmaq-experiments/scripts/golden_diff.py` | 1, 3, 4, 6 | harness behavior changes local/reviewed reproducibility and golden evidence |
| runtime provenance | `fedmaq-experiments/docs/freeze/gate-4-evidence-2026-08-29.md` | 1, 3, 4, 6 | execution identity changes stale audit, bound review, local, and golden evidence |
| Gate 3 evidence reference | `fedmaq-experiments/docs/freeze/gate-3-review-2026-08-29.md` | 3 | the affected gate's reviewer output is no longer current |
| Gate 4 evidence reference | `fedmaq-experiments/docs/freeze/gate-4-evidence-2026-08-29.md` | 4 | the affected gate's verification record is no longer current |
| Gate 5 verifier artifact | `fedmaq-experiments/docs/freeze/verify_gate_5.py` | 5 | the retained predicate implementation is the compatibility evidence |
| scope boundary | `fedmaq-experiments/docs/freeze/assurance-envelope-2026-08-29.json` | 1, 2, 3, 4, 5, 6, 7 | the boundary itself is invalidating and every gate must be re-evaluated against it |
| telemetry contract | `fedmaq-experiments/src/fedmaq/core/telemetry.py` | 5 | schema, units, or null/zero semantics stale the producer-consumer attestation |

The retained command was executed as follows:

```text
uv run python docs/freeze/verify_gate_7.py
behavior: content_hash_changed=yes invalidates=1,3,4,6
configuration: content_hash_changed=yes invalidates=1,3,4,6
protocol_claims: content_hash_changed=yes invalidates=1,3
manuscript_wording: content_hash_changed=yes invalidates=1,3
harness_rules: content_hash_changed=yes invalidates=1,3,4,6
runtime_provenance: content_hash_changed=yes invalidates=1,3,4,6
gate_3_evidence_reference: content_hash_changed=yes invalidates=3
gate_4_evidence_reference: content_hash_changed=yes invalidates=4
gate_5_verifier_artifact: content_hash_changed=yes invalidates=5
scope_boundary: content_hash_changed=yes invalidates=1,2,3,4,5,6,7
telemetry_contract: content_hash_changed=yes invalidates=5
disposable_copies=11
classification_assertions=11
repository_files_modified=no
```

## Disposition

All orchestrator-resolvable blockers are closed. Issue #45 remains a hard
precondition for the selected-formulation/second freeze and #22 step 6's full
downstream campaign; it does not block Gate 6 or this limited step-1 pipeline
freeze availability. The thesis author must still explicitly declare any
freeze. This artifact records availability and invalidation behavior; it does
not itself declare any freeze.
