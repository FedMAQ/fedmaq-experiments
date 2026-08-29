# Gate 7 — Blocker record, freeze availability, and invalidation demonstration

- Envelope: `fedmaq-experiments:docs/freeze/assurance-envelope-2026-08-29.json`
- Gate: 7
- Candidate: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`
- Bound revision vector: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`,
  `fedmaq-literature@1be87e08d232b8d47ec9c71e80922d7c8235bb5a`,
  `fedmaq-manuscript@5cc82393d639321c240897dbd7eeb728dfc82a02`
- Producer: assurance orchestrator
- Created: 2026-08-29
- Content digest: recorded in the envelope's `evidence_sha256` entry.

## Current blockers and availability

- Gates 0, 1, 2, and 4: PASS within the declared assurance boundary.
- Gates 3 and 5: BLOCKED pending verifiable Gate 3 model provenance and a
  reproducibly persisted Gate 5 verifier.
- Gate 6: BLOCKED. The exact-commit JupyterHub golden capture/compare remains
  thesis-author-executed and has not been supplied.
- Gate 7: BLOCKED. This record demonstrates the invalidation matrix, but the
  pipeline cannot be declared frozen while Gate 6 evidence is absent, and only
  the thesis author may declare a freeze.
- Pipeline freeze: **not available**.
- Evidence freeze: **not available**; campaign artifacts and provenance are
  not complete.
- Results freeze: **not available**; claim-support review is not complete.

The availability statements are against the current envelope boundary,
including the declared #42 theoretical residual and the exclusion of
`fedmaq-analyses` except for Gate 5's narrow compatibility attestation.

## Invalidation-matrix demonstration

The following dry run made disposable copies, appended a marker only to each
copy, verified that its content hash changed, and deleted the temporary tree.
No repository file was modified. The listed gates are the exact gates made
stale by each representative delta under the #74 matrix and the current
envelope dependencies.

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
| scope boundary | `fedmaq-experiments/docs/freeze/assurance-envelope-2026-08-29.json` | 1, 2, 3, 4, 5, 6, 7 | the boundary itself is invalidating and every gate must be re-evaluated against it |
| telemetry contract | `fedmaq-experiments/src/fedmaq/core/telemetry.py` | 5 | schema, units, or null/zero semantics stale the producer-consumer attestation |

The dry-run command reported:

```text
disposable_copies=10
classification_assertions=10
repository_files_modified=no
```

## Disposition

Gate 7 remains **BLOCKED** until Gate 6's user-run evidence is ingested and
the thesis author records the desired freeze declaration. This artifact
demonstrates availability and invalidation behavior; it does not itself
declare any freeze.
