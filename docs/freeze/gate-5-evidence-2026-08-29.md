# Gate 5 — Telemetry producer-consumer compatibility attestation

- Envelope: `fedmaq-experiments:docs/freeze/assurance-envelope-2026-08-29.json`
- Gate: 5
- Producer revision: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`
- Consumer reference revision: `fedmaq-analyses@d68a3c44ebd7b9b9301b1f5912190bfdf7ea65fd`
- Bound revision vector: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`,
  `fedmaq-literature@1be87e08d232b8d47ec9c71e80922d7c8235bb5a`,
  `fedmaq-manuscript@5cc82393d639321c240897dbd7eeb728dfc82a02`.
- Producer: assurance orchestrator
- Created: 2026-08-29
- Content digest: recorded in the envelope's `evidence_sha256` entry.

This is a schema/semantic compatibility check only. It does not import
analysis notebooks, figures, manifests, or result interpretations into the
pipeline-readiness evidence.

## Producer schema

The candidate producer declares the stable CSV fields in
`src/fedmaq/core/telemetry.py:47-76` (`COMMON_CSV_FIELDNAMES`) and writes the
same metric mapping to local CSV/JSONL in `:365-420`.

Producer field/semantics inventory:

| Field | Unit/semantics |
| --- | --- |
| `round` | integer round identifier |
| `test/accuracy` | dimensionless accuracy in `[0, 1]` |
| `communication/round_bytes` | aggregate bidirectional bytes for the round |
| `communication/cumulative_bytes` | cumulative aggregate bidirectional bytes |
| `communication/cumulative_mb` | `cumulative_bytes / 1024^2` |
| `communication/round_payload_bytes` | pre-encoding payload bytes; additive per round |
| `communication/round_secondary_bytes` | optional as-published secondary bytes; blank/absent means not reported, not zero |
| `system/*_time_sec` | simulated or measured seconds, as named |

The producer explicitly preserves the optional secondary-axis distinction:
`RoundSnapshot.round_secondary_bytes` is `None` when no client reports the
axis (`src/fedmaq/core/telemetry.py:80-82,252-257`), and CSV missing values are
written as empty strings (`:411-420`).

## Consumer expectation

The analyses loader at
`fedmaq-analyses/src/fedmaq_analysis/loaders.py:17-52`:

- resolves `hydra_output` and optional `telemetry_path` relative to each
  manifest;
- reads the producer CSV without renaming or dropping columns;
- overlays canonical run metadata (`dataset`, `experiment_group`,
  `algorithm_config`, `variant`, `phase`, `alpha`, `seed`, `formulation`, and
  `integrity_status`); and
- returns a long frame consumed by `rounds_to_target`, which requires
  `round` and `test/accuracy` (`fedmaq-analyses/src/fedmaq_analysis/stats.py:12-16`).

The analyses data contract additionally requires the canonical manifest keys,
relative telemetry paths, source hashes, and an allowed `integrity_status` at
`fedmaq-analyses/data/README.md:22-38`.

## Compatibility predicate

For every registered run manifest `m` and resolved CSV `c`, compatibility is
true exactly when all of the following hold:

```text
required_manifest ⊆ keys(m)
hydra_output and telemetry_path, when present, are relative to m
required_csv ⊆ columns(c)
columns(c) contain no duplicates
all present *_bytes values are numeric and non-negative
all present *_time_sec values are numeric and non-negative
present cumulative_mb = cumulative_bytes / 1024^2 within 1e-9
round_secondary_bytes is either absent/blank or numeric and non-negative;
for non-DAdaQuant manifests it must remain absent/blank
```

The executable verifier requires the five fields the current consumer actually
needs to read and compute rounds-to-target (`round`, `test/accuracy`,
`communication/round_bytes`, `communication/cumulative_bytes`, and
`communication/cumulative_mb`). The candidate's payload and optional-secondary
fields are validated when present; their absence in the pre-candidate
provisional records is compatible with the consumer and is not rewritten as a
zero value.

The predicate rejects renamed required fields, wrong byte/MB units, negative
values, duplicate headers, missing run identity, and accidental conversion of
an absent optional secondary value into a reported zero. Extra producer fields
are accepted because the consumer reads them without dropping or renaming
columns.

## Attestation result

The predicate was executed against all 177 registered manifests and their
resolved telemetry CSVs in the consumer checkout:

```text
checked_manifests=177
secondary_column_present=0
errors=0
```

The same predicate was run against one disposable current record with three
mutations:

```text
positive_current_record=yes
negative_renamed_required_field=rejected
negative_wrong_mb_unit=rejected
negative_absent_secondary_to_zero=rejected
negative_absolute_telemetry_path=rejected
```

The positive check and all three negative cases were run from a temporary
copy; repository and analysis files were not modified.

The zero secondary-column count is compatible with the predicate: these are
existing provisional records, while the candidate producer's optional
secondary field remains available for future DAdaQuant records. No result
claim is made from this check.

## Retained executable verifier

`docs/freeze/verify_gate_5.py` is the retained, self-testing command artifact.
It is deliberately outside the source-certificate scope: placing a new helper
in `scripts/` or `tests/` would alter the candidate's frozen behavior surface
and invalidate Gates 1–4. It reads only the consumer manifests and telemetry
CSVs, requires the audited consumer revision, and creates its negative cases
in a temporary directory.

```text
uv run python docs/freeze/verify_gate_5.py --analyses-root ..\\fedmaq-analyses \
  --require-revision d68a3c44ebd7b9b9301b1f5912190bfdf7ea65fd --self-test

checked_manifests=177
secondary_column_present=0
errors=0
positive_current_record=yes
negative_renamed_required_field=rejected
negative_wrong_mb_unit=rejected
negative_absent_secondary_to_zero=rejected
```

## Disposition and invalidation owner

The named invalidation owner is the assurance orchestrator: any producer field
rename, unit change, null/zero semantic change, consumer expectation change, or
manifest-path/identity contract change invalidates Gate 5 and any dependent
evidence. A change to result-analysis logic alone is outside this narrow
attestation and belongs to the later evidence/results workflow.

Gate 5 is **PASS**. The predicate is persisted as a reproducible, retained
command artifact; the positive and negative checks above were rerun at the
audited consumer revision. The attestation remains narrow and does not import
result analysis into pipeline-readiness evidence.
