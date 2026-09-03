# Final pre-freeze audit — implementation-completion gate (N31)

Independent audit of the implemented remediation, run as the completion gate that
[N31](2026-09-02-pipeline-audit-confirmation.md) makes mandatory before pipeline-freeze
consideration. Scope is the `fedmaq-experiments` repository only; the manuscript,
literature wiki, analyses, and presentations repositories were not read or modified.

- **Baseline entering the audit:** `main` @ `5f85868`, clean tree. `5f85868` is an
  envelope-only commit over source candidate `0c6028e`.
- **Owning issue:** [#92](https://github.com/FedMAQ/fedmaq-experiments/issues/92)
  (parent of #96, per that issue's own `## Parent` header). #86 owns the implementation
  contract.

## Verdict

**Changes required.**

Two freeze-blocking defects were found and fixed in this pass. Two acceptance criteria of
the closed issue #96 remain unmet; they are not fixed here because each carries blast
radius beyond a pre-freeze polish, and they are routed below.

Three findings were raised as requiring author disposition — **S1**, **S2**, and **S3**.
S3 is the generalization of C1 to the pre-freeze confirmation matrix; unlike C1 it must
**not** be fixed by the same mechanism, for the reason given in its section. A second pass
landed the mechanism S3 actually needs — see the Addendum at the end of this report.

**A third pass has since closed S3 entirely: it was never open.** ADR-0008 had already
recorded the Stage 3 confirmation as skipped as degenerate against an empty surviving set,
and this report reached that conclusion's opposite twice by bad evidence — inferring "no
exploration" from an empty *local* `outputs/`, which `docs/agents/execution-model.md`
forbids, and treating the historical `expected_runs.json` as the replacement campaign's
gate-enforced contract, which `matrix_contracts` is. Both errors are recorded in full in the
S3 section rather than edited away. **S1 and S2 remain author-owned**; S1 is dispositioned as
riding to #86.

This report certifies neither that the pipeline is ready, nor that #101 is complete, nor
that manuscript reconciliation may begin. The source changes landed here **invalidate the
#100 assurance candidate**: a fresh golden repeatability gate and a new assurance envelope
must be run by the thesis author against the new candidate before freeze is considered.
That candidate is **`a597edb`** (`7c1fdb7` when this report was first written, `4c4337b`
after the second pass, `a597edb` after the third — see the S3 correction), statically green
but not yet gated — the repeatability run is what converts it from a candidate into an
assured one, and it has not happened.

## Findings

### C1 — Stage 1b could be dispatched at a `p` no selection chose (critical, FIXED)

`conf/matrix/power_mean_omega.yaml` carried the selected power-mean degree only as a
comment (`# PLACEHOLDER: algorithm.p=<selected_p>`), with no `algorithm.p` override on
either run. Four layers failed to catch the consequence:

1. `conf/algorithm/_power_mean_base.yaml` defaults `p: 0`, so Hydra composition silently
   resolved both Stage-1b runs to `p=0` — itself a legitimate Stage-1a candidate, and so a
   plausible value rather than a crash.
2. `identity_key()` in [`src/fedmaq/core/run_identity.py`](../../src/fedmaq/core/run_identity.py)
   takes `(dataset, experiment_group, algorithm_config, variant, alpha, formulation, seed)`
   — no `p`, no `omega`. The Stage-1b variants are `omega0.25` / `omega0.75` and
   `formulation` is `None` for `power_mean`, so a `p=0` cell and a `p=-1` cell produce
   **byte-identical identity keys and output directories**.
3. The dispatch path performed no unresolved-selection check. `plan_matrix` validated the
   matrix against the protocol contract but nothing looked at whether a run's values had
   been chosen. (Closed by the second pass — see the Addendum.)
4. `tests/test_stage_contracts.py::test_ledgers_have_required_counts_and_are_disjoint`
   asserts Stage 1b has **12 cells** — which wrong-`p` evidence satisfies exactly.

The generated ledgers cannot detect this, for the reason
[Lens 3](2026-09-02-reporting-evidence-traceability-audit.md) already gave: their
scientific identities are derived from the matrices they check, so an edit and a
regeneration move both sides together. Confirmed empirically — after adding the `p`
override, all five `--check` generators still reported *current*, because none of them
reads `algorithm.p`.

**Partial mitigation that does exist.** `select_power_mean_omega_iso_byte()` in
`scripts/analysis.py` routes runs whose `experiment_group` equals the Stage-1b
group into a guarded branch that calls `p_matches(r, p_norm)` and raises, naming
`r.job_dir` and both `p` values. An earlier reading of the function's `else` branch as the
Stage-1b path was wrong; the guard is real and fail-closed. So the failure mode was
**diagnosable waste, not silent corruption** — but detection came only at analysis, after
all 12 GPU cells had been spent, and recovery was booby-trapped: because the identity key
omits `p`, the wrong-`p` run occupies the correct cell's canonical directory, and the
documented recovery path (`run_matrix.py --skip_completed`, per
`src/fedmaq/core/client_manager.py:136`) would skip it as already complete.

**Fix landed.** Both Stage-1b runs now carry `- algorithm.p=???`. Hydra's override grammar
accepts `???` and composes it to OmegaConf `MISSING`, so `_QuantParams.from_cfg()` raises
`MissingMandatoryValue: Missing mandatory value: algorithm.p` at quantization-policy
construction — before the first training round, on cell 1. Verified end-to-end for both
runs.

The mechanism is worth stating precisely, because the guarantee is conditional — and the
first version of this paragraph understated how conditional. Measured on a composed
`algorithm.p=???`:

| read | result |
| --- | --- |
| `cfg.algorithm["p"]` | raises `MissingMandatoryValue` |
| `cfg.algorithm.get("p", 0.0)` | returns `0.0` — **fail-open** |
| `cfg.algorithm.get("p")` | returns `None` — **fail-open** |
| `to_container(resolve=True)["p"]` → `_parse_power_mean_degree()` | raises `ValueError` |

There are four reads, not two, and two of them fail open. The raise depends on the
`alg_cfg["p"]` subscript at `src/fedmaq/core/quantization_planner.py:103`, which is taken
only when the formulation requires `p`; `src/fedmaq/core/manifest.py:147` records the run
with `.get`, so an unresolved `p` would leave no trace of the placeholder in that run's
provenance. This is why the second pass moved the check to the planner, which fails closed
regardless of consumer — see the Addendum.

This deliberately does **not** follow #96's instruction to copy the freeze-confirm
convention. `conf/matrix/pass3_freeze_confirm.yaml:83` writes real working values under a
comment saying to replace them; plausible-default-plus-comment is precisely the failure
mode being fixed.

### C2 — The golden-gate skill instructed an operation that can never pass (critical, FIXED)

[N27](2026-09-02-pipeline-audit-confirmation.md) required four surfaces to be revised: the
skill, the harness, the evidence schema, and cleanup behaviour. The harness and its tests
were rebuilt correctly — `scripts/golden_diff.py` separates `transition_diagnostic()` from
`repeatability_report()`, and `tests/test_stage_contracts.py` asserts the transition is
never pass-eligible. **The skill was never revised**, and had drifted into direct conflict
with the harness it drives:

- It defined the gate as capture at `BASELINE`, compare at `CANDIDATE` — the operation
  N27 says "can only be a transition diagnostic".
- `compare` is now a backward-compatible alias for `transition`, which prints
  `transition diagnostic recorded`, reports `"pass": false` / `"pass_eligible": false`, and
  never exits non-zero.
- The skill's stated pass criterion was the log ending in `All golden diffs passed.` That
  string **no longer exists in the harness**. It survives only in the skill itself and in
  `docs/freeze/gate-7-evidence-2026-08-29.md`, a historical evidence record.
- `repeatability` — the only operation that emits PASS, and the only one that exits
  non-zero on failure — was **not mentioned anywhere in the skill**.

An operator following the skill exactly would run the diagnostic, never see the success
string, and declare the golden gate failed; or, reading `transition diagnostic recorded`
as success, would certify a freeze on a comparison that asserts its own ineligibility.
Because the golden gate is the release gate the author must run before freeze, this was
freeze-blocking.

**Fix landed.** `.agents/skills/jupyterhub-golden-gate/SKILL.md` rewritten to lead with the
two operations and their differing standing, to give `repeatability` (one commit, both
captures from the same clean tree) as the release gate, to mark the transition diagnostic
optional and non-certifying, and to key the pass criterion to the harness's actual string
`PASS — independent captures are bit-exact`. `.agents/` is outside `source_manifest.json`,
so this fix carries no source blast radius.

### S1 — #96 closed with two acceptance criteria unmet (significant, ROUTED)

Issue #96 is closed. Walking its acceptance list against the code, 18 of 20 bullets are met
— including round-0 acceptance, the all-NaN optional-column skip, JSONL as a required
checked artifact, `derive_seed` hash-combination, parsed-number variant matching, the
non-zero closure exit behind `--allow-incomplete`, per-seed iso-byte budgeting, the Gate-2
precondition, and the FedPAQ-pipeline / memory-sensitivity / variable-vs-uniform readouts.
Two are not:

| # | Criterion | State |
| --- | --- | --- |
| 3 (half) | "the identity key incorporates `p` and `omega` so two Stage-1b cells cannot collide on disk" | **Unmet.** `identity_key()` takes neither. The closing receipt claims "Manifests expose p/omega/formulation" — true, at `src/fedmaq/core/manifest.py:147-148` — and silently drops the identity-key half of the same bullet. |
| 4 | "The Stage-1b matrix carries explicit `p` overrides marked as pre-dispatch placeholders" | **Met.** Fixed as C1 and enforced by the second pass; claimed by no closing receipt. |
| 11 / #102-15 | "Selection domains … registered in the replacement protocol, with a startup cross-check against the matrices"; "Startup validation compares the dispatch expansion against that independently registered authority" | **Half met.** Correction to this report's first version: `selection_domains` is *not* an unread passthrough. `preregistration_contract()` returns it, `register_protocol()` hashes the whole contract into `preregistration_sha256` (`protocol.py:161`), and `is_promotable_manifest()` recomputes and compares it (`protocol.py:209-211`) — so editing it invalidates the promotability of every existing manifest. What is genuinely absent is the **cross-check against the matrices**: nothing compares the registered seed list, `q` ladder, `p` support or `omega` support to what the matrices actually expand to. |

The unmet half is one theme: **no authority independent of the matrices constrains the
dispatched design.** (Narrower than this report first stated — `matrix_contracts` *is* such
an authority for stage/split/ledger/rounds/cell-count/content-hash, and the second pass
put it under test; what remains unconstrained is the selection domains.) That is the
disposition Lens 3 routed to #96 and marked as the
condition under which "the existence of a valid hash must not be described as proof that
the dispatched scientific design matched the preregistration." It has not landed, so that
caveat still holds and must not be stated otherwise in the manuscript.

Not fixed here deliberately. Adding `p`/`omega` to `identity_key()` changes every canonical
output directory, every path in `source_manifest.json`, and the golden capture roots; the
collision it prevents is already diagnosable via the existing raise. The independent
startup authority is a designed workstream, not a pre-freeze polish. Both belong to #86's
implementation contract.

### S2 — The freeze envelope declares a content hash it does not carry (significant, ROUTED)

`docs/freeze/assurance-envelope-2026-09-03.json` states its purpose as an "Immutable,
content-hashed record" and declares `"content_hash": {"algorithm": "sha256", "value": null}`.
The field is read by nothing — no consumer in `src/`, `scripts/`, or `tests/`.

Not fixed here, on purpose, and the ordering is the reason rather than the scope. A verifier
would live under `scripts/` and its test under `tests/`, both inside `source_manifest.json`'s
scope — so writing it would re-invalidate the source candidate and re-arm a release gate that
has not yet been run, strictly enlarging what has to be redone. The fix is also naturally
*downstream*: the body worth hashing includes the repeatability outcome, which does not exist
yet. So this is reduced to a mechanical patch to apply after the gate, not a design decision
left open.

**Ready-to-apply resolution.** A new envelope is required for the new candidate regardless.
When writing it, take option (a) or (b) and do not leave the field null a third time.

(a) *Carry a real hash.* Adopt the working in-repo pattern at
`src/fedmaq/core/protocol.py:170`, `{**envelope_body, "sha256": _sha256(envelope_body)}`, and
state the input domain in the document itself so it is checkable rather than decorative:

```json
"content_hash": {
  "algorithm": "sha256",
  "input_domain": "canonical JSON of this document with the content_hash key removed, sorted keys, separators (',', ':'), UTF-8",
  "value": "<sha256>"
}
```

The value is then reproducible in one line, which is what makes it a provenance record rather
than an assertion:

```bash
uv run python -c "import hashlib,json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text('utf-8')); d.pop('content_hash'); print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())" docs/freeze/assurance-envelope-<date>.json
```

(b) *Drop the claim.* Remove the `content_hash` object and strike "content-hashed" from
`purpose`. The envelope still binds the candidate through `revision_vector` and `candidate`,
which are the fields anything actually relies on.

Option (a) is preferable — the envelope's stated job is to be content-hashed — but (b) is
strictly better than the current state, which describes a guarantee it does not provide.

### M1 — Root config cell accounting is stale (minor, ROUTED to #101)

`conf/config.yaml` still describes "The 183 runs of §4.5" and a 105-run primary grid, and
lists `formulation_study`, `pass2_explore`, `pass2_factorial` and `pass3_freeze_confirm`
as retired. That list is `config.yaml`'s claim, not this audit's: S3 shows the
gate-enforced contract still expects `pass3_freeze_confirm`'s cells, which is part of why
its status needs author disposition. The live contract is 415 = 145 + 84 + 12 + 174.
Left unedited
deliberately: #96 states that every number is written once after the code settles, and
#101 owns that reconciliation. Editing it now would write a number that pass will rewrite.

### M2 — Unreachable defensive branch in the Stage-1b selector (minor, informational)

The `else` branch of `select_power_mean_omega_iso_byte()` admits `omega0.25` / `omega0.75` /
`omega0.5` / `p0` variants without a `p` check. It is dead-defensive: a census of
`conf/matrix/` shows those variants occur only in `power_mean_omega.yaml` (Stage-1b group)
and `power_mean_design.yaml` (Stage-1a group), both of which are handled by the two guarded
branches above it. No action needed; recorded so it is not re-litigated.

### M3 — Telemetry has no download-byte columns (minor, informational)

`COMMON_CSV_FIELDNAMES` in `src/fedmaq/core/telemetry.py` carries `round_upload_bytes` and
`cumulative_upload_bytes` but no `round_download_bytes` column, though `RoundSnapshot`
carries the field, and no cumulative-download counterpart. Download is recoverable as
`round_bytes − round_upload_bytes`. Directional accounting is therefore derivable but not
first-class. Relevant to L1-05's warning against conflating the upload and bidirectional
axes; no correctness defect.

### M4 — Payload archive is unversioned, but is not an evidence surface (minor, informational)

`src/fedmaq/core/payload_archive.py` separates uploads from downloads per round, which
satisfies the directional half of the "separate archive artifacts" assertion, but carries no
schema-version constant — unlike `control_messages.py`, which stamps `CONTROL_SCHEMA_VERSION`
— and persists via `pickle`. It is gated on `telemetry.log_payloads`, which defaults to
`False` and **appears nowhere in `conf/`**, so it is disabled for every campaign cell. It is
a debugging capture, not an evidence artifact, and is not freeze-relevant. Recorded so the
assertion is not re-opened.

### S3 — The second placeholder is live, and C1's fix must NOT be copied to it (significant, CLOSED third pass — see the correction at the end of this section)

A sweep of `conf/` for placeholder patterns — the generalization C1 motivates — returns
exactly one other site: the `fedmaq-surviving-set` arm of
`conf/matrix/pass3_freeze_confirm.yaml`, which carried three real, working booleans under a
`# PLACEHOLDER — set from exploration_margin.json before dispatch.` comment. This is
structurally the C1 hazard: `identity_key()` encodes the *variant string*
(`surviving-set`), not the booleans, so a forgotten edit would dispatch cells that are
indistinguishable on disk from a genuine surviving set.

**It has not yet been dispatched, so the hazard is live rather than historical.** The
matrix's own `BEFORE RUNNING:` block names
`scripts/analysis_output/exploration_margin.json` as the source of the real values. That
file does not exist; neither does `scripts/analysis_output/`. `outputs/` contains only
`ci`, `golden`, and `smoke` — no campaign results of any kind, and nothing under any
`pass3_freeze_confirm` identity key. Exploration has not run.

Whether this matrix is *scheduled* to run is a design question this audit cannot settle
from the repository, and the two available signals disagree. `conf/config.yaml:42` says the
exploration chain "completes and freezes first," but that file's accounting is stale (M1),
so it is not evidence. Against that, `scripts/dump_expected_runs.py:69` actively includes
`pass3_freeze_confirm` and `docs/freeze/expected_runs.json:571-578` enumerates its eight
identity keys — and `expected_runs.json` is a generated artifact under `--check`
enforcement that passed `just check` in this same session. The gate-enforced contract
currently expects these cells. That is the stronger signal, and it is why this is
significant rather than minor.

**Author disposition required**, one of:
- edit the three overrides from `exploration_margin.json` before dispatch, as the file's own
  header instructs; or
- confirm the matrix is superseded by the re-cut design and remove it from
  `dump_expected_runs.py`, regenerating `expected_runs.json`.

It is **not** fixed here, for the reason below, which is the part most likely to be got
wrong by a later reader:

**The C1 sentinel would make this fail-*open*, which is strictly worse.** The consumer is
`alg_cfg.get("soft_voting", False)` at `src/fedmaq/core/strategy_hooks/fedmaq.py:112` — a
`.get()`, not the `[...]` read that makes C1's sentinel bite. On any path where the config
has been flattened to a plain dict, `algorithm.soft_voting=???` yields the string `'???'`,
which is **truthy**, and the mechanism turns on silently. The honest default the file
carries today, under a loud `BEFORE RUNNING:` block that states the values "are NOT a
prediction of what will survive," is the safer of the two.

**Do not apply `???` to boolean overrides anywhere.** The sentinel is safe only where the
consumer subscripts a `DictConfig` or parses the value, as C1's does.

**Update (second pass).** The mechanism this finding needed — one that is neither the
fail-open sentinel nor a comment — has since landed: the arm declares
`pending_selection: [soft_voting, ema_student, grad_norm_ema]`, and `plan_matrix` refuses
the matrix while that list is non-empty. The hazard is now inert; the **scientific
disposition below is still open and still author-owned**. Two things found while landing it
bear on that disposition:

- The header's claim that the placeholder values "are the current fedmaq.yaml defaults" was
  **inverted**. `conf/algorithm/fedmaq.yaml:74,79,83` freeze all three `false` under
  "Decision 79/80 … empty surviving set … Do not flip these". The arm's `=true` overrides
  therefore contradict the frozen decision rather than restating a default. Corrected in
  the file.
- If Decision 79/80's empty surviving set stands, `fedmaq-surviving-set` and
  `fedmaq-unrefined` describe the same configuration, and the matrix's two-arm comparison
  has nothing to compare. That is evidence for the "superseded" branch of the disposition
  below, but it is a scientific judgement this audit does not make.

**Correction (third pass, 2026-09-03). S3's scientific disposition was not open. It had
already been made, and this audit reached for the wrong evidence twice.** The two errors are
recorded rather than edited away, because both are reasoning failures a later reader could
repeat.

*First error — "Exploration has not run" is false, and was inferred illegitimately.* It rests
on `outputs/` containing only `ci`, `golden`, and `smoke`. That inference is explicitly
forbidden by `docs/agents/execution-model.md`: results live on the datacenter allocation, and
**an empty local `outputs/` proves nothing about which stages have run**. ADR-0008 is Accepted
and records exploration as **executed 2026-08-02**, with all 26 `pass2_factorial` runs clean
and an empty surviving set. The missing `scripts/analysis_output/exploration_margin.json` is
likewise a local-artifact absence, not evidence of a missing run.

*Second error — the wrong ledger was called "the gate-enforced contract."*
`docs/freeze/expected_runs.json` is the **historical** pipeline's closure manifest: 270 cells
across eight legacy groups, generated from `REPORTABLE_MATRICES`. The **replacement**
campaign's gate-enforced contract is `matrix_contracts` in
`conf/protocol/replacement-v1.yaml` — twelve registered matrices totalling 415 cells
(matched_tuning 145, stage_1a 84, stage_1b 12, downstream 174), and `pass3_freeze_confirm`
does not appear in it. The same file sets `historical_artifacts_promotable: false`, which
keeps the two ledgers apart by construction. So `expected_runs.json` still listing those eight
cells is not a signal that the replacement design expects them; it is the historical record
doing its job.

*Disposition, therefore: neither of the two branches offered above.* ADR-0008 records the
pre-registered empty-freeze branch executing directly and the Stage 3 confirmation being
**skipped as degenerate** — a documented interpretation, flagged at the time rather than
silently elided. The first branch (edit the overrides from `exploration_margin.json`) asks for
values from a stage whose verdict was "empty". The second branch (remove it from
`dump_expected_runs.py`) collides with ADR-0010: a pre-registered branch is never deleted even
after it has been ruled out. What was actually needed was neither a run nor a deletion but a
**record correction**: the matrix header no longer points at "the audit's S3 disposition" for
a question ADR-0008 had already answered, and ADR-0008 carries a dated addendum recording that
the historical closure certificate stays permanently open at `pass3_freeze_confirm: 0 of 8`,
that this is accurate, and that `--allow-incomplete` is the deliberate at-invocation
acknowledgement for historical analysis.

*Authorization, recorded because the file is frozen.* `conf/matrix/pass3_freeze_confirm.yaml`
is one of ADR-0010's thirteen frozen `conf/` files (exactly thirteen cite the old `Decision N`
numbering; this is one), and AGENTS.md requires explicit authorization to edit one. The thesis
author authorized this specific edit. It is comment-only: no resolved value moved, which
`dump_frozen_configs.py --check` confirms — `docs/freeze/resolved_configs.yaml` is current and
does not contain this matrix at all. What did move is the file's hash in
`docs/freeze/source_manifest.json`, regenerated in the same commit, which is why the assurance
candidate advanced to `a597edb`.

**S3 is closed. No author disposition is required.**

## High-risk assertions: confirm / reject

| Assertion | Disposition |
| --- | --- |
| Stage 1b can be dispatched with an unresolved `p` and silently run at the default | **Confirmed**, with correction: analysis fails closed, so the loss was 12 wasted cells plus a booby-trapped recovery, not silently corrupt evidence. Fixed (C1). |
| No central contract reader gates dispatch | **Partly confirmed — this report's first version overstated it.** `validate_matrix_against_protocol()` (`src/fedmaq/core/protocol.py:71`) *is* a central contract reader, called from `plan_matrix` for repo matrices on the `scientific` ledger; it compares stage, split, ledger, rounds, cell count and a sha256 of the full resolved matrix against `matrix_contracts`. And `selection_domains` is bound into `preregistration_sha256`, not unread. What was true: it had **no test coverage at all** (closed in the second pass), it exempts any matrix with no registered contract, and no check compared a run's *selection state* to anything. Routed (S1) for the selection-domain cross-check only. |
| `control_messages.py` is imported by nothing | **Confirmed.** Its only importer is `tests/test_control_messages.py`. **No action taken, and none recommended pre-freeze:** N29 requires "a versioned, byte-tested control-message schema with declared direction and multiplicity", which the module already is (`CONTROL_SCHEMA_VERSION`, `ControlDirection`, struct-packed, tested). Wiring it into telemetry would change round byte totals, hence the iso-byte protocol, hence every communication claim in the manuscript. That is #101/#103 territory, not a pre-freeze polish. A tripwire pinning it as registered-but-unwired is the author's call. |
| Ablation Configuration 8 is an orphan needing wiring or deletion | **Rejected — already satisfied.** Two tests in `tests/test_config_and_dispatch.py` guard it bidirectionally: `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` asserts the arm is present iff `_frozen_refinements()` is non-empty, and `test_configuration_8_can_express_any_freeze` asserts the config holds every refinement flag off. #101 should close this item with that reason. |
| The golden gate does not distinguish transition from repeatability | **Split.** Harness and tests: **rejected**, already correct. Skill: **confirmed** and freeze-blocking. Fixed (C2). |
| The assurance envelope's content hash is absent | **Confirmed.** Routed (S2). |
| Envelope-only commits must follow the source candidate | **Confirmed as already correct.** `git show --stat 5f85868` touches only the envelope file (35 insertions) over source candidate `0c6028e`. The same sequencing applies to this pass. |
| `tests/test_simulation.py` is cited in ~13 places and will abort operationally | **Rejected for this repository.** The file does not exist, and it is cited **once** in-repo — `docs/research/2026-09-02-pipeline-audit-confirmation.md:307`, itself a note flagging that #45 cites it. It is **not** live in `conf/matrix/pass3_freeze_confirm.yaml`, so there is no dispatch-time abort. The real module is `tests/test_simulation_lifecycle.py`, present and manifest-tracked. Remaining citations are manuscript-side and belong to #101. |

Additionally, **N32 is resolved**: the standalone type gate was red and absent from
`just check`; `typecheck` is now in the `check` recipe and passes.

## Acceptance tests

**Added.** `tests/test_config_and_dispatch.py::test_stage_1b_p_is_explicit_and_fails_closed_until_selection`
— asserts each Stage-1b run carries an `algorithm.p` override, and that while the value is
the `???` sentinel, reading `composed.algorithm.p` raises `MissingMandatoryValue`.
Bidirectional in the idiom of the Configuration 8 tripwire: it also passes once the author
writes a concrete degree in, so it does not block the Stage-1a write-back. The only state
it rejects is the lossy one — `p` absent, or resolvable to a default no selection chose.
Verified to **fail** on the pre-fix matrix (`assert 'algorithm.p' in {'algorithm.omega': '0.25'}`)
and pass after.

**Still missing**, blocked on the routed findings:

- A test that two Stage-1b cells at different `p` cannot collide on disk. Cannot be written
  until `identity_key()` carries `p`/`omega` (S1).
- A startup test that dispatch expansion equals an independently registered authority, and
  that drift in split, wire protocol, stage, treatment, or exact set is rejected. Blocked on
  #96 bullet 11 / #102-15 (S1). Regenerating a ledger from the same edited matrix is not an
  independent check. **Partly delivered by the second pass**: stage, split, ledger, rounds,
  cell count and full-content hash are now tested against `matrix_contracts` for every
  registered matrix. The selection-domain half — seed list, `q` ladder, `p` and `omega`
  support compared to what the matrices expand to — is still absent.
- A test asserting the freeze envelope's declared `content_hash` matches its content (S2).

## Amendments to the plan

1. The plan's instruction to follow the freeze-confirm placeholder convention is **rejected**;
   that convention writes plausible working defaults, which is the defect. Restated after
   the second pass, because the first wording contradicted S3 — S3's booleans *are*
   selection-dependent, and the sentinel is exactly wrong for them: use the `???` sentinel
   only where the consumer subscripts or parses the value, and a declared
   `pending_selection` marker everywhere else. Neither is a comment.
2. N27's completion criteria must name the **skill** explicitly as a verified surface. A
   correct harness with an incorrect skill is not a working gate, and the harness's
   `compare` → `transition` alias preserves the old muscle memory while silently changing
   semantics.
3. Ticket closure must cite every acceptance bullet individually. #96's receipts claimed a
   bullet's first half and dropped its second; three bullets were never mentioned.
4. Record that the analysis-time `p` guard exists, so the Stage-1b risk is not overstated in
   the manuscript as a silent-corruption hazard.

## Author actions before freeze

Step 1 below is **done**. The rest require GPU dispatch on the intended host or tag
authority, and remain the author's.

1. ~~Review and commit the source candidate.~~ **Committed at `7c1fdb7`, then superseded by
   `4c4337b`** (the second pass — see the Addendum). Each was staged to manifest-scope files
   only, with `just check` green on that tree beforehand per ADR-0017, and
   `check_freeze.py --check` reports the certificate current at each. The `.agents/` and
   `docs/` edits landed separately, outside `source_manifest.json`'s scope (`conf/**`,
   `scripts/**`, `src/**`, `tests/**`, `justfile`, `pyproject.toml`, `uv.lock`), so they do
   not move the candidate. **Superseded by the third pass:** `a597edb` edits
   `conf/matrix/pass3_freeze_confirm.yaml` (comment-only, but `conf/**/*.yaml` is in
   `scope.include`) and regenerates `source_manifest.json`, so it is a source change and the
   candidate moved again. **`a597edb` is the revision the new envelope must pin, and it
   supersedes `0c6028e`, `7c1fdb7` and `4c4337b` as the assurance candidate.**
2. Run the golden **repeatability** gate at `a597edb`, using the corrected skill. Its
   preconditions now hold: the commit exists, the tree is clean, and `_capture()` will record
   `dirty: false` — which is what would have failed before.

   **Run it on JupyterHub, not locally.** This was deliberately not executed by the agent,
   and not only on the ownership rule. `repeatability_report()` compares the two captures
   only against each other, so environment agreement is trivially satisfied on any single
   box: the nondeterminism the gate exists to catch — Ray scheduling, cuDNN autotune, kernel
   selection, thread counts — is exactly what a different host would surface. A local PASS is
   therefore the least informative place to run it, while `_write_report()` would still
   write `"status": "PASS"` into the canonical `outputs/golden/step2_repeatability/*.json`
   path. That is a plausible-looking value in a slot nothing validates — C1 and S3's failure
   mode, reproduced in the assurance ledger. The skill is host-pinned for this reason
   (`SKILL.md` line 72: commits, file edits, or GPU-allocation changes between captures
   invalidate them).

```bash
uv run python scripts/golden_diff.py repeatability
```

3. Write the new assurance envelope in a **separate envelope-only commit**, matching the
   `0c6028e` → `5f85868` pattern, and record the repeatability outcome from step 2 in it.

   **Pin the revision the gate actually ran at, read back from the evidence** — not the one
   this report names from memory. `_metadata()` takes `commit` from each capture's
   `run_manifest.json`, so the only revision with gate evidence behind it is whatever
   `first.commit` says in `outputs/golden/step2_repeatability/<alg>.json`. The source
   candidate is `a597edb`, but any docs commit landing on top of it is source-identical
   (`docs/` and `.agents/` are outside manifest scope), so a gate run from today's `main`
   will record that later SHA instead. Either is defensible as the candidate; pinning one
   while the evidence names the other is not. Take it from the JSON:

   ```bash
   uv run python -c "import json;print(json.load(open('outputs/golden/step2_repeatability/fedmaq.json'))['first']['commit'])"
   ```

   Resolve **S2** while writing it, using option (a) or (b) in that section — both are
   mechanical, and leaving the field null a third time is the one outcome to avoid.
   `docs/` is outside manifest scope, so this commit does not disturb the certificate.
4. Disposition S1 and S2 on #86/#92 before declaring freeze — N31 blocks freeze
   consideration while any critical or significant finding is undisposed. S1 rides to #86
   by author decision; S2 is the envelope re-declaration and is the last artifact to land,
   since its hash is only computable once the final commit exists. **S3 needs nothing**: the
   third-pass correction in its section closes it against ADR-0008. `pass3_freeze_confirm`
   does not run, is not deleted, and its three overrides are never to be written — do not
   reach for C1's `???` sentinel there either; it fails open on booleans.

## Change set

First-pass source candidate, committed at `7c1fdb7` (superseded `0c6028e`, and itself
superseded by `4c4337b` — see the Addendum):

- `conf/matrix/power_mean_omega.yaml` — fail-closed `algorithm.p=???` on both Stage-1b runs.
- `tests/test_config_and_dispatch.py` — the acceptance test above, plus `math` and
  `MissingMandatoryValue` imports.
- `docs/freeze/source_manifest.json` — regenerated (`check_freeze.py --write`, 173 files).

Outside `source_manifest.json`, no source blast radius, landed in a separate docs commit so
the pinned candidate stays minimal:

- `.agents/skills/jupyterhub-golden-gate/SKILL.md` — rewritten for the two-operation gate.
- `docs/research/2026-09-03-final-prefreeze-audit.md` — this report.

At `7c1fdb7`, `just check` passed on this change set, end to end and unmasked (exit 0): freeze certificate
current, Ruff format and lint clean, mypy clean, **537 passed** (536 baseline + the one new
acceptance test), all five `--check` generators current, and the assurance fixture green.
Both deterministic digests are unchanged from the baseline —
`d695901297f1745afe385b7b4842d5c2dee6caaa31ddd7cc9ec25c9054ab63ab` (fixture) and
`6d005f54a292596fe645784451d1b531e5f4a2abf5d46aaa1bbfcc06cea1c3d8` (analysis) — so the
matrix edit moved no deterministic surface.

On the requested Ruff polish: it ran and it is clean. `just check`'s `format` and `lint`
stages are `ruff format --check` and `ruff check` over the tree, and both passed on this
change set, as they did at the baseline — so there was no reformatting left to do. `just
fix` was deliberately not invoked, because it rewrites files outside this change set and
would sweep unrelated edits into the source candidate. mypy is likewise clean.

## Addendum — second pass, 2026-09-03: the dispatch guard

Landed at **`4c4337b`**, which supersedes `7c1fdb7` as the assurance candidate. Corrections
this pass forced on the report above are marked inline; the substantive ones are the
`.get` fail-open table in C1, the S1 `selection_domains` reason, the "no central contract
reader" disposition, and the count of unmet #96 bullets (two, not three).

### What C1's sentinel actually guaranteed

Not enough. `???` fails closed under a subscript and under
`_parse_power_mean_degree()`, but `.get(key, default)` returns the default and `.get(key)`
returns `None` — and those are live reads: `quantization_planner.py:103` subscripts `p`
only when the formulation requires it, and `manifest.py:147` records the run with `.get`.
On a boolean the sentinel is not merely weak but inverted:
`alg_cfg.get("soft_voting", False)` reads the string `'???'` as **truthy** and turns the
mechanism on. So the sentinel could not be generalized to S3, and it was not safe on its
own even where it was used.

### The guard

`plan_matrix` now refuses to plan any matrix carrying an unresolved selection, and reports
the run labels and key names. Two markers, because one cannot cover both cases:

- **`???` in an override value**, for keys whose consumer subscripts or parses. Readable at
  the planner because the token sits inside an `overrides` *list* — a string element, not a
  config node — so `to_container(..., resolve=True)` passes it through verbatim.
- **A declared `pending_selection` list**, for keys whose placeholder is *plausible*.
  Clearing the list is the act that records the verdict.

Three properties worth stating, because each was a way to get this wrong:

- **Whole-matrix, and `--only` cannot bypass it.** One unresolved row means the matrix's
  identity is still moving, so rows dispatched under the old identity are not comparable to
  rows dispatched after. This forecloses running `pass3_freeze_confirm`'s fully-resolved
  `fedmaq-unrefined` arm early — a reasonable thing to want. It is a one-line change to
  scope the check to the selected labels if the author prefers that trade.
- **Ahead of protocol validation**, so a pre-selection matrix reports its placeholder rather
  than a fingerprint mismatch that reads as tampering.
- **Run-spec keys are a closed set** (`alg`, `label`, `overrides`, `pending_selection`,
  `seeds`, `variant`). `expand_matrix` reads them through `.get`, so a misspelt
  `pending_selection` would be ignored silently — the same fail-open class, relocated from
  the value to the key.

Verified at the CLI: `run_matrix.py --dry_run` exits 2 on both pre-selection matrices,
naming the labels and keys, with or without `--only`; resolved matrices dry-run normally.

### The protocol-fingerprint divergence `7c1fdb7` created, and why it is not repaired here

Editing `conf/matrix/power_mean_omega.yaml` in the first pass changed its resolved-content
sha256 from `bc52de88…cda9` to a new value, diverging from the digest pinned at
`matrix_contracts.power_mean_omega.sha256`. From that commit until this one, `plan_matrix`
raised on every Stage-1b dispatch. **`just check` could not see it**: nothing in the suite
called `validate_matrix_against_protocol`. A sweep of all twelve registered contracts found
exactly one divergent — the one the first pass edited.

It is deliberately **not** repaired by rewriting the pinned hash. Two reasons: the file
changes again when the Stage-1a verdict is written in, so any digest registered now is
stale before use; and `conf/protocol/replacement-v1.yaml` is the pre-registration authority,
which agents may not edit. Instead the stale state is made honest — the guard fires first
and reports the placeholder — and the matrix header now documents the re-registration debt
the author settles at write-in, naming where the correct digest comes from.

### Tests added

`tests/test_config_and_dispatch.py`, seven new:

| test | what it pins |
| --- | --- |
| `test_the_missing_sentinel_is_not_safe_by_itself` | the four reads above, including the two that fail open |
| `test_dispatch_refuses_an_override_left_at_the_missing_sentinel` | `???`, `'???'` and `+`-prefixed forms all refused |
| `test_dispatch_refuses_a_run_declaring_pending_selection` | the declared marker refuses |
| `test_a_misspelt_run_spec_key_is_rejected_rather_than_ignored` | `pending_selections` cannot disarm the guard |
| `test_dispatch_plans_normally_once_the_selection_is_written_in` | bidirectional; the guard is not a permanent block |
| `test_the_two_pre_selection_matrices_refuse_to_dispatch_today` | it fires on the real matrices |
| `test_every_registered_matrix_either_matches_its_contract_or_is_pre_selection` | first coverage of `validate_matrix_against_protocol` |

The last one closes the class of defect the first pass introduced: for every name in
`matrix_contracts`, the matrix either validates against the protocol **or** is pre-selection
*and* provably undispatchable. The pre-selection exemption is computed from
`unresolved_selection()`, not from an allowlist, so it expires by itself at write-in — at
which point `power_mean_omega`'s stale digest fails immediately, which is the intended
signal.

### Residual limits — what this does not guarantee

- `pending_selection` protects against *forgetting* to write a value in. It does not protect
  against clearing the list without editing the override. A declared marker cannot; only a
  value the consumer refuses can, and for booleans no such value exists.
- The guard sees the dispatch path. It does not constrain what the chosen values *are*; the
  selection-domain cross-check (S1, bullet 11) is still absent.
- `validate_matrix_against_protocol` still returns silently for any matrix with no
  registered contract, so `pass3_freeze_confirm` (`ledger: historical`) is fingerprint-
  ungated. Only the selection guard covers it.

### Gate status

`just check` green at `4c4337b`, exit 0: freeze certificate current, Ruff format and lint
clean, mypy clean, **544 passed** (537 baseline + 7), all five `--check` generators current,
assurance fixture green. Both deterministic digests unchanged from the baseline
(`d69590…63ab` fixture, `6d005f…c3d8` analysis), so nothing here moved a deterministic
surface. Tree clean, `_git_provenance()` reports `dirty: False` at `4c4337b` — the
repeatability gate's preconditions hold.

**This changes nothing about what remains author-owned**: the repeatability gate on
JupyterHub, the envelope-only commit, the S1/S2/S3 dispositions, the `power_mean_omega`
re-registration at write-in, and whether `pass3_freeze_confirm` runs at all.
