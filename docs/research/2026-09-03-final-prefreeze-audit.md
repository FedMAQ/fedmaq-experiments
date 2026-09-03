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

Two freeze-blocking defects were found and fixed in this pass. Three acceptance criteria
of the closed issue #96 remain unmet; they are not fixed here because each carries blast
radius beyond a pre-freeze polish, and they are routed below.

Three findings require author disposition before freeze is declared — **S1**, **S2**, and
**S3**. S3 is the generalization of C1 to the pre-freeze confirmation matrix; unlike C1 it
must **not** be fixed by the same mechanism, for the reason given in its section.

This report certifies neither that the pipeline is ready, nor that #101 is complete, nor
that manuscript reconciliation may begin. The source changes landed here **invalidate the
#100 assurance candidate**: a fresh golden repeatability gate and a new assurance envelope
must be run by the thesis author against the new candidate before freeze is considered.
That candidate is now committed at **`7c1fdb7`**, statically green but not yet gated — the
repeatability run is what converts it from a candidate into an assured one, and it has not
happened.

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
3. `scripts/run_matrix.py` performs no dispatchability check. A census of
   `run_matrix.py`, `matrix_executor.py`, and `matrix_planner.py` for
   `stage_ledgers|stage_manifest|dispatchable|selection|placeholder|PENDING` returns one
   hit: `matrix_executor.py:81: "state": "pending"`.
4. `tests/test_stage_contracts.py::test_ledgers_have_required_counts_and_are_disjoint`
   asserts Stage 1b has **12 cells** — which wrong-`p` evidence satisfies exactly.

The generated ledgers cannot detect this, for the reason
[Lens 3](2026-09-02-reporting-evidence-traceability-audit.md) already gave: their
scientific identities are derived from the matrices they check, so an edit and a
regeneration move both sides together. Confirmed empirically — after adding the `p`
override, all five `--check` generators still reported *current*, because none of them
reads `algorithm.p`.

**Partial mitigation that does exist.** `select_power_mean_omega_iso_byte()` in
`scripts/analysis.py:2006-2014` routes runs whose `experiment_group` equals the Stage-1b
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

The mechanism is worth stating precisely, because the guarantee is conditional. The raise
comes from the `alg_cfg["p"]` read at `src/fedmaq/core/quantization_planner.py:103` while
the algorithm config is still a `DictConfig`. `OmegaConf.to_container(cfg, resolve=True)`
does **not** raise on `MISSING` — it yields the plain string `'???'`. So on any path that
flattens the config to a plain dict before the planner sees it, that read returns `'???'`
and `_parse_power_mean_degree()` raises `ValueError` instead. Both paths fail closed before
training, which is what matters here, but only the `DictConfig` path is pinned by the test.

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

### S1 — #96 closed with three acceptance criteria unmet (significant, ROUTED)

Issue #96 is closed. Walking its acceptance list against the code, roughly 17 of 20 bullets
are met — including round-0 acceptance, the all-NaN optional-column skip, JSONL as a
required checked artifact, `derive_seed` hash-combination, parsed-number variant matching,
the non-zero closure exit behind `--allow-incomplete`, per-seed iso-byte budgeting, the
Gate-2 precondition, and the FedPAQ-pipeline / memory-sensitivity / variable-vs-uniform
readouts. Three are not:

| # | Criterion | State |
| --- | --- | --- |
| 3 (half) | "the identity key incorporates `p` and `omega` so two Stage-1b cells cannot collide on disk" | **Unmet.** `identity_key()` takes neither. The closing receipt claims "Manifests expose p/omega/formulation" — true, at `src/fedmaq/core/manifest.py:147-148` — and silently drops the identity-key half of the same bullet. |
| 4 | "The Stage-1b matrix carries explicit `p` overrides marked as pre-dispatch placeholders" | **Was unmet.** Fixed as C1; claimed by no closing receipt. |
| 11 / #102-15 | "Selection domains … registered in the replacement protocol, with a startup cross-check against the matrices"; "Startup validation compares the dispatch expansion against that independently registered authority" | **Unmet.** `selection_domains` has exactly one reference in the entire codebase — `src/fedmaq/core/protocol.py:67`, `document.get("selection_domains", {})` — a passthrough with a default. Nothing writes it, validates against it, or tests it. |

The three unmet bullets are one theme: **no authority independent of the matrices
constrains dispatch.** That is the disposition Lens 3 routed to #96 and marked as the
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
lists retired matrices (`formulation_study`, `pass2_explore`, `pass2_factorial`,
`pass3_freeze_confirm`). The live contract is 415 = 145 + 84 + 12 + 174. Left unedited
deliberately: #96 states that every number is written once after the code settles, and
#101 owns that reconciliation. Editing it now would write a number that pass will rewrite.

### M2 — Unreachable defensive branch in the Stage-1b selector (minor, informational)

The `else` branch at `scripts/analysis.py:2024-2036` admits `omega0.25` / `omega0.75` /
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

### S3 — The second placeholder is live, and C1's fix must NOT be copied to it (significant, ROUTED)

A sweep of `conf/` for placeholder patterns — the generalization C1 motivates — returns
exactly one other site: `conf/matrix/pass3_freeze_confirm.yaml:83`, where the
`fedmaq-surviving-set` arm carries three real, working booleans under a
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

## High-risk assertions: confirm / reject

| Assertion | Disposition |
| --- | --- |
| Stage 1b can be dispatched with an unresolved `p` and silently run at the default | **Confirmed**, with correction: analysis fails closed, so the loss was 12 wasted cells plus a booby-trapped recovery, not silently corrupt evidence. Fixed (C1). |
| No central contract reader gates dispatch | **Confirmed.** `run_matrix.py` consults no ledger, manifest, or selection provenance; `selection_domains` is an unread passthrough. Routed (S1). |
| `control_messages.py` is imported by nothing | **Confirmed.** Its only importer is `tests/test_control_messages.py`. **No action taken, and none recommended pre-freeze:** N29 requires "a versioned, byte-tested control-message schema with declared direction and multiplicity", which the module already is (`CONTROL_SCHEMA_VERSION`, `ControlDirection`, struct-packed, tested). Wiring it into telemetry would change round byte totals, hence the iso-byte protocol, hence every communication claim in the manuscript. That is #101/#103 territory, not a pre-freeze polish. A tripwire pinning it as registered-but-unwired is the author's call. |
| Ablation Configuration 8 is an orphan needing wiring or deletion | **Rejected — already satisfied.** `tests/test_config_and_dispatch.py:145,309` guard it bidirectionally: `test_configuration_8_exists_only_while_there_is_a_layer_to_remove` asserts the arm is present iff `_frozen_refinements()` is non-empty, and `test_configuration_8_can_express_any_freeze` asserts the config holds every refinement flag off. #101 should close this item with that reason. |
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
  independent check.
- A test asserting the freeze envelope's declared `content_hash` matches its content (S2).

## Amendments to the plan

1. The plan's instruction to follow the freeze-confirm placeholder convention is **rejected**;
   that convention writes plausible working defaults, which is the defect. Fail-closed
   sentinels only for selection-dependent values.
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

1. ~~Review and commit the source candidate.~~ **Committed at `7c1fdb7`**, staged to exactly
   the three manifest-scope files, with `just check` green on that tree beforehand per
   ADR-0017. `check_freeze.py --check` reports the certificate current at that revision. The
   `.agents/` and `docs/` edits landed separately, outside `source_manifest.json`'s scope
   (`conf/**`, `scripts/**`, `src/**`, `tests/**`, `justfile`, `pyproject.toml`, `uv.lock`),
   so they do not move the candidate. **`7c1fdb7` is the revision the new envelope must pin,
   and it supersedes `0c6028e` as the assurance candidate.**
2. Run the golden **repeatability** gate at `7c1fdb7`, using the corrected skill. Its
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
   `0c6028e` → `5f85868` pattern. It pins `7c1fdb7`, not `0c6028e`, and it should record the
   repeatability outcome from step 2. Resolve **S2** while writing it, using option (a) or
   (b) in that section — both are mechanical, and leaving the field null a third time is the
   one outcome to avoid. `docs/` is outside manifest scope, so this commit does not disturb
   the certificate.
4. Disposition S1, S2, and S3 on #86/#92 before declaring freeze — N31 blocks freeze
   consideration while any critical or significant finding is undisposed. S3 is the one
   that can still corrupt a dispatch: decide whether `pass3_freeze_confirm` runs, and if it
   does, its three overrides must be written from `exploration_margin.json` first. Do not
   reach for C1's `???` sentinel there; it fails open on booleans.

## Change set

Source candidate, committed at `7c1fdb7` (supersedes `0c6028e` as the assurance candidate):

- `conf/matrix/power_mean_omega.yaml` — fail-closed `algorithm.p=???` on both Stage-1b runs.
- `tests/test_config_and_dispatch.py` — the acceptance test above, plus `math` and
  `MissingMandatoryValue` imports.
- `docs/freeze/source_manifest.json` — regenerated (`check_freeze.py --write`, 173 files).

Outside `source_manifest.json`, no source blast radius, landed in a separate docs commit so
the pinned candidate stays minimal:

- `.agents/skills/jupyterhub-golden-gate/SKILL.md` — rewritten for the two-operation gate.
- `docs/research/2026-09-03-final-prefreeze-audit.md` — this report.

`just check` passes on this change set, end to end and unmasked (exit 0): freeze certificate
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
