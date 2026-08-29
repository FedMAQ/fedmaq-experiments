# Gate 1 — Literature/protocol audit: evidence

Envelope: `docs/freeze/assurance-envelope-2026-08-29.json` (#75). Delta audited:
`530f8ae..6783a30` (46 commits, 32 behavior-bearing), plus re-verification of prior
verdicts the delta could disturb, plus a fresh cross-check against
fedmaq-literature. Repos and pins: fedmaq-experiments candidate `6783a30`
(re-pin remains valid at `main`'s current `667e5f5`, which touches only the
gate-0 envelope artifact — not a `scope.include` path, not claims-bearing
content); fedmaq-literature `1be87e08d232b8d47ec9c71e80922d7c8235bb5a`;
fedmaq-manuscript `5cc82393d639321c240897dbd7eeb728dfc82a02`.

## 1. Source-grounded findings

### fedmaq-experiments (the delta)

- **Simulation lifecycle decomposition** (`5617bf0`, `d01650b` — architecture
  pass 3). `run(cfg) -> TelemetryManager` split into `SimulationAssembly`,
  `SimulationRun`, `SimulationPlan`, `SimulationApplications`,
  `SimulationBuilder`. Behavior-preserving with one deliberate semantic change:
  the original `finally` block could mask a primary exception (e.g. a
  `TimeoutError`) with a secondary finalization error; `SimulationRun.execute()`
  now explicitly preserves the primary error and logs the secondary one. Tested
  directly — `tests/test_simulation_lifecycle.py::
  test_simulation_run_preserves_timeout_when_finalization_also_fails` exercises
  exactly this path. **Finding, not a defect**: this is a change on a path the
  golden capture/compare cannot exercise (it only sees successful runs), so it
  is recorded here rather than left implicit. Disposition: no repair needed —
  intentional, tested, and correctly favors surfacing the primary failure.
- **Architecture pass 3 context/metrics seams** (`d01650b`). Touches
  `strategy_hooks/fedmaq.py` and `strategy_hooks/fedmd.py`; mechanical
  seam-typing, covered by new `tests/test_config_defaults.py` and
  `tests/test_loss_metrics_capability.py`. No behavior mismatch found.
- **Architecture pass 2 seam recording** (`6ac56c8`, #61). Docs-only
  (`CONTEXT.md`, ADR-0007, ADR-0018) — no code path changed, so pass 1/2
  verdicts are unaffected by this delta on their own terms (see carry-forward
  list, §3).
- **Quantizer unbiasedness / l2-l∞ split** (#24, `src/fedmaq/baselines/
  quantization.py`). `_stochastic_round` gives `E[result]=scaled`
  (floor w.p. `1-frac(x)`, else ceil), replacing biased `np.round`. Scale uses
  l2 (`np.linalg.norm`) for the `q>1` path and l∞ (`np.max(np.abs)`)
  unconditionally for the `q<=1` sign-quantization path.
  `FedMAQPostProcessCompressionHook` (error-feedback path, `postprocess.py`)
  deliberately keeps l∞ even at `q>1` per ADR-0019, which records the
  contraction-property failure under l2 with error feedback as the reason and
  explicitly leaves the formal re-derivation open (owned by #42, out of scope
  by declaration — see §2). Matches the pinned manuscript's own disclosure
  (chapter_3.tex lines 84/86, §"fedmaq-manuscript" below). No undisclosed
  mismatch.
- **`tier1_binding_fraction` metric fix** (`e32638d`, present at HEAD in
  `strategy_hooks/fedmaq.py`). Compares `_snap_floor(min(q_k_max, q_hat))`
  against `_snap_floor(q_hat)` — realized/snapped bit-widths — rather than the
  raw unsnapped cap. Confirmed present and unchanged by later commits in the
  delta.
- **Power-mean formulation** (#34/ADR-0021, `src/fedmaq/core/
  quantization_planner.py`). `_weighted_power_mean` implements the named
  limits ADR-0021 records: ω=0→ñ, ω=1→g̃, p=MINIMUM_POWER_MEAN→min(g̃,ñ),
  p=0→geometric-mean-style weighted product, p<0 with a zero operand→0.0
  (avoids div-by-zero), else the general power-mean formula. Cross-checked
  against ADR-0021's decision text (the harmonic/minimum-cell gap ADR-0012
  left, and the F1/F2-as-one-continuous-axis reframing) — code matches the
  taxonomy as decided. Not independently re-derived against chapter_3.tex's
  formulation section wording this pass; carried forward on #21's basis (§3).
- **Byte-accounting seam** (#25/#26). `transport.py` centralizes every arm
  through `UploadReport.from_payloads` → `measure_bytes` (zlib), replacing the
  five independent implementations the module docstring itself names
  (FedPAQ/DAdaQuant's analytic formula, FedMAQ's zlib path, the identity
  hook's raw `nbytes`, FedKD's raw SVD factor count, and FedDistill's
  undocumented fifth). Spot-checked at `6783a30`: `measure_bytes`/
  `measure_payload` call sites in `transport.py`, `client.py`,
  `strategy_hooks/cfd.py`, `strategy_hooks/feddistill.py` all route through
  the shared seam; no arm-local re-implementation remains.
  - **DAdaQuant secondary byte axis** (`d3919e7`, #26): `dadaquant_coder.py`
    implements 0-run-length + Elias omega coding, logged *alongside* the
    primary measured-bytes seam rather than substituted into it, per the
    2026-08-27 decision on #21 and ADR-0020. Matches the fedmaq-literature
    DAdaQuant node's mechanism description exactly (§"fedmaq-literature"
    below) — DAdaQuant is correctly identified as the only baseline whose
    source paper mandates a transport coder beyond quantization itself.
  - **FedDistill download-leg fix** (`d031cc8`): closed a bug where
    `FedDistill`'s `download_size_bytes` override still returned raw `nbytes`
    for the weight and logit legs, over-charging the baseline 9.4% on the
    headline cumulative-MB axis in FedMAQ's favor. The override now delegates
    the weight leg to the default (seamed) path and measures the logit
    payload through the same `measure_bytes` call. Confirmed at `6783a30`:
    `strategy_hooks/feddistill.py::download_size_bytes` imports and returns
    `UploadReport`, calls `super().download_size_bytes()` for the weight leg,
    and wraps the logit payload in `UploadReport.from_payloads`. Tested
    (`tests/test_payload_capture.py`, round-1 and round-2 branches).

### fedmaq-literature (pinned `1be87e08d`)

Checked the five method nodes for the algorithms whose code changed in this
delta — `methods/fedpaq.md`, `methods/dadaquant.md`, `methods/feddistill.md`,
`methods/fedkd.md`, `methods/fedmaq.md` — against the corresponding code at
`6783a30`:

- **FedPAQ**: node specifies an unbiased QSGD-style quantizer on the model
  delta. Matches `quantization.py`'s unbiased stochastic rounding (above).
- **DAdaQuant**: node specifies doubly-adaptive precision (time + client) over
  a Federated-QSGD engine with 0-run-length + Elias-ω coding. The coding half
  matches `dadaquant_coder.py` exactly (above); the doubly-adaptive precision
  logic itself was not touched by this delta and is not re-derived here.
- **FedDistill**: node specifies per-label average logit exchange, no weight
  transfer. Matches `strategy_hooks/feddistill.py`'s download-leg fix, which
  measures exactly a weight leg (delegated) and a logit-matrix leg — consistent
  with the node's stated payload shape.
- **FedKD**: node specifies mentee-mentor mutual distillation + dynamic-SVD
  gradient compression, only the mentee communicated. Not touched by this
  delta beyond the byte-accounting seam (`76360a1`, #25) adding a pre-encoding
  `payload_bytes` companion to the download leg — a reporting-only change, no
  mechanism change. Consistent with the node.
- **FedMAQ**: node specifies the two-tier precision scaling (hard cap +
  power-mean-family soft allocation) and server-side proxy ensemble
  distillation. Matches `quantization_planner.py`'s power-mean implementation
  and `strategy_hooks/fedmaq.py`'s `tier1_binding_fraction` fix (both above).

The remaining ~70 baseline/other-paper nodes under `fedmaq-wiki/papers/*.md`
are out of scope by declaration (§2, owning ticket #11) and were not read this
pass.

### fedmaq-manuscript (pinned `5cc82393`, chapter_3.tex)

- Line 84: stochastic-vs-deterministic rounding claim and its role in FedPAQ's
  convergence guarantee — matches `_stochastic_round`.
- Line 86: the l∞ error-compensation-path disclosure, including the
  explicitly-left-open theoretical question ("Whether the noise bound
  developed later in this section carries over unchanged to the ℓ∞ variant is
  a separate question this thesis leaves open") — matches
  `FedMAQPostProcessCompressionHook`'s l∞ choice and ADR-0019.
- Line 100: the QSGD bound constant `min(d/s², √d/s)` — this is exactly the
  surface #42's scope-boundary entry covers (formal re-derivation not
  performed here; recorded as out of scope by declaration, §2).

No undisclosed manuscript/code mismatch found on the surfaces checked.

### fedmaq-literature FedMAQ node — fresh pipeline-bearing constant/structural
### sweep (the #11 manuscript arm of this gate)

The gate-1 notes name a fresh #11 pass as the manuscript arm of this gate; #11
itself stays open (Living/Recurring — see §3), but the arm is discharged here
by checking every pipeline-bearing claim the `fedmaq.md` literature node makes
against the candidate pin, bounded by the scope boundary (§2 entry 1 excludes
only `sec:theo_bounds_quant`; nothing else in the manuscript chapters is
excluded):

- `Q = {1,2,3,4,5,6,7,8,16,32}` — matches `conf/algorithm/fedmaq.yaml`'s
  `bit_widths`.
- `c_unit = 512` MB — matches `fedmaq.yaml`'s `c_unit: 512.0`.
- `c_k ~ U(2048, 16384)` — matches `strategy.py:112`,
  `rng.uniform(2048.0, 16384.0, ...)`, comment cites §4.1.
- `D_proxy = 3000`, held out pre-Dirichlet, never client-touched — matches
  `conf/experiment/default.yaml`'s `num_public_samples: 3000` (§4.3 Table 4.1
  cited in-line) and `partitioning.py`'s `generate_or_load_partitions`: the
  public pool is sliced off (`class_indices[c] = np.setdiff1d(...)`, removing
  selected public indices from the pool) in "Step 1" before any subsequent
  Dirichlet/writer allocation to clients — public and client indices are
  disjoint by construction, not by convention.
- `T = 1.0` — matches `fedmaq.yaml`'s `temperature: 1.0`, which the file's own
  comment records as a decision, not a coincidental default.
- `R = 100` — matches `conf/experiment/default.yaml`'s `total_rounds: 100`
  (comment cites §4.3 Table 4.1).
- `ω = 0.5`, `κ = 1.0`, `τ_g = τ_n = 0.5` — match `fedmaq.yaml`'s `gamma1`/
  `gamma2: 0.5` and `kappa: 1.0`, `tau_g`/`tau_n: 0.5`, and
  `quantization_planner.py`'s `omega` default of `0.5`.
- Two-stage aggregation with parameter-averaging as the noise attenuator —
  matches `strategy_hooks/fedmaq.py::aggregate_fit`: it receives
  `aggregated_parameters` already produced by the outer FedAvg strategy, then
  runs `distill_ensemble_into_global` as a second, distinct refinement stage.
- Equal-weight teachers, no uncertainty-based selection (unlike DynFed) —
  matches `fedmaq.yaml`'s `soft_voting: false`, FROZEN per Decision 79/80 (the
  entropy/precision-weighted alternative exists in code but is config-gated
  off in the shipped configuration).
- No client-side distillation path — matches `fedmaq.yaml`'s
  `client_kd_reg: false`.
- Single scalar bit-width per client-round, not layer-wise — matches
  `strategy_hooks/fedmaq.py`: `self._current_plan.client_q` is a
  `dict[cid, int]`, one bit-width per client per round.

All constants and structural claims checked match the code and config at
`6783a30`. No mismatch found; no escalation to the author is triggered by
this pass (contrast: a mismatch here would escalate rather than route to
gate-2 repair, since `conf/algorithm/fedmaq.yaml` is frozen file-level under
the existing tag).

## 2. Scope boundary (recorded verbatim, per `boundary_is_invalidating`)

Source: `docs/freeze/assurance-envelope-2026-08-29.json`, `scope_boundary`
key, re-fetched from the live envelope this gate. Reproduced verbatim below,
unedited:

```json
{
  "source": "FedMAQ/fedmaq-experiments#74, amendment 2026-08-29 section 1",
  "in_scope": "fedmaq-experiments, fedmaq-literature and fedmaq-manuscript, covering all compared algorithms plus FedMAQ, source fidelity, configuration/protocol, telemetry, provenance, manuscript claims, and simulation-lifecycle invariants affected by architecture pass 3.",
  "out_of_scope_by_declaration": [
    {
      "surface": "fedmaq-manuscript chapter_3.tex sec:theo_bounds_quant - the eps = min(d/s^2, sqrt(d)/s) bound, the Tier-2 rationale, and the sum_k p_k^2 argument",
      "owning_ticket": "FedMAQ/fedmaq-experiments#42",
      "basis": "A formal re-derivation against the shipped l-infinity error-feedback operator is not a routine repair. ADR-0019 Consequences records it open with an explicit instruction not to infer a verdict. The gate reads PASS within a boundary that names what it did not check, instead of PASS over a known mismatch."
    },
    {
      "surface": "fedmaq-journal-article paper.tex front matter and sections/**",
      "owning_ticket": "FedMAQ/fedmaq-experiments#41",
      "basis": "Outside pipeline-readiness evidence; #41 carries the site inventory and the author freeze-historically disposition."
    },
    {
      "surface": "fedmaq-literature fedmaq-wiki/papers/*.md (~70 baseline/other-paper nodes)",
      "owning_ticket": "FedMAQ/fedmaq-experiments#11",
      "basis": "Swept but not read in full by pass 22. Living-log work, not a pipeline-readiness contract."
    },
    {
      "surface": "fedmaq-experiments docs/agents/**, docs/audits/**, docs/experiments/**, docs/freeze/**",
      "owning_ticket": "FedMAQ/fedmaq-experiments#11",
      "basis": "Not audited at all by pass 22. docs/freeze/** is gate input - its integrity is checked by the freeze certificate and the golden gate, not by prose audit."
    },
    {
      "surface": "fedmaq-analyses, fedmaq-presentations",
      "owning_ticket": null,
      "basis": "Already excluded as pipeline evidence by #74 Implementation Decisions; restated here so the boundary is one list."
    }
  ],
  "boundary_is_invalidating": "A change to this boundary is itself a gate invalidation. Any gate reaching PASS must record this boundary verbatim in its evidence.",
  "pipeline_freeze_proceeds_with_42_open": true
}
```

## 3. Carried-forward verdicts (named, with basis)

- **Architecture passes #46/#53** (passes 1 and 2): not re-examined directly
  this gate. Basis restated to match what was actually verified: `git log
  530f8ae..6783a30 --oneline` was grepped for pass 1/pass 2 attribution
  (`#46`, `#53`, `pass 1`, `pass 2`, and spelling variants) and the only match
  is `6ac56c8` ("docs: record architecture pass 2 seams (#61)"), which is
  docs-only (`CONTEXT.md`, ADR-0007, ADR-0018 — no code path touched). This
  confirms no commit in the delta is *attributed by message* to pass 1 or
  pass 2 work; it is not a check against the passes' own recorded surface
  lists (#46/#53 were not individually re-read this gate), so the carry-
  forward rests on commit attribution, not on an independent surface-by-
  surface re-derivation.
- **Architecture pass 3 (#73)**: not carried forward — freshly verified this
  gate (§1, simulation lifecycle + context/metrics seams).
- **#24/#25/#26 quantizer/byte-accounting seam**: not carried forward —
  freshly verified this gate in full (§1: quantizer unbiasedness/scale split,
  the unified `measure_bytes` seam, the DAdaQuant secondary axis, and the
  FedDistill download-leg fix).
- **#34 power-mean reformulation**: not carried forward — freshly re-derived
  this gate. The named-limit implementation was checked against ADR-0021's
  decision text (§1), and separately `chapter_3.tex`'s own formulation
  section (lines 151-192 at the pinned manuscript revision) was freshly
  grepped and compared: the general formula
  `M_p = (ω·g^p + (1-ω)·n^p)^(1/p)`, the `p=0` weighted-geometric special
  case `g^ω · n^(1-ω)`, the `ω=0`/`ω=1` endpoint limits, the `p→-∞` minimum
  limit, and the `p<0`-with-a-zero-operand `→0` guard were all read directly
  from `_weighted_power_mean` (`quantization_planner.py` lines 169-184) at
  `6783a30` and match the manuscript's Eq. 165-166 general formula, the p=0
  case at line 172, the minimum limit at line 170, and the zero-operand rule
  at line 174, line-for-line in substance. No wording-level mismatch found;
  this is a fresh, fully-read check, not a carry-forward from #21.
- **#11 pass 22's unreached surfaces** (the ~70 literature `papers/*.md`
  nodes and fedmaq-experiments `docs/agents/**`, `docs/audits/**`,
  `docs/experiments/**`, `docs/freeze/**`): remain out of scope by
  declaration (§2, entries 3-4, owning ticket #11) and are not read by this
  gate. This is distinct from the fresh #11 manuscript-arm pass performed in
  §1 above (the FedMAQ literature node vs. the candidate pin's config/code),
  which is in scope and was freshly done — that pass discharges the gate's
  manuscript-arm requirement without reaching into entries 3-4's excluded
  surfaces. #11 itself is Living/Recurring and stays open regardless.

## 4. No unresolved static mismatch inside the boundary

Across the findings in §1, every code/manuscript and code/literature
correspondence checked matched its source, or the discrepancy was already
disclosed and dispositioned (the l∞ error-feedback exception, ADR-0019;
the DAdaQuant secondary-axis parallel-not-substituted design, ADR-0020). The
one substantive defect found and fixed within the delta itself
(`d031cc8`, FedDistill's 9.4% download over-charge) is already repaired,
tested, and confirmed present at the candidate pin — it is not an open
mismatch. No unresolved static mismatch was found inside the scope boundary
defined in §2.

## Disposition

Gate 1: **PASS**, evidence per §1-4 above.
