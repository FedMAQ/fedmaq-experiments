# FedMAQ Thesis Domain

Multi-adaptive quantization and ensemble distillation for memory-constrained federated learning under non-IID data. Canonical glossary for terms shared across `fedmaq-experiments` (code) and `fedmaq-manuscript` (thesis) — resolves naming drift between the two.

## Authority map

Six-repo thesis workspace. `fedmaq-experiments` is the domain hub — sibling repos
index its `.agents/rules/` rather than duplicating domain content.

| Repo | Owns | Agent entry |
| --- | --- | --- |
| [fedmaq-experiments](./) | Code, Hydra, Flower, WandB, **the domain rules** | `AGENTS.md`, `CLAUDE.md` |
| [fedmaq-literature](../fedmaq-literature/) | PDFs, markdown conversions, OKF knowledge graph | `AGENTS.md`, `CLAUDE.md` |
| [fedmaq-analyses](../fedmaq-analyses/) | Notebooks, thesis figures | `AGENTS.md`, `CLAUDE.md` |
| [fedmaq-manuscript](../fedmaq-manuscript/) | LaTeX thesis (Ch 1–6), **its own writing rules** | `AGENTS.md`, `CLAUDE.md` |
| [fedmaq-presentations](../fedmaq-presentations/) | Beamer slides | `AGENTS.md`, `CLAUDE.md` |
| [fedmaq-journal-article](../fedmaq-journal-article/) | IEEE Access submission | `AGENTS.md`, `CLAUDE.md` |

**Cross-repo rule:** non-experiments repos must not duplicate domain content; they
index `../fedmaq-experiments/.agents/rules/`. The workspace agentic-context
contract is [ADR-0015](docs/adr/0015-workspace-agentic-context-contract.md).

**Within this repo**, when two sources disagree: `conf/**` beats prose describing
it; `docs/adr/` beats everything for *why*; the pinned GitHub Issues beat every
file for *what is true right now*. Run counts, dispatch state and sync status live
only in Issues — if you find a number in a tracked file, it is stale by
construction.

## Working conventions

- **One canonical home per fact.** A number, status or decision lives in exactly
  one place; everything else points at it and never restates it.
- **No archives.** Git history is the record; superseded documents are not parked
  in an `archive/` folder.
- **Reference lives behind pointers.** Settled, rarely-touched material belongs in
  `docs/agents/`, out of the always-loaded rules.
- **No committed handoff file.** Session-to-session orientation is temporary and
  belongs to the handoff procedure, not a tracked document.
- **No section numbers in heading titles** (`## Important Context`, not
  `## 1. Important Context`) — avoids renumbering churn and broken anchors.
  Explicit IDs stay: `ADR-0007`, audit finding `F10`, manuscript `§4.1`.
- **`docs/plans/` is active-only.** A plan exists only while it has open
  questions; on resolution it becomes an ADR and the plan file is deleted.

## Language

### Precision Scaling (Section 3.3)

**Soft quality signal**:
The blended [0,1] score $s_k^{(t)}$ combining a client's normalized gradient norm and normalized dataset size. Code: computed inline as `term` per formulation branch in `fedmaq.py`, not materialized as a standalone variable.
_Avoid_: intermediate signal, blended score

**Soft quality target**:
The bit-width value $\hat q_k^{(t)}$ derived from the soft quality signal, before Tier-1 clamping. Code: `q_hat` in `fedmaq.py`.
_Avoid_: soft quality function.

**Formulation**:
The current recut uses the `power_mean` family plus structurally separate resource-only, gradient-primary, and threshold rules; [ADR-0021](docs/adr/0021-power-mean-formulation-family.md) and the matrix/config files own that design. The frozen v1 record uses numeric F0–F4 labels and `formulation: 2`; retain those labels when interpreting v1 configs and evidence, but do not present them as the current recut.
_Avoid_: "Alternative N" for "Formulation N" and "soft quality-target formulation" for "soft quality target". "Linear" names F1 and "multiplicative" names F2. Do not call F0 a non-adaptive control: it is resource-only at Tier 2 while Tier 1 remains adaptive.

**Power-mean family**:
The two-signal aggregate $M_p(\tilde g,\tilde n;\omega) = (\omega\tilde g^p + (1-\omega)\tilde n^p)^{1/p}$ for finite nonzero $p$, where $p$ is the **compensation degree** and $\omega$ is the state-signal weight. Its named limiting forms are arithmetic ($p=1$), weighted geometric ($p\to0$), harmonic ($p=-1$), and minimum ($p\to-\infty$). The design, limit, zero, and ablation semantics are [ADR-0021](docs/adr/0021-power-mean-formulation-family.md).
_Avoid_: treating $p$ as a second formulation identifier.

**Formulation constants**:
The current names are the manuscript's canonical Greek symbols and their aligned config keys:

| Canonical | Config key | Role |
|---|---|---|
| $\kappa$ | `kappa` | Formulation 3 data modulator — renamed from `lambda_val` via #20 |
| $\tau_g$, $\tau_n$ | `tau_g`, `tau_n` | Formulation 4 thresholds (already aligned) |

The power-mean family uses `p` and one `omega`; the historical v1 `gamma1`/`gamma2` names remain only when describing that frozen record. Do not rename manuscript symbols toward retired config keys.

**Bit-width**:
A discrete value from the permissible set $\mathcal{Q} = \{1,2,3,4,5,6,7,8,16,32\}$ — never an arbitrary continuous integer.

**Tier 1 / Tier 2**:
FedMAQ's two-tier precision scaling design. Tier 1 is the hard feasibility constraint from client memory ($Q_k^{max}$), computed as a separate `min()` clamp in code, never blended into the soft quality signal. Tier 2 is the soft quality optimization (signal, target, formulation) layered on top and floored by Tier 1's cap.
_Avoid_: "three coequal dimensions of awareness"; resource is a Tier-1 hard clamp,
not a third Tier-2 signal.

### Tier 1 Memory Ceiling Telemetry

Per-round evidence that the Tier 1 clamp binds rather than sitting inert.
Implemented in `e547b50`; plotted by `scripts/memory_ceiling.py`.

**Tier 1 ceiling** ($Q_k^{max}$):
The per-client bit-width cap `max(1, floor(c_k / c_unit))` derived from the
client's modelled memory budget, exposed without changing the assigned `q`.
Logged as `algorithm/fedmaq/{avg,min,max,std}_q_k_max`.
_Avoid_: memory ceiling (bare — ambiguous with host VRAM, below)

**Tier 1 binding fraction**:
The fraction of sampled clients whose realized bit-width differs with and
without the Tier 1 clamp — i.e. where Tier 1, not Tier 2, is the active
constraint. This is what separates "the clamp is binding on most clients" from
"the clamp is inert and Tier 2 is doing all the work"; without it a realized
`avg_q` is unreadable. Logged as `algorithm/fedmaq/tier1_binding_fraction`.
Compare snapped bit-widths, not the raw ceiling: equal snapping means the clamp
did not change the assignment.

Host VRAM/RAM is simulator infrastructure, not modelled client memory, and is not
evidence of device feasibility.

### Ensemble Distillation

**Ensemble distillation**:
The head term for **FedMAQ's own** server-side second aggregation stage. `server-side`, `proxy-based`, and `multi-teacher` are contrastive prefixes for client-side, data-free, or single-teacher foils. Bare "distillation" is a valid short form after the mechanism is established.
_Avoid_: attaching "knowledge" to FedMAQ's own mechanism. **Knowledge distillation** remains the general family and other methods' mechanism; **federated distillation** names the FedDistill/FD lineage; **data-free distillation** names others' work.

**Knowledge distillation** remains the general family and is correct for other
methods and general client-side statements. "KD" remains the abbreviation.

**What this rule governs.** Manuscript prose, current ADR prose, and the FedMAQ
entries in `fedmaq-literature/kg/`. Code identifiers, frozen config names,
extracted paper text, and author-owned alternate artifacts retain their own
vocabulary and are not prose rename targets.

### Ablation Study

**State-only ablation**:
Ablation Configuration **3** — state (gradient-norm) awareness only drives Tier-2 quantization; server-side distillation is retained. This names what the arm configures.
_Avoid_: state-only-plus-distillation ablation, DynFed-core reference arm

**DynFed-style reference point**:
The analytical role of Configuration **3** as a comparison point reproducing
DynFed's core mechanism without its non-reproducible active teacher selection.
It is not a claim of direct DynFed benchmarking.
_Avoid_: DynFed-core reference arm, DynFed-style reference arm.

### Communication Comparison (Section 4, `sec:metrics_communication`)

The terms below distinguish the primary communication comparison from its scalar
descriptor and from the v2 study's separate paired budget. The criterion decision is
[ADR-0012](docs/adr/0012-formulation-selection-and-the-iso-byte-amendment.md).

**Accuracy-vs-cumulative-MB curve**:
The **primary** communication-efficiency comparison. Its MB axis is aggregate
bidirectional client--server traffic: each round sums the model download and measured
upload for every sampled client, then accumulates those totals across rounds. No free
parameters. It is the axis on which selection verdicts are read.
_Avoid_: MB per client, uploaded MB, or a single-round compression ratio as substitutes.

**Minimum common cumulative-MB budget**:
Where a scalar head-to-head is required: top-1 accuracy at $B = \min$ over the arms
compared of each arm's final cumulative MB. The budget is read off the data, chosen by
nobody, so it introduces no tunable comparison parameter.

“Iso-byte” is reserved for the paired per-seed v2 budget below; it is not a synonym for
the minimum common cumulative-MB budget.

**Bytes-to-target**:
Cumulative aggregate bidirectional client--server megabytes required to reach a
per-configuration target accuracy. It is a descriptor, never the primary verdict.
_Avoid_: bits-to-target-accuracy, bits-to-accuracy, and cumulative-MB-to-target.

**Rounds-to-target**:
A different quantity: the rounds term of the bytes-to-target product, not a
communication measure on its own. Pair it with per-round payload when interpreting it.
_Avoid_: rounds-to-converge.

### Run Identity

**Run identity**:
A run is identified by the matrix and resolved configuration that produced it,
not by the algorithm hook name alone. The owner is `fedmaq.core.run_identity`,
which builds and parses the canonical seven-segment output path and serializes
the analysis identity key from dataset, experiment group, algorithm config,
variant, heterogeneity parameter, formulation and seed. Callers must use this
module rather than slicing path segments inline; a failed canonical-path parse
is reported as a malformed run rather than silently assigned to a group.
[ADR-0009](docs/adr/0009-run-identity-and-analysis-scoping.md)

### Byte-Accounting Seam

The measurement layer the terms above are read off of. One canonical function
replaced four independent per-arm implementations. [ADR-0018](docs/adr/0018-byte-accounting-seam.md)

**Payload bytes**:
The pre-encoding size a hook hands to compression each round — codes plus scale,
before the content-sensitive step below. Logged per-hook as `payload_bytes`,
aggregated as `communication/round_payload_bytes`. Reproducible from config alone,
unlike measured bytes.
_Avoid_: transmitted bytes, wire bytes (both actually name measured bytes, below)

**Measured bytes**:
The single accounting primitive — `measure_bytes(payload: bytes) -> int` in
`transport.py` — every arm routes through for the transmitted, post-encoding size.
Content-sensitive (zlib for the compressed arms), which is why it feeds the
cumulative-MB terms above but cannot itself be recomputed offline from a logged
count alone.
_Avoid_: measured seam (the commit-message name for the refactor that produced
this function, not the quantity the function returns)

**Raw-payload capture**:
Opt-in per-round retention of a hook's actual payload bytes, not just their
measured length, gated by `experiment.telemetry.log_payloads` (default off).
Exists because measured bytes can't be re-scored against a hypothetical encoder
from a count alone — full offline reproducibility needs the payloads themselves.
Off by default: a multi-MB blob per client per round is a real cost over Flower's
simulated Ray channel.
Persistence is owned by `PayloadArchive` in `fedmaq.core.payload_archive`;
telemetry supplies the one per-round record call and does not own the filesystem
format or write policy.
_Avoid_: payload logging (ambiguous with ordinary telemetry, which is always on)

### Secondary Byte Axis

A second, parallel measurement alongside the seam above. [ADR-0020](docs/adr/0020-secondary-byte-axis.md)

**As-published coder**:
An arm's transport encoding measured the way its own source paper specifies,
rather than under the primary axis's held-constant zlib. DAdaQuant's paper
mandates 0-run-length encoding plus Elias omega coding. Logged as
`secondary_bytes_uploaded`
per client, aggregated as `communication/round_secondary_bytes`; absent
(not zero) on every other arm's rows.
_Avoid_: secondary encoder (names the mechanism, not the measurement it
produces — the two prior terms already claimed by the primary seam's own
vocabulary above)

**0-RLE + Elias omega**:
DAdaQuant's coder: runs of zero-valued quantization codes collapsed to a
run-length, each non-zero code's magnitude coded with Elias omega (a
universal code for positive integers with no fixed maximum, so it adapts as
`q_t` escalates). Implemented in `dadaquant_coder.py`, exercised only by
`DAdaQuantCompressionHook`. Round-trip tested directly — this is a coder,
not a heuristic estimate.
_Avoid_: entropy coding (the paper's broader category; this term names the
specific scheme actually implemented)

### Quantizer Unbiasedness

The rounding/normalization operator applied once a bit-width is chosen — downstream of
Precision Scaling above, which only picks the bit-width. One shared implementation
replaced FedPAQ's biased deterministic rounding and DAdaQuant's independent inline
copy of the correct stochastic one. [ADR-0019](docs/adr/0019-quantizer-unbiasedness-and-the-l-infinity-exception.md)

**Stochastic rounding**:
Unbiased dithered rounding (`floor(x)` w.p. `1-frac(x)`, else `ceil(x)`; `E[result] = x`
for any input) — `_stochastic_round` in `quantization.py`, shared by FedPAQ, plain
FedMAQ, `FedMAQPostProcessCompressionHook`'s `q>1` path, and DAdaQuant. Replaces
`np.round`, which is deterministic and biased.
_Avoid_: rounding (ambiguous with the deterministic `q≤1` sign-quantization branch,
which is unaffected and stays exact)

**l2 scale / l∞ scale**:
The two normalization divisors a quantizer's `_scale(d)` can return: `‖d‖₂`
(l2, `np.linalg.norm`) or `max|d|` (l∞). FedPAQ/FedMAQ's `q>1` path uses l2, matching
the manuscript's `Q_s(v_j) = ‖v‖₂·sgn(v_j)·ξ_j`; their `q≤1` sign branch and
DAdaQuant throughout stay l∞ — both unconditional per-path choices, not inherited
defaults. `FedMAQPostProcessCompressionHook` (error feedback) is the one `q>1`
exception that also keeps l∞: l2 diverges unboundedly there (ADR-0019).
_Avoid_: "the l2 switch" alone without naming which path — it does not apply
uniformly across every quantizer in this codebase.

**Compression RNG**:
The `np.random.Generator` a hook's stochastic rounding draws from, reseeded every
federated round (`StandardFit.fit()`) from `(seed, partition_id, server_round)` —
deliberately independent of the training RNG and of
`quantization_planner.py`'s dataloader-seeding stream. No hook falls back to an
unseeded default; reaching a stochastic branch with `rng=None` raises.
_Avoid_: assuming a compressor_hook's `rng` set at `client_fn` construction time is
what compress() actually draws from — Flower rebuilds `client_fn` fresh every round,
so only the per-round reseed matters.

### Freeze States

Three distinct states, three distinct gates, three distinct owners of what may be
claimed. They are reached in order and never collapse into one another; "frozen"
unqualified is not a state this project has. Each is declared only by the thesis
author, on evidence bound to one assurance envelope.

**Assurance envelope**:
The dated, content-hashed record that binds one candidate's revision vector
(`fedmaq-experiments`, `fedmaq-literature`, `fedmaq-manuscript`), baseline and
candidate commits, golden-harness and configuration identity, gate states, and
evidence references under a single SHA-256. Gate states are PASS, FAIL, BLOCKED
and INVALIDATED — a gate with missing, stale or conflicting evidence is never an
implicit PASS. Evidence belongs to exactly one envelope; a behavior- or
claim-affecting change creates a new one.
_Avoid_: **the freeze certificate** (that is `docs/freeze/source_manifest.json`'s
architecture hash over `conf/**`, `src/**`, `scripts/**`, `tests/**` and the
build files — one input to one gate, not the envelope)

**Pipeline freeze**:
The code, configuration and protocol are defensible and may be run. Requires the
static audit, repair and re-audit, independent review, local verification,
telemetry compatibility attestation, and the exact-commit JupyterHub golden gate.
It licenses dispatch; it licenses nothing about the runs' contents.
_Avoid_: treating pipeline freeze as evidence or results freeze.

**Evidence freeze**:
The prescribed campaign artifacts exist, are ingested, and carry hash provenance.
Requires pipeline freeze plus the campaign's own manifests and integrity checks.
It licenses analysis; it licenses no claim about what the analysis shows.
_Avoid_: treating a completed dispatch as evidence freeze — completion is not
provenance closure

**Results freeze**:
The manuscript's claims are supported by the frozen evidence. Requires evidence
freeze plus claim-support review. This is the only state that licenses a
scientific conclusion in prose.
_Avoid_: **the results are frozen** as shorthand for "the runs finished"

### Server-KD Repair Study

The v2 terminology is defined here; the protocol itself is
[ADR-0016](docs/adr/0016-v2-evaluation-protocol-and-advance-rule.md). Run state and results
live in Issues and evidence owners.

**FedMAQ-v2**:
The separately versioned exploratory study that changes only server-side KD, every non-KD
factor locked to v1. Not a successor: v1's artifacts, configs and protocol are preserved
untouched.
_Avoid_: FedMAQ 2.0, FedMAQ-next, **the v2 fix** (presupposes the outcome)

**Server-KD repair**:
One of the five candidate families, never combined within a pass. `no_kd` is the mandatory
anchor and is **itself a repair** — removing the mechanism is the trivial one, not the
absence of a treatment. This is what keeps a no-advance outcome a finding rather than an
empty result.
_Avoid_: **the winning repair** before a freeze artifact records one

**Candidate repair family**:
The five are **this thesis's own exploration design**, not five families drawn from the
literature — `no_kd` is one of them and no paper proposes it. Exactly one,
**quality-weighted server ensemble distillation**, has corroborated prior art; the
manuscript's *Ensemble Teacher Weighting* subsection is its survey home. The other
candidate dispositions are historical evidence, not a literature taxonomy.
_Avoid_: **the five repair families in the literature** (four of them are not)

**Paired per-seed byte budget** ($B^*_s$):
The v2 scalar head-to-head. For a paired comparison and seed $s$, the minimum of the two
same-seed terminal cumulative-byte budgets, defined independently per pair; both curves are
scored there by linear interpolation within each curve's own observed inclusive range, never
extrapolated. **v2-only.**

**Not the minimum common cumulative-MB budget above.** The two are distinct: one
is read across compared arms, while this v2-only budget is read per pair and seed
by interpolation. Do not substitute one for the other.
_Avoid_: iso-byte budget in general prose.

**Study 1 / Study 2**:
Planning labels only. They do **not** enter manuscript prose; use the existing v1
part-names and **the FedMAQ-v2 server-KD repair study** for v2.

**FedDistill vs. the v2 literature's "FedKD"**:
This project's baseline table names **Jeong et al. as FedDistill**. A v2 source calls
that work **FedKD**, colliding with this project's **FedKD baseline** and the v2
candidate **FedKT**. Cite by author and year whenever the v2 literature is intended.
